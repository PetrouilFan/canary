"""Model clients: OpenAI-compatible providers, retries, roles, mock, rate limits.

One client per role. Streaming is exposed as provider-shaped chunks so the API
layer can forward OpenAI-compatible SSE. Cross-process rate limiting uses a
flock-guarded token state file per provider (spec 4.11).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from canary.core import util
from canary.core.config import Config
from canary.core.observability import Log

RETRY_BACKOFF = (2.0, 4.0, 8.0)
RETRY_AFTER_CAP = 60.0


class ModelError(Exception):
    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None,
                 retry_after: float | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status
        self.retry_after = retry_after


class ContextOverflow(ModelError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str

    @classmethod
    def from_openai(cls, raw: dict) -> ToolCall:
        fn = raw.get("function") or {}
        return cls(
            id=raw.get("id") or f"call_{util.new_nonce()}",
            name=fn.get("name", ""),
            arguments=fn.get("arguments", "") or "",
        )

    def to_openai(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass
class ModelResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    finish_reason: str = "stop"
    model: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def prompt_tokens(self) -> int:
        return int(self.usage.get("prompt_tokens") or 0)

    @property
    def completion_tokens(self) -> int:
        return int(self.usage.get("completion_tokens") or 0)

    def message(self) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        return msg


def is_context_overflow(status: int | None, body: str) -> bool:
    if status not in (400, 413, 422, 500):
        return False
    text = (body or "").lower()
    markers = (
        "context length",
        "context_length",
        "maximum context",
        "max context",
        "context window",
        "too many tokens",
        "input is too long",
        "reduce the length",
        "sequence length",
    )
    return any(m in text for m in markers)


# ---------------------------------------------------------------------------
# cross-process rate limiting
# ---------------------------------------------------------------------------

class RateLimiter:
    """Fixed-window rpm/tpm + concurrency with a flock-guarded state file."""

    def __init__(self, config: Config, provider: str, limits: dict | None):
        self.config = config
        self.provider = provider or "default"
        limits = limits or {}
        self.rpm = int(limits.get("rpm") or 0)
        self.tpm = int(limits.get("tpm") or 0)
        self.concurrency = int(limits.get("concurrency") or 0)
        self.enabled = any((self.rpm, self.tpm, self.concurrency))
        self.dir = config.data_path / "ratelimit"
        self.path = self.dir / f"{self.provider}.json"
        self.lock_path = self.dir / f"{self.provider}.lock"
        self._active = False

    def _load(self, now: float) -> dict:
        state = util.read_json(self.path, None)
        if not isinstance(state, dict):
            state = {}
        if now - float(state.get("window_start") or 0) >= 60:
            state["window_start"] = now
            state["requests"] = 0
            state["tokens"] = 0
        state.setdefault("requests", 0)
        state.setdefault("tokens", 0)
        state.setdefault("active", [])
        # GC leaked concurrency slots from dead processes
        alive = []
        for entry in state["active"]:
            pid = entry.get("pid")
            started = entry.get("ts") or 0
            if pid and util.pid_alive(int(pid)) and now - started < 1800:
                alive.append(entry)
        state["active"] = alive
        return state

    def acquire(self, estimated_tokens: int) -> None:
        if not self.enabled:
            self._active = True
            return
        estimated_tokens = max(1, estimated_tokens)
        now = time.time()
        lock = util.FileLock(self.lock_path, op="ratelimit", wait=False)
        if not lock.acquire():
            raise ModelError("rate limiter busy", retryable=True)
        try:
            state = self._load(now)
            if self.rpm and state["requests"] >= self.rpm:
                raise ModelError(
                    f"provider {self.provider}: rpm limit {self.rpm} reached",
                    retryable=True,
                    status=429,
                )
            if self.tpm and state["tokens"] + estimated_tokens > self.tpm:
                raise ModelError(
                    f"provider {self.provider}: tpm limit {self.tpm} reached",
                    retryable=True,
                    status=429,
                )
            if self.concurrency and len(state["active"]) >= self.concurrency:
                raise ModelError(
                    f"provider {self.provider}: concurrency limit {self.concurrency} reached",
                    retryable=True,
                    status=429,
                )
            state["requests"] += 1
            state["tokens"] += estimated_tokens
            state["active"].append({"pid": os.getpid(), "ts": now})
            util.atomic_write_json(self.path, state)
            self._active = True
        finally:
            lock.release()

    def record_usage(self, actual_tokens: int) -> None:
        if not self.enabled or not actual_tokens:
            return
        lock = util.FileLock(self.lock_path, op="ratelimit", wait=False)
        if not lock.acquire():
            return
        try:
            state = self._load(time.time())
            state["tokens"] = max(state.get("tokens", 0), 0) + int(actual_tokens)
            util.atomic_write_json(self.path, state)
        finally:
            lock.release()

    def release(self) -> None:
        if not self.enabled or not self._active:
            return
        self._active = False
        lock = util.FileLock(self.lock_path, op="ratelimit", wait=False)
        if not lock.acquire():
            return
        try:
            state = self._load(time.time())
            pid = os.getpid()
            state["active"] = [e for e in state["active"] if e.get("pid") != pid]
            util.atomic_write_json(self.path, state)
        finally:
            lock.release()


# ---------------------------------------------------------------------------
# mock provider
# ---------------------------------------------------------------------------

class MockModel:
    """Deterministic offline model. Script entries: str | dict | ModelResponse.

    A dict may contain 'content' and/or 'tool_calls' (list of
    {name, arguments}). When the script is exhausted the last text entry
    repeats; an empty script answers 'ok'.
    """

    def __init__(self, script: list | None = None, *, model: str = "mock"):
        self.script = list(script or [])
        self.model = model
        self.calls: list[list[dict]] = []

    def _next(self) -> ModelResponse:
        if not self.script:
            return ModelResponse(content="ok", model=self.model)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, ModelResponse):
            return item
        if isinstance(item, str):
            return ModelResponse(content=item, model=self.model)
        if isinstance(item, dict):
            tool_calls = []
            for tc in item.get("tool_calls") or []:
                if isinstance(tc, ToolCall):
                    tool_calls.append(tc)
                else:
                    tool_calls.append(
                        ToolCall(
                            id=tc.get("id") or f"call_{util.new_nonce()}",
                            name=tc.get("name", ""),
                            arguments=json.dumps(tc.get("arguments") or {})
                            if not isinstance(tc.get("arguments"), str)
                            else tc["arguments"],
                        )
                    )
            return ModelResponse(
                content=item.get("content") or "",
                tool_calls=tool_calls,
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                finish_reason=item.get("finish_reason")
                or ("tool_calls" if tool_calls else "stop"),
                model=self.model,
            )
        return ModelResponse(content=str(item), model=self.model)

    def call(
        self, messages: list[dict], tools: list[dict] | None = None, **_: Any
    ) -> ModelResponse:
        self.calls.append(messages)
        resp = self._next()
        if not resp.usage:
            prompt = max(1, len(json.dumps(messages)) // 4)
            completion = max(1, (len(resp.content) + 20 * len(resp.tool_calls)) // 4)
            resp.usage = {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            }
        if resp.tool_calls and tools is not None:
            allowed = {t.get("function", {}).get("name") for t in tools}
            resp.tool_calls = [tc for tc in resp.tool_calls if tc.name in allowed]
        return resp

    def stream(
        self, messages: list[dict], tools: list[dict] | None = None, **kwargs: Any
    ) -> Iterator[dict]:
        resp = self.call(messages, tools, **kwargs)
        chunk = {
            "id": f"chatcmpl-mock-{util.new_nonce()}",
            "object": "chat.completion.chunk",
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": resp.content},
                    "finish_reason": None,
                }
            ],
        }
        yield chunk
        if resp.tool_calls:
            for i, tc in enumerate(resp.tool_calls):
                yield {
                    "id": f"chatcmpl-mock-{util.new_nonce()}",
                    "object": "chat.completion.chunk",
                    "model": self.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": i,
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {"name": tc.name, "arguments": tc.arguments},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
        yield {
            "id": f"chatcmpl-mock-{util.new_nonce()}",
            "object": "chat.completion.chunk",
            "model": self.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": resp.finish_reason}],
            "usage": resp.usage,
        }


# ---------------------------------------------------------------------------
# HTTP client (OpenAI-compatible)
# ---------------------------------------------------------------------------

class HTTPModel:
    def __init__(self, role: str, entry: dict, config: Config, log: Log | None = None):
        self.role = role
        self.entry = entry
        self.config = config
        self.log = log
        self.provider = entry.get("provider") or "openai"
        self.model = entry.get("model") or ""
        self.base_url = (entry.get("base_url") or "").rstrip("/")
        self.context_length = int(entry.get("context_length") or 32768)
        self.api_key_env = entry.get("api_key_env") or "HARNESS_MODEL_API_KEY"
        self.default_timeout = float(config.get("model_timeout_s", 120))
        self.params: dict[str, Any] = {}
        for key in ("temperature", "top_p", "max_tokens"):
            if entry.get(key) is not None:
                self.params[key] = entry[key]
        if role == "compression" and "temperature" not in self.params:
            self.params["temperature"] = 0.0
        limits = (
            entry.get("limits")
            or (config.models.get("providers", {}) or {}).get(self.provider)
            or {}
        )
        if isinstance(limits, dict):
            limits = {k: limits.get(k) for k in ("rpm", "tpm", "concurrency")}
        self.limiter = RateLimiter(config, self.provider, limits)
        self._client = httpx.Client(
            timeout=httpx.Timeout(
                connect=10.0, read=self.default_timeout, write=30.0, pool=5.0
            )
        )

    # -- helpers -------------------------------------------------------------

    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key and self.provider != "mock":
            raise ModelError(f"missing API key: environment variable {self.api_key_env} is not set")
        return key

    def _payload(self, messages: list[dict], tools: list[dict] | None, stream: bool) -> dict:
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        payload.update(self.params)
        return payload

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key()}",
            "Content-Type": "application/json",
        }

    def _estimate(self, messages: list[dict]) -> int:
        return max(1, len(json.dumps(messages, default=str)) // 4)

    # -- non-streaming -------------------------------------------------------

    def call(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        timeout_s: float | None = None,
    ) -> ModelResponse:
        payload = self._payload(messages, tools, stream=False)
        last_error: ModelError | None = None
        for attempt in range(len(RETRY_BACKOFF) + 1):
            self.limiter.acquire(self._estimate(messages))
            try:
                resp = self._client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._headers(),
                    timeout=timeout_s or self.default_timeout,
                )
                if resp.status_code >= 400:
                    body = resp.text
                    if resp.status_code == 429:
                        retry_after = resp.headers.get("Retry-After")
                        numeric = (retry_after or "").replace(".", "", 1).isdigit()
                        wait = min(float(retry_after), RETRY_AFTER_CAP) if numeric else None
                        last_error = ModelError(
                            f"provider 429: {body[:400]}",
                            retryable=True,
                            status=429,
                            retry_after=wait,
                        )
                    elif is_context_overflow(resp.status_code, body):
                        raise ContextOverflow(
                            f"context overflow: {body[:300]}", status=resp.status_code
                        )
                    elif resp.status_code >= 500:
                        last_error = ModelError(
                            f"provider {resp.status_code}: {body[:400]}",
                            retryable=True,
                            status=resp.status_code,
                        )
                    else:
                        raise ModelError(
                            f"provider {resp.status_code}: {body[:600]}",
                            status=resp.status_code,
                        )
                else:
                    data = resp.json()
                    um = data.get("usage") or {}
                    self.limiter.record_usage(int(um.get("total_tokens") or 0))
                    choice = (data.get("choices") or [{}])[0]
                    message = choice.get("message") or {}
                    tool_calls = [
                        ToolCall.from_openai(tc) for tc in message.get("tool_calls") or []
                    ]
                    return ModelResponse(
                        content=message.get("content") or "",
                        tool_calls=tool_calls,
                        usage=um,
                        finish_reason=choice.get("finish_reason") or "stop",
                        model=data.get("model") or self.model,
                        raw=data,
                    )
            except (
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.WriteTimeout,
                httpx.PoolTimeout,
                httpx.RemoteProtocolError,
            ) as exc:
                last_error = ModelError(f"transport error: {exc}", retryable=True)
            except (httpx.HTTPError, ValueError) as exc:
                last_error = ModelError(f"model call failed: {exc}", retryable=False)
            finally:
                self.limiter.release()

            if last_error is None or not last_error.retryable or attempt >= len(RETRY_BACKOFF):
                break
            wait = (
                last_error.retry_after
                if last_error.retry_after is not None
                else RETRY_BACKOFF[attempt]
            )
            if self.log:
                self.log.warn(
                    "model_retry",
                    role=self.role,
                    attempt=attempt + 1,
                    wait_s=wait,
                    error=str(last_error),
                )
            time.sleep(wait)

        raise last_error or ModelError("model call failed")

    # -- streaming -----------------------------------------------------------

    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        timeout_s: float | None = None,
    ) -> Iterator[dict]:
        """Yield provider chunks (OpenAI shape). Retries only before first chunk."""
        payload = self._payload(messages, tools, stream=True)
        last_error: ModelError | None = None
        for attempt in range(len(RETRY_BACKOFF) + 1):
            self.limiter.acquire(self._estimate(messages))
            started = False
            try:
                with self._client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._headers(),
                    timeout=timeout_s or self.default_timeout,
                ) as resp:
                    if resp.status_code >= 400:
                        body = resp.read().decode("utf-8", errors="replace")
                        if resp.status_code == 429:
                            last_error = ModelError(
                                f"provider 429: {body[:400]}", retryable=True, status=429
                            )
                        elif is_context_overflow(resp.status_code, body):
                            raise ContextOverflow(
                                f"context overflow: {body[:300]}", status=resp.status_code
                            )
                        elif resp.status_code >= 500:
                            last_error = ModelError(
                                f"provider {resp.status_code}: {body[:400]}",
                                retryable=True,
                            )
                        else:
                            raise ModelError(
                                f"provider {resp.status_code}: {body[:600]}",
                                status=resp.status_code,
                            )
                    else:
                        for line in resp.iter_lines():
                            if not line:
                                continue
                            if line.startswith("data:"):
                                raw = line[5:].strip()
                                if raw == "[DONE]":
                                    return
                                try:
                                    chunk = json.loads(raw)
                                except json.JSONDecodeError:
                                    continue
                                started = True
                                um = chunk.get("usage")
                                if um:
                                    self.limiter.record_usage(
                                        int(um.get("total_tokens") or 0)
                                    )
                                yield chunk
                        return
            except (
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.WriteTimeout,
                httpx.PoolTimeout,
                httpx.RemoteProtocolError,
            ) as exc:
                if started:
                    raise ModelError(f"stream interrupted: {exc}") from exc
                last_error = ModelError(f"transport error: {exc}", retryable=True)
            finally:
                self.limiter.release()

            if (
                last_error is None
                or not last_error.retryable
                or attempt >= len(RETRY_BACKOFF)
                or started
            ):
                break
            wait = (
                last_error.retry_after
                if last_error.retry_after is not None
                else RETRY_BACKOFF[attempt]
            )
            time.sleep(wait)

        raise last_error or ModelError("model stream failed")

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


class ModelClient:
    """Facade: HTTP provider or mock, addressed by role."""

    def __init__(self, role: str, entry: dict, config: Config, log: Log | None = None):
        self.role = role
        self.entry = entry
        self.config = config
        provider = entry.get("provider") or "openai"
        if provider == "mock":
            script = entry.get("script")
            self.backend = MockModel(script=script)
            self.context_length = int(entry.get("context_length") or 32768)
        else:
            self.backend = HTTPModel(role, entry, config, log)
            self.context_length = self.backend.context_length
        self.provider = provider

    @property
    def model(self) -> str:
        return getattr(self.backend, "model", "")

    def call(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        timeout_s: float | None = None,
    ) -> ModelResponse:
        return self.backend.call(messages, tools, timeout_s=timeout_s)

    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        timeout_s: float | None = None,
    ) -> Iterator[dict]:
        return self.backend.stream(messages, tools, timeout_s=timeout_s)

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if close:
            close()


class Models:
    """Role registry: client_for(role), context length, generation params."""

    def __init__(self, config: Config, log: Log | None = None):
        self.config = config
        self.log = log
        self._clients: dict[str, ModelClient] = {}

    def client_for(self, role: str) -> ModelClient:
        if role not in self._clients:
            entry = self.config.role(role)
            self._clients[role] = ModelClient(role, dict(entry), self.config, self.log)
        return self._clients[role]

    def context_length(self, role: str) -> int:
        return self.client_for(role).context_length

    def mock(self, script: list | None = None, role: str = "health") -> MockModel:
        return MockModel(script=script, model=f"mock-{role}")

    def close(self) -> None:
        for client in self._clients.values():
            client.close()
        self._clients.clear()
