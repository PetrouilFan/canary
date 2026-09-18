"""Agent loop: turns, model calls, tool dispatch, workers, cancellation.

Spec §4.1. One turn is one external request through to a final assistant
text. The loop is deliberately small: assemble context, call the model,
dispatch tools, repeat until the model stops calling tools or a limit is hit.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import util
from .config import Config
from .context import ContextEngine
from .governance import Governance
from .health import Health
from .jobs import JobLimitError, Jobs
from .memory import Memory
from .models import ContextOverflow, ModelError, ModelResponse, ToolCall
from .observability import Log
from .session import Session, SessionManager
from .tools import ToolRegistry


class SessionBusy(RuntimeError):
    """Raised when a session already has a turn in flight and fail_fast is set."""


@dataclass
class _Turn:
    session: Session
    role: str = "main"
    on_event: Callable[[dict], None] | None = None
    messages: list[dict] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)
    model_calls: int = 0
    tool_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    completion_tokens: int = 0
    start: float = 0.0
    last_nudge: float = 0.0
    tool_failures: dict[str, int] = field(default_factory=dict)
    compacted: bool = False
    limit_hit: str = ""
    error: str = ""
    cancelled: bool = False
    workers: list[str] = field(default_factory=list)
    budget_baseline: int = 0


def _parse_args(raw: str) -> tuple[dict, str]:
    if not raw or not raw.strip():
        return {}, ""
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        return {}, str(exc)
    if not isinstance(value, dict):
        return {}, "arguments must be a JSON object"
    return value, ""


def _looks_like_correction(text: str) -> bool:
    lowered = text.strip().lower()
    prefixes = (
        "no,",
        "no.",
        "nope",
        "wrong",
        "that's wrong",
        "thats wrong",
        "incorrect",
        "actually",
        "stop",
        "don't",
        "dont",
        "not what",
        "i said",
        "you missed",
        "redo",
        "try again",
        "fix",
    )
    return any(lowered.startswith(prefix) for prefix in prefixes)


class Agent:
    """One agent instance: shares the state store, owns a context window."""

    def __init__(
        self,
        config: Config | dict | None = None,
        *,
        state_path: str | os.PathLike[str] | None = None,
        ephemeral: bool = False,
        self_modify: bool | None = None,
        workspace: str | os.PathLike[str] | None = None,
        is_worker: bool = False,
        parent: Agent | None = None,
        model: str | None = None,
        tools: list[str] | None = None,
    ):
        if isinstance(config, Config):
            base = config
            if workspace is not None and base.workspace is None:
                base = Config(
                    state_path=base.state_path,
                    root=base.root,
                    ephemeral=base.ephemeral,
                    overrides=base.overrides,
                    load_files=True,
                    workspace=workspace,
                )
            self.config = base
        elif isinstance(config, dict):
            self.config = Config(
                values=config,
                state_path=state_path,
                ephemeral=ephemeral,
                workspace=workspace,
            )
        else:
            self.config = Config(
                state_path=state_path, ephemeral=ephemeral, workspace=workspace
            )
        if self_modify is not None:
            self.config.set("self_modify", bool(self_modify))
        if model:
            self.default_model = model
        else:
            self.default_model = "main"

        self.parent = parent
        self.is_worker = bool(is_worker or parent is not None)
        self.spawn_depth = 1 if self.is_worker else 0
        self._local = threading.local()
        self._worker_seq = 0
        self._worker_lock = threading.Lock()
        self._closed = False
        self.last_tool_error = ""
        self.last_usage: dict[str, Any] = {}
        self.last_total_tokens = 0
        self.total_tokens = 0
        self.last_result: dict[str, Any] = {}

        if parent is not None:
            self.log = parent.log
            self.models = parent.models
            self.governance = parent.governance
            self.memory = parent.memory
            self.sessions = parent.sessions
            self.jobs = parent.jobs
            self.health = parent.health
        else:
            self.log = Log(self.config)
            self.models = self._make_models()
            self.governance = Governance(self.config, self.log)
            self.memory = Memory(self.config, self.log)
            self.sessions = SessionManager(self.config, self.log)
            self.jobs = Jobs(self.config, self.log)
            self.jobs.recover()
            self.health = None
            if self.config.serve_mode and not self.config.ephemeral:
                self.health = Health(self.config, self.log, agent=self)
                ok, reason = self.health.check_ready()
                self.health.set_ready(ok, reason)

        self._ensure_dirs()
        self.tools = ToolRegistry(self)
        if self.is_worker:
            self.tools.unregister("task")
        if tools is not None:
            self.tools.restrict(tools)
        self.ctx = ContextEngine(
            self.config, self.log, memory=self.memory
        )

    # -- construction helpers ------------------------------------------------

    def _make_models(self) -> Any:
        from .models import Models

        return Models(self.config, self.log)

    def _compression(self) -> Any:
        try:
            return self.models.client_for("compression")
        except Exception:  # noqa: BLE001 - compression is optional
            return None

    def _ensure_dirs(self) -> None:
        paths = [
            self.config.state_path,
            self.config.data_path,
            self.config.data_path / "tmp",
            self.config.data_path / "memory_index",
            self.config.state_path / "logs",
            self.config.state_path / "memory",
            self.config.state_path / "memory" / "entries",
            self.config.state_path / "extensions",
            self.config.state_path / "evals",
            self.config.workspace_path,
        ]
        for path in paths:
            util.ensure_dir(path)

    # -- properties ----------------------------------------------------------

    @property
    def session_id(self) -> str | None:
        turn = getattr(self._local, "turn", None)
        if turn is not None:
            return turn.session.id
        return getattr(self._local, "session_id", None)

    @session_id.setter
    def session_id(self, value: str | None) -> None:
        self._local.session_id = value

    @property
    def agent_id(self) -> str:
        return str(self.config.get("agent.id") or "")

    def tool_specs(self) -> list[dict]:
        return self.tools.specs()

    def info(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "release_id": self.config.release_id,
            "commit_sha": self.config.commit_sha,
            "state_path": str(self.config.state_path),
            "root": str(self.config.root) if self.config.root else None,
            "ephemeral": self.config.ephemeral,
            "self_modify": bool(self.config.get("self_modify")),
            "worker": self.is_worker,
            "session_id": self.session_id,
            "tools": self.tools.names(),
            "disabled_tools": self.tools.disabled(),
            "models": self.config.role_names(),
            "last_total_tokens": self.last_total_tokens,
        }

    # -- sessions/turns ------------------------------------------------------

    def _session_for(self, session_id: str | None) -> tuple[Session, bool]:
        if session_id:
            return self.sessions.get_or_create(session_id), False
        sid = f"stateless-{uuid.uuid4().hex[:10]}"
        return self.sessions.create(session_id=sid, name="stateless"), True

    def _drop_stateless(self, session: Session) -> None:
        import shutil

        shutil.rmtree(session.dir, ignore_errors=True)

    def run(
        self,
        message: str,
        *,
        session_id: str | None = None,
        model: str | None = None,
        on_event: Callable[[dict], None] | None = None,
        fail_fast: bool = False,
    ) -> str:
        if self._budget_exhausted():
            return "[error] daily token budget exhausted"
        session, stateless = self._session_for(session_id)
        timeout = float(self.config.get("turn_timeout_s", 1800)) or 1800.0
        lock = session.lock("turn", wait=not fail_fast, timeout=timeout)
        if not lock.acquire():
            if stateless:
                self._drop_stateless(session)
            raise SessionBusy(f"session {session.id} already has a turn in flight")
        try:
            return self._run_turn(
                session, message, model or self.default_model, on_event
            )
        finally:
            lock.release()
            if stateless:
                self._drop_stateless(session)

    # -- the loop ------------------------------------------------------------

    def _run_turn(
        self,
        session: Session,
        message: str,
        role: str,
        on_event: Callable[[dict], None] | None,
    ) -> str:
        turn = _Turn(
            session=session,
            role=role,
            on_event=on_event,
            start=time.monotonic(),
        )
        self._local.turn = turn
        session.begin_turn()
        session.clear_cancel()
        self._cancelled = threading.Event()
        self.jobs.begin_turn()
        state = "idle"
        try:
            session.append_message("user", message)
            injections = session.drain_inbox()
            summary = session.get("summary", "") or ""
            specs = self.tool_specs()
            turn.messages = self.ctx.build(
                session,
                user_message=message,
                tools=specs,
                injections=injections,
                summary=summary,
            )
            self._emit("turn_start", session_id=session.id, role=role)
            self._loop(turn, specs)
            state = self._turn_state(turn)
        except Exception as exc:  # noqa: BLE001 - a turn must never kill the server
            state = "error"
            turn.error = turn.error or f"{type(exc).__name__}: {exc}"
            self.log.error("turn_failed", error=str(exc), session_id=session.id)
        finally:
            for worker_session in turn.workers:
                try:
                    worker = self.sessions.load(worker_session)
                    if worker is not None:
                        worker.request_cancel()
                except Exception:  # noqa: BLE001
                    pass
            session.set_context_usage(self.ctx.count_tokens(turn.messages))
            session.end_turn(state)
            self._log_turn(turn, session, state)
            self._post_turn(session, message, turn)
            self._local.turn = None
            self.last_result = {
                "session_id": session.id,
                "status": state,
                "model_calls": turn.model_calls,
                "tool_calls": turn.tool_calls,
                "tokens_in": turn.tokens_in,
                "tokens_out": turn.tokens_out,
                "error": turn.error,
                "cancelled": turn.cancelled,
            }
        text = self._final_text(turn, state)
        self._emit("turn_end", session_id=session.id, status=state, text=text)
        return text

    def _loop(self, turn: _Turn, specs: list[dict]) -> None:
        config = self.config
        max_model_calls = int(config.get("max_model_calls_per_turn", 64))
        max_tool_calls = int(config.get("max_tool_calls_per_turn", 128))
        turn_timeout = float(config.get("turn_timeout_s", 1800)) or 1800.0
        budget_daily = self._budget_daily()
        budget_enforced = budget_daily > 0 and bool(
            config.get("budget.enforce", False)
        )
        turn.budget_baseline = self._tokens_today_total()
        while True:
            if self._is_cancelled(turn):
                turn.cancelled = True
                turn.limit_hit = "cancelled"
                return
            if time.monotonic() - turn.start >= turn_timeout:
                turn.limit_hit = "turn timeout"
                return
            if turn.model_calls >= max_model_calls:
                turn.limit_hit = "model call limit"
                self._append_system(turn, "[system] model call limit reached; stopping")
                return
            if budget_enforced and self._budget_used(turn) >= budget_daily:
                turn.limit_hit = "token budget"
                self._append_system(turn, "[system] token budget reached; stopping")
                self.log.info(
                    "budget_stop",
                    session_id=turn.session.id,
                    model_calls=turn.model_calls,
                    used=self._budget_used(turn),
                    daily=budget_daily,
                )
                return
            self._maybe_prune(turn)
            self._maybe_nudge(turn)
            turn.model_calls += 1
            try:
                response = self._call_model(turn, specs)
            except ContextOverflow:
                if turn.compacted:
                    turn.error = "context overflow after compression"
                    return
                self._forced_compress(turn)
                continue
            except ModelError as exc:
                turn.error = str(exc)
                return
            self._record_usage(turn, response)
            if response.content:
                turn.text_parts.append(response.content)
                self._emit("assistant", content=response.content)
            self._append_assistant(turn, response)
            if not response.tool_calls:
                return
            limit_hit = self._dispatch_tools(turn, response.tool_calls, max_tool_calls)
            if limit_hit:
                turn.limit_hit = limit_hit
                return

    def _call_model(self, turn: _Turn, specs: list[dict]) -> ModelResponse:
        client = self.models.client_for(turn.role)
        timeout = float(self.config.get("model_timeout_s", 120)) or 120.0
        if turn.on_event is None:
            return client.call(turn.messages, tools=specs, timeout_s=timeout)
        return self._stream_model(client, turn, specs, timeout)

    def _stream_model(
        self, client: Any, turn: _Turn, specs: list[dict], timeout: float
    ) -> ModelResponse:
        content = ""
        tool_slots: dict[int, dict] = {}
        finish = ""
        usage: dict[str, Any] = {}
        iterator = client.stream(turn.messages, tools=specs, timeout_s=timeout)
        for chunk in iterator:
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                if piece:
                    content += piece
                    self._emit("delta", content=piece)
                for call in delta.get("tool_calls") or []:
                    index = int(call.get("index") or 0)
                    slot = tool_slots.setdefault(
                        index, {"id": "", "name": "", "arguments": ""}
                    )
                    if call.get("id"):
                        slot["id"] = call["id"]
                    fn = call.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
        tool_calls = [
            ToolCall(id=slot["id"] or f"call_{index}", name=slot["name"],
                     arguments=slot["arguments"])
            for index, slot in sorted(tool_slots.items())
        ]
        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage or {},
            finish_reason=finish,
            model=getattr(client, "model", ""),
            raw={},
        )

    def _record_usage(self, turn: _Turn, response: ModelResponse) -> None:
        usage = response.usage or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        total = int(usage.get("total_tokens") or (prompt + completion))
        turn.tokens_in += prompt
        turn.tokens_out += completion
        turn.completion_tokens += completion
        self.last_usage = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        }
        if total:
            self.last_total_tokens = total
            self.total_tokens += total

    # -- message plumbing ----------------------------------------------------

    def _append_assistant(self, turn: _Turn, response: ModelResponse) -> None:
        calls = [call.to_openai() for call in response.tool_calls]
        turn.session.append_message(
            "assistant", response.content or "", tool_calls=calls or None
        )
        message: dict[str, Any] = {"role": "assistant", "content": response.content or ""}
        if calls:
            message["tool_calls"] = calls
        turn.messages.append(message)

    def _append_system(self, turn: _Turn, text: str) -> None:
        turn.messages.append({"role": "system", "content": text})

    def _dispatch_tools(
        self, turn: _Turn, calls: list[ToolCall], max_tool_calls: int
    ) -> str:
        for call in calls:
            if self._is_cancelled(turn):
                turn.cancelled = True
                return "cancelled"
            if turn.tool_calls >= max_tool_calls:
                self._append_system(
                    turn, "[system] tool call limit reached; stopping"
                )
                return "tool call limit"
            turn.tool_calls += 1
            args, parse_error = _parse_args(call.arguments)
            if parse_error:
                result = f"error: invalid tool arguments JSON ({parse_error})"
            else:
                result = self._run_tool(turn, call.name, args)
            turn.session.append_message(
                "tool", result, tool_call_id=call.id, name=call.name
            )
            turn.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": result,
                }
            )
            self._emit("tool", name=call.name, result=util.truncate(result, 2000))
        return ""

    def _run_tool(self, turn: _Turn, name: str, args: dict) -> str:
        result = self.tools.call(name, args)
        failed = result.startswith("error:") or result.startswith("[tool reload failed")
        if failed:
            count = turn.tool_failures.get(name, 0) + 1
            turn.tool_failures[name] = count
            if count >= 2:
                result = (
                    f"{result}\n[system] tool '{name}' has failed {count} times "
                    "this turn; stop retrying it and try another approach"
                )
        impact = self.tools.impact().get(name, {}).get("impact", "free")
        fields: dict[str, Any] = {"impact": impact}
        if impact in ("high", "unbounded"):
            fields["args"] = json.dumps(args, ensure_ascii=False, sort_keys=True)[:200]
        self.log.event(
            "tool_call", tool=name, session_id=turn.session.id, failed=failed, **fields
        )
        return result

    # -- context maintenance -------------------------------------------------

    def _maybe_prune(self, turn: _Turn) -> None:
        ratio = self.ctx.usage_ratio(turn.messages)
        if ratio <= self.ctx.threshold:
            return
        messages, report = self.ctx.prune(
            turn.messages, session=turn.session, user_message=self._user_text(turn)
        )
        turn.messages = messages
        self._emit(
            "prune",
            units=report.pruned_units,
            tokens_before=report.tokens_before,
            tokens_after=report.tokens_after,
        )

    def _maybe_nudge(self, turn: _Turn) -> None:
        ratio = self.ctx.usage_ratio(turn.messages)
        triggered = [float(n) for n in self.ctx.nudges if ratio >= float(n)]
        if not triggered:
            return
        level = max(triggered)
        if level <= turn.last_nudge:
            return
        turn.last_nudge = level
        nudge = self.ctx.nudge_for(ratio)
        if nudge:
            self._append_system(turn, nudge)

    def _forced_compress(self, turn: _Turn) -> None:
        messages, report = self.ctx.compress(
            turn.messages,
            focus="context overflow; keep decisions, paths, open errors",
            session=turn.session,
            model_client=self._compression(),
        )
        turn.messages = messages
        turn.compacted = True
        self._persist_summary(turn, messages)
        self._emit(
            "compressed",
            tokens_before=report.tokens_before,
            tokens_after=report.tokens_after,
            forced=True,
        )

    def _persist_summary(self, turn: _Turn, messages: list[dict]) -> None:
        for message in messages:
            content = message.get("content") or ""
            if isinstance(content, str) and content.startswith("[context summary]"):
                turn.session.set("summary", content)
                return

    def _user_text(self, turn: _Turn) -> str:
        for message in reversed(turn.messages):
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    return content
        return ""

    # -- cancellation --------------------------------------------------------

    def _is_cancelled(self, turn: _Turn) -> bool:
        local = getattr(self, "_cancelled", None)
        if local is not None and local.is_set():
            return True
        try:
            return turn.session.cancel_requested()
        except Exception:  # noqa: BLE001
            return False

    def cancel(self, session_id: str | None = None) -> bool:
        """Cancel the local turn and/or request cancellation for a session."""
        local = getattr(self, "_cancelled", None)
        if local is not None:
            local.set()
        if session_id:
            session = self.sessions.load(session_id)
            if session is None:
                return False
            session.request_cancel()
            return True
        return True

    # -- workers -------------------------------------------------------------

    def spawn_worker(
        self,
        prompt: str,
        name: str | None = None,
        tools: list[str] | None = None,
        model: str | None = None,
        timeout_s: float = 900.0,
    ) -> str:
        if self.is_worker:
            return "error: workers cannot spawn further workers (depth limited to 1)"
        with self._worker_lock:
            self._worker_seq += 1
            seq = self._worker_seq
        parent_session = self.session_id or "standalone"
        worker_session = f"worker:{parent_session}:{seq}"
        worker = Agent(
            config=self.config,
            is_worker=True,
            parent=self,
            tools=tools,
        )
        if name:
            session = self.sessions.get_or_create(worker_session)
            session.set("name", name)
        box: dict[str, Any] = {}

        def target() -> None:
            try:
                box["text"] = worker.run(
                    prompt, session_id=worker_session, model=model or "main"
                )
            except Exception as exc:  # noqa: BLE001 - reported back to the parent
                box["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                worker.close()

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout_s)
        turn = getattr(self._local, "turn", None)
        if turn is not None:
            turn.workers.append(worker_session)
        if thread.is_alive():
            worker.cancel(worker_session)
            thread.join(5.0)
            return f"error: worker timed out after {timeout_s:.0f}s (session {worker_session})"
        if turn is not None:
            turn.tokens_in += int(worker.last_usage.get("prompt_tokens") or 0)
            turn.tokens_out += int(worker.last_usage.get("completion_tokens") or 0)
        if "error" in box:
            return f"error: worker failed: {box['error']}"
        return str(box.get("text", ""))

    # -- harness surface used by tools ---------------------------------------

    def propose(self, patch: str, motivation: str | None = None) -> Any:
        if self.health is None:
            return "error: self-modification is unavailable in embedded mode"
        if not self.config.get("self_modify"):
            return "error: self_modification is disabled for this copy"
        return self.health.publish(
            patch, motivation=motivation, session_id=self.session_id
        )

    def revert(self, release_id: str | None = None) -> Any:
        if self.health is None:
            return "error: releases are unavailable in embedded mode"
        if not self.config.get("self_modify"):
            return "error: self_modification is disabled for this copy"
        return self.health.revert(release_id, session_id=self.session_id)

    def compress_current(self, focus: str | None = None) -> Any:
        turn = getattr(self._local, "turn", None)
        if turn is None:
            return "error: no active turn to compress"
        messages, report = self.ctx.compress(
            turn.messages,
            focus=focus or "",
            session=turn.session,
            model_client=self._compression(),
        )
        turn.messages = messages
        self._persist_summary(turn, messages)
        self._emit(
            "compressed",
            tokens_before=report.tokens_before,
            tokens_after=report.tokens_after,
            forced=False,
        )
        return {
            "tokens_before": report.tokens_before,
            "tokens_after": report.tokens_after,
            "retained": report.retained,
        }

    def schedule_eval_job(self, tag: str | None = None) -> Any:
        return self._schedule_job("eval", ["eval", "--json"], tag=tag)

    def schedule_diagnose_job(self) -> Any:
        return self._schedule_job("diagnose", ["diagnose", "--json"])

    def _schedule_job(self, job_type: str, argv: list[str], tag: str | None = None) -> Any:
        command = [sys.executable, "-m", "canary", *argv]
        if tag:
            command += ["--tag", tag]
        try:
            return self.jobs.spawn(
                command,
                shell=False,
                type=job_type,
                name=job_type,
                session_id=self.session_id,
                env=self._job_env(),
            )
        except JobLimitError as exc:
            return f"error: {exc}"

    def _job_env(self) -> dict[str, str]:
        env = {
            "CANARY_ROOT": str(self.config.root) if self.config.root else "",
            "HARNESS_STATE_PATH": str(self.config.state_path),
            "HARNESS_RELEASE_ID": self.config.release_id,
            "HARNESS_COMMIT_SHA": self.config.commit_sha,
        }
        if self.config.root:
            env["PYTHONPATH"] = str(self.config.root) + os.pathsep + env.get(
                "PYTHONPATH", ""
            )
        return {key: value for key, value in env.items() if value}

    # -- turn helpers --------------------------------------------------------

    def _emit(self, event_type: str, **fields: Any) -> None:
        turn = getattr(self._local, "turn", None)
        if turn is None or turn.on_event is None:
            return
        try:
            turn.on_event({"type": event_type, **fields})
        except Exception:  # noqa: BLE001 - subscribers must never break a turn
            pass

    def _turn_state(self, turn: _Turn) -> str:
        if turn.cancelled:
            return "aborted: cancelled"
        if turn.error:
            return "error"
        if turn.limit_hit:
            return f"aborted: {turn.limit_hit}"
        return "idle"

    def _final_text(self, turn: _Turn, state: str) -> str:
        text = "\n".join(part for part in turn.text_parts if part).strip()
        if turn.error:
            if text:
                return f"{text}\n[error] {turn.error}"
            return f"[error] {turn.error}"
        if not text and state != "idle":
            return f"[turn {state}]"
        return text

    def _log_turn(self, turn: _Turn, session: Session, state: str) -> None:
        duration_ms = int((time.monotonic() - turn.start) * 1000)
        self.log.metric(
            session_id=session.id,
            status=state,
            model_calls=turn.model_calls,
            tool_calls=turn.tool_calls,
            tokens_in=turn.tokens_in,
            tokens_out=turn.tokens_out,
            duration_ms=duration_ms,
            aborted=turn.cancelled,
        )
        self.log.event(
            "turn_end",
            session_id=session.id,
            status=state,
            model_calls=turn.model_calls,
            tool_calls=turn.tool_calls,
            tokens_in=turn.tokens_in,
            tokens_out=turn.tokens_out,
            duration_ms=duration_ms,
        )
        self._check_budget()

    def _check_budget(self) -> None:
        daily = self._budget_daily()
        if daily <= 0:
            return
        used = self._tokens_today_total()
        warn_at = float(self.config.get("budget.warn_at", 0.8) or 0.8)
        if used >= daily * warn_at:
            ratio = round(used / daily, 4)
            self.log.warn("budget_warning", used=used, daily=daily, ratio=ratio)
            self.log.metric(
                kind="budget_warning",
                budget_used=used,
                budget_daily=daily,
                budget_ratio=ratio,
            )

    def _budget_daily(self) -> int:
        return int(self.config.get("budget.daily_tokens", 0) or 0)

    def _tokens_today_total(self) -> int:
        today = self.log.tokens_today()
        return int(today.get("tokens_in", 0)) + int(today.get("tokens_out", 0))

    def _budget_used(self, turn: _Turn) -> int:
        """Turn-entry baseline plus this turn's in-process usage."""
        return turn.budget_baseline + turn.tokens_in + turn.tokens_out

    def _budget_exhausted(self) -> bool:
        if not bool(self.config.get("budget.enforce", False)):
            return False
        daily = self._budget_daily()
        if daily <= 0:
            return False
        return self._tokens_today_total() >= daily

    def _post_turn(self, session: Session, message: str, turn: _Turn) -> None:
        if self.is_worker or turn.cancelled:
            return
        if turn.tool_calls or _looks_like_correction(message) or turn.completion_tokens >= 400:
            try:
                conversation = turn.session.transcript(last_n=40)
                self.memory.extract(
                    conversation,
                    max_entries=int(self.config.get("memory.extract_max_per_turn", 3) or 3),
                    source=f"session:{session.id}",
                    model_client=self._compression(),
                )
            except Exception as exc:  # noqa: BLE001 - extraction is best effort
                self.log.warn("memory_extract_failed", error=str(exc))
        try:
            self.memory.consolidate(
                model_client=self._compression(),
                idle_after_s=float(self.config.get("memory.consolidate_idle_min", 15))
                * 60.0,
            )
        except Exception as exc:  # noqa: BLE001 - consolidation is best effort
            self.log.warn("memory_consolidate_failed", error=str(exc))

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.tools.stop_watcher()
        except Exception:  # noqa: BLE001
            pass
        if self.parent is None:
            try:
                self.models.close()
            except Exception:  # noqa: BLE001
                pass

    def shutdown(self) -> None:
        if self.health is not None:
            try:
                self.health.set_draining(True)
            except Exception:  # noqa: BLE001
                pass
        self.close()

    def __enter__(self) -> Agent:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
