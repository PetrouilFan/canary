"""Canary API server (spec §4.7).

The server is a thin layer over :class:`canary.core.agent.Agent`. It owns the
production listener fd (created above the process or inherited from systemd)
and, in canary mode, holds a second listener that only starts accepting after
the parent sends ``SIGUSR1``. The listening socket is never rebound or closed
during handover — see spec §5.3.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import queue
import signal
import socket
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

import uvicorn
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..core import util
from ..core.agent import Agent, SessionBusy
from ..core.evals import Evals

STREAM_POLL = 0.25
SSE_HEARTBEAT = 15.0


# ---------------------------------------------------------------------------
# listener helpers
# ---------------------------------------------------------------------------


def make_listener(host: str, port: int, backlog: int = 2048) -> socket.socket:
    """Create the stable production listener (never SO_REUSEPORT)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(backlog)
    sock.set_inheritable(True)
    return sock


def listener_from_env() -> socket.socket | None:
    """systemd socket activation: ``LISTEN_FDS``/``LISTEN_PID`` (fd 3+)."""
    try:
        fds = int(os.environ.get("LISTEN_FDS", "0") or 0)
        pid = int(os.environ.get("LISTEN_PID", "0") or 0)
    except ValueError:
        return None
    if fds >= 1 and pid == os.getpid():
        return socket.socket(fileno=3)
    return None


def socket_from_fd(fd: int) -> socket.socket:
    return socket.socket(fileno=fd)


# ---------------------------------------------------------------------------
# uvicorn integration
# ---------------------------------------------------------------------------


class _DrainServer(uvicorn.Server):
    """uvicorn server that reports shutdown requests before draining."""

    def __init__(self, config: uvicorn.Config, on_exit_request: Callable[[], None]):
        super().__init__(config)
        self._on_exit_request = on_exit_request

    def handle_exit(self, sig: int, frame: Any) -> None:
        if self.should_exit:
            self.force_exit = True
            return
        try:
            self._on_exit_request()
        except Exception:  # noqa: BLE001 - shutdown must continue
            pass
        self.should_exit = True


class _TurnStream:
    """Run a blocking ``Agent.run`` and surface its events asynchronously."""

    def __init__(self) -> None:
        self.queue: queue.Queue[dict | None] = queue.Queue()
        self.result: Any = None
        self.error: BaseException | None = None

    def start(self, runner: Callable[[Callable[[dict], None]], Any]) -> None:
        def work() -> None:
            try:
                self.result = runner(self.queue.put)
            except BaseException as exc:  # noqa: BLE001 - forwarded to the client
                self.error = exc
            finally:
                self.queue.put(None)

        threading.Thread(target=work, daemon=True, name="canary-api-turn").start()

    async def events(self) -> AsyncIterator[dict]:
        while True:
            item = await asyncio.to_thread(self.queue.get)
            if item is None:
                return
            yield item


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------


class APIServer:
    def __init__(
        self,
        agent: Agent,
        *,
        host: str = "127.0.0.1",
        port: int | None = None,
        allow_insecure_local: bool = False,
        listener: socket.socket | None = None,
        listener_fd: int | None = None,
        canary_mode: bool = False,
        canary_port: int | None = None,
    ) -> None:
        self.agent = agent
        self.config = agent.config
        self.log = agent.log
        self.host = host
        self.port = int(port if port is not None else self.config.get("port", 8080))
        self.allow_insecure_local = allow_insecure_local
        self.listener = listener
        self.listener_fd = listener_fd
        self.canary_mode = canary_mode
        self.canary_port = canary_port
        self.app = self._build_app()
        self._servers: list[uvicorn.Server] = []
        self._tasks: list[asyncio.Task] = []
        self._extra_tasks: list[asyncio.Task] = []
        self._serve_done = threading.Event()
        self._draining = threading.Event()
        self._promoted = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self.started_at = util.utc_now()

    # -- lifecycle -----------------------------------------------------------

    @property
    def drain_timeout(self) -> float:
        return float(self.config.get("releases.drain_timeout_s", 60) or 60)

    def serve_forever(self) -> None:
        try:
            asyncio.run(self._main())
        finally:
            self._serve_done.set()

    async def _main(self) -> None:
        loop = asyncio.get_running_loop()
        self._loop = loop
        if self.canary_mode:
            if self.listener_fd is not None:
                sock = socket_from_fd(self.listener_fd)
                loop.add_signal_handler(signal.SIGUSR1, self._promote, sock)
            self._sync_health()
            server = self._server(self.canary_port)
            self._tasks.append(asyncio.create_task(server.serve()))
            self.log.event("canary_child_ready", port=self.canary_port,
                           release_id=self.config.release_id)
            while self._tasks or self._extra_tasks:
                self._tasks.extend(self._extra_tasks)
                self._extra_tasks = []
                done, pending = await asyncio.wait(
                    self._tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    task.result()
                self._tasks = list(pending)
            return
        listener = self.listener or listener_from_env() or make_listener(
            self.host, self.port
        )
        self.listener_fd = listener.fileno()
        self._sync_health()
        agent_health = getattr(self.agent, "health", None)
        if agent_health is not None:
            agent_health.set_ready(True)
        server = self._server(None)
        self._tasks.append(asyncio.create_task(server.serve(sockets=[listener])))
        self.log.event("serve_started", host=self.host, port=self.port,
                       release_id=self.config.release_id)
        for task in self._tasks:
            await task

    def _sync_health(self) -> None:
        health = getattr(self.agent, "health", None)
        if health is None:
            return
        health.listener_fd = self.listener_fd
        health.port = self.port
        health.drain_callback = self.request_drain

    def _server(self, port: int | None) -> _DrainServer:
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=port or self.port,
            log_level="warning",
            access_log=False,
            timeout_graceful_shutdown=int(self.drain_timeout),
        )
        server = _DrainServer(config, on_exit_request=self._on_exit_request)
        self._servers.append(server)
        return server

    def _promote(self, sock: socket.socket) -> None:
        if self._promoted:
            return
        self._promoted = True
        self.log.event("canary_promoted", port=self.port,
                       release_id=self.config.release_id)
        server = self._server(None)
        task = asyncio.ensure_future(server.serve(sockets=[sock]))
        self._extra_tasks.append(task)

    def _on_exit_request(self) -> None:
        health = getattr(self.agent, "health", None)
        if health is not None:
            health.set_draining(True)
        self.log.event("shutdown_requested", release_id=self.config.release_id)
        self.log.health_probe("drain", "start", 0, "shutdown requested")
        self._stop_servers()

    def request_drain(self, timeout: float | None = None) -> bool:
        """Callable used by ``Health.drain_callback`` after promotion."""
        if not self._draining.is_set():
            self._draining.set()
            self._on_exit_request()
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._stop_servers)
        wait = float(timeout if timeout is not None else self.drain_timeout) + 10.0
        ok = self._serve_done.wait(wait)
        self.log.health_probe("drain", "ok" if ok else "timeout", wait * 1000,
                              f"finished={ok}")
        return ok

    def _stop_servers(self) -> None:
        for server in self._servers:
            server.should_exit = True

    # -- auth ----------------------------------------------------------------

    def _authorize(self, request: Request) -> None:
        key = os.environ.get("HARNESS_API_KEY", "")
        provided = ""
        header = request.headers.get("authorization") or ""
        if header.lower().startswith("bearer "):
            provided = header[7:].strip()
        host = request.client.host if request.client else ""
        if self.allow_insecure_local and util.is_loopback(host) and not provided:
            return
        if not key:
            raise HTTPException(
                status_code=401,
                detail="HARNESS_API_KEY is not configured; start with "
                       "--allow-insecure-local for loopback access",
            )
        if not provided or not hmac.compare_digest(provided, key):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    # -- app -----------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        agent = self.agent
        config = self.config
        log = self.log

        app = FastAPI(
            title="Canary",
            version="0.1.0",
            docs_url=None,
            redoc_url=None,
            dependencies=[Depends(self._authorize)],
        )

        # -- health ----------------------------------------------------------

        @app.get("/health")
        def health() -> dict[str, Any]:
            current = getattr(agent, "health", None)
            if current is None:
                return {"ok": True, "release_id": config.release_id,
                        "commit_sha": config.commit_sha}
            return current.health_response()

        @app.get("/ready")
        def ready() -> JSONResponse:
            current = getattr(agent, "health", None)
            if current is None:
                return JSONResponse({"ready": True, "release_id": config.release_id})
            body, status = current.ready_response()
            return JSONResponse(body, status_code=status)

        @app.post("/agent/smoke")
        def smoke() -> dict[str, Any]:
            session_id = f"smoke-{uuid.uuid4().hex[:8]}"
            role = "health" if "health" in config.role_names() else None
            try:
                text = agent.run(
                    "Reply with the single word: ok", session_id=session_id,
                    model=role,
                )
                ok = bool(text) and not text.startswith("error:")
                return {"ok": ok, "response": text[:500],
                        "release_id": config.release_id,
                        "commit_sha": config.commit_sha}
            except Exception as exc:  # noqa: BLE001 - probe must answer
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                        "release_id": config.release_id}
            finally:
                session = agent.sessions.load(session_id)
                if session is not None:
                    import shutil

                    shutil.rmtree(session.dir, ignore_errors=True)

        # -- OpenAI-compatible ----------------------------------------------

        @app.get("/v1/models")
        def models() -> dict[str, Any]:
            created = int(time.time())
            data = []
            for role in config.role_names():
                entry = config.role(role)
                data.append({
                    "id": role,
                    "object": "model",
                    "created": created,
                    "owned_by": "canary",
                    "provider": entry.get("provider"),
                    "model": entry.get("model"),
                    "context_length": entry.get("context_length"),
                })
            return {"object": "list", "data": data}

        @app.post("/v1/chat/completions")
        async def chat_completions(payload: dict = Body(...)) -> Any:
            if payload.get("tools"):
                raise HTTPException(
                    status_code=400,
                    detail="client-supplied tools are not supported; the harness "
                           "owns tool execution",
                )
            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                raise HTTPException(status_code=400, detail="messages must be a "
                                                            "non-empty array")
            requested = str(payload.get("model") or "")
            role = self._role_for(requested)
            session_id = payload.get("session_id")
            stream = bool(payload.get("stream"))
            if stream:
                return self._openai_stream(messages, session_id, role, requested)
            return await self._openai_completion(messages, session_id, role, requested)

        # -- admin -------------------------------------------------------------

        @app.get("/agent/status")
        def status() -> dict[str, Any]:
            out = agent.info()
            current = getattr(agent, "health", None)
            if current is not None:
                out.update(current.status())
            out["tokens_today"] = log.tokens_today()
            out["models"] = {
                role: {
                    "provider": config.role(role).get("provider"),
                    "model": config.role(role).get("model"),
                    "context_length": config.role(role).get("context_length"),
                }
                for role in config.role_names()
            }
            out["started_at"] = self.started_at
            out["listener_port"] = self.port
            out["canary_mode"] = self.canary_mode
            return out

        @app.get("/agent/metrics")
        def metrics(limit: int = Query(100, ge=1, le=5000)) -> dict[str, Any]:
            rows = [json.loads(line) for line in util.tail_lines(
                config.data_path / "metrics.jsonl", limit
            ) if line.strip()]
            return {"metrics": rows, "count": len(rows)}

        @app.get("/agent/sessions")
        def sessions(
            visible_to: str | None = Query(None),
            include_archived: bool = Query(False),
        ) -> dict[str, Any]:
            items = agent.sessions.list(
                visible_to=visible_to, include_archived=include_archived
            )
            return {"sessions": items, "count": len(items)}

        @app.get("/agent/sessions/{session_id}")
        def session_detail(
            session_id: str,
            last_n: int = Query(50, ge=1, le=1000),
        ) -> dict[str, Any]:
            session = agent.sessions.load(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="unknown session")
            messages = session.transcript(last_n=last_n)
            return {"session": session.info(), "messages": messages,
                    "events": session.events(last_n=50)}

        @app.post("/agent/sessions/{session_id}/cancel")
        def session_cancel(session_id: str) -> dict[str, Any]:
            session = agent.sessions.load(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="unknown session")
            session.request_cancel()
            return {"ok": True, "session_id": session_id}

        @app.post("/agent/sessions/{session_id}/inject")
        def session_inject(session_id: str, payload: dict = Body(...)) -> JSONResponse:
            message = str(payload.get("message") or "")
            if not message:
                raise HTTPException(status_code=400, detail="message is required")
            result = agent.sessions.inject_message(
                session_id,
                message,
                from_agent=str(payload.get("from_agent") or "api"),
                from_session=payload.get("from_session"),
                persist=bool(payload.get("persist")),
            )
            status_code = 429 if result.get("status") == "error" else 200
            return JSONResponse(result, status_code=status_code)

        @app.get("/agent/sessions/{session_id}/events")
        async def session_events(
            session_id: str,
            last_event_id: str | None = Header(None, alias="Last-Event-ID"),
        ) -> StreamingResponse:
            session = agent.sessions.load(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="unknown session")
            start = 0
            if last_event_id:
                try:
                    start = int(last_event_id) + 1
                except ValueError:
                    start = 0

            async def gen() -> AsyncIterator[str]:
                index = start
                last_ping = time.monotonic()
                while True:
                    events = session.events()
                    while index < len(events):
                        event = events[index]
                        etype = event.get("type", "event")
                        yield f"id: {index}\nevent: {etype}\n" \
                              f"data: {json.dumps(event, default=str)}\n\n"
                        index += 1
                    if time.monotonic() - last_ping >= SSE_HEARTBEAT:
                        last_ping = time.monotonic()
                        yield ": ping\n\n"
                    await asyncio.sleep(STREAM_POLL)

            return StreamingResponse(gen(), media_type="text/event-stream")

        @app.get("/agent/tools")
        def tools() -> dict[str, Any]:
            info = agent.tools.info()
            info["specs"] = agent.tools.specs()
            return info

        @app.post("/agent/tools/{name}/enable")
        def tool_enable(name: str) -> dict[str, Any]:
            if not agent.tools.enable(name):
                raise HTTPException(status_code=404, detail="unknown tool")
            return {"ok": True, "tool": name, "enabled": True}

        @app.post("/agent/tools/{name}/disable")
        def tool_disable(name: str) -> dict[str, Any]:
            if not agent.tools.disable(name):
                raise HTTPException(status_code=404, detail="unknown tool")
            return {"ok": True, "tool": name, "enabled": False}

        @app.get("/agent/memory/search")
        def memory_search(
            q: str = Query(..., min_length=1),
            k: int = Query(10, ge=1, le=100),
            tags: str | None = Query(None),
        ) -> dict[str, Any]:
            tag_list = [t.strip() for t in tags.split(",")] if tags else None
            hits = agent.memory.search(q, k=k, tags=tag_list)
            return {"query": q, "hits": [hit.to_dict() for hit in hits],
                    "count": len(hits)}

        @app.post("/agent/memory/save")
        def memory_save(payload: dict = Body(...)) -> dict[str, Any]:
            body = str(payload.get("text") or payload.get("body") or "")
            if not body:
                raise HTTPException(status_code=400, detail="text is required")
            entry = agent.memory.save(
                body,
                tags=payload.get("tags") or [],
                entry_id=payload.get("id"),
                source=str(payload.get("source") or "api"),
                importance=payload.get("importance", 0.5),
                relations=payload.get("relations"),
            )
            return {"ok": True, "id": entry.id, "tags": entry.tags}

        @app.get("/agent/jobs")
        def jobs_list(
            status: str | None = Query(None),
            agent_id: str | None = Query(None),
        ) -> dict[str, Any]:
            return agent.jobs.list(status=status, agent_id=agent_id)

        @app.get("/agent/jobs/{job_id}/status")
        def job_status(job_id: str) -> dict[str, Any]:
            job = agent.jobs.status(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="unknown job")
            return job

        @app.get("/agent/jobs/{job_id}/log")
        def job_log(
            job_id: str,
            offset: int | None = Query(None, ge=0),
            lines: int | None = Query(None, ge=1, le=100000),
        ) -> dict[str, Any]:
            page = agent.jobs.log(job_id, offset=offset, lines=lines)
            if page is None:
                raise HTTPException(status_code=404, detail="unknown job")
            return page

        @app.post("/agent/jobs/{job_id}/kill")
        def job_kill(job_id: str) -> dict[str, Any]:
            result = agent.jobs.kill(job_id)
            if result is None:
                raise HTTPException(status_code=404, detail="unknown job")
            return result if isinstance(result, dict) else {"ok": bool(result),
                                                           "job_id": job_id}

        @app.post("/agent/evals/run")
        def evals_run(payload: dict = Body(default={})) -> dict[str, Any]:
            evals = Evals(config, log, registry=agent.tools)
            report = evals.run(
                tag=payload.get("tag"),
                limit=payload.get("limit"),
                model_role=payload.get("model"),
            )
            return report

        @app.get("/agent/evals/results")
        def evals_results(limit: int = Query(50, ge=1, le=10000)) -> dict[str, Any]:
            rows = Evals(config, log, registry=agent.tools).results(limit=limit)
            return {"results": rows, "count": len(rows)}

        @app.get("/agent/evals/tasks")
        def evals_tasks(
            tag: str | None = Query(None),
            limit: int | None = Query(None, ge=1),
        ) -> dict[str, Any]:
            tasks = Evals(config, log, registry=agent.tools).load_tasks(
                tag=tag, limit=limit
            )
            return {"tasks": [task.to_dict() for task in tasks],
                    "count": len(tasks)}

        @app.post("/agent/governance/validate")
        def governance_validate(payload: dict = Body(...)) -> dict[str, Any]:
            target = str(payload.get("path") or "")
            if not target:
                raise HTTPException(status_code=400, detail="path is required")
            allowed, reason = agent.governance.check(target)
            return {"path": target, "allowed": allowed, "reason": reason,
                    "candidates": agent.governance.candidates(target)}

        @app.post("/agent/propose")
        def propose(payload: dict = Body(...)) -> dict[str, Any]:
            patch = str(payload.get("patch") or "")
            if not patch:
                raise HTTPException(status_code=400, detail="patch is required")
            result = agent.propose(
                patch,
                motivation=payload.get("motivation"),
            )
            if isinstance(result, str):
                raise HTTPException(status_code=400, detail=result)
            return result

        @app.post("/agent/revert")
        def revert(payload: dict = Body(default={})) -> dict[str, Any]:
            result = agent.revert(payload.get("release_id"))
            if isinstance(result, str):
                raise HTTPException(status_code=400, detail=result)
            return result

        @app.get("/agent/releases")
        def releases() -> dict[str, Any]:
            current = getattr(agent, "health", None)
            if current is None:
                return {"current": None, "releases": [], "green": [],
                        "running_release": config.release_id}
            return current.list_releases()

        @app.post("/agent/unlock")
        def unlock(payload: dict = Body(...)) -> JSONResponse:
            nonce = str(payload.get("nonce") or "")
            if not nonce:
                raise HTTPException(status_code=400, detail="nonce is required")
            current = getattr(agent, "health", None)
            if current is None:
                raise HTTPException(status_code=409,
                                    detail="no codebase lock in this mode")
            result = current.unlock(nonce)
            return JSONResponse(result,
                                status_code=200 if result.get("ok") else 409)

        @app.post("/agent/run")
        async def agent_run(payload: dict = Body(...)) -> Any:
            message = str(payload.get("message") or "")
            if not message:
                raise HTTPException(status_code=400, detail="message is required")
            session_id = payload.get("session_id")
            role = payload.get("model")
            fail_fast = bool(payload.get("fail_fast"))
            if payload.get("stream"):
                stream = _TurnStream()
                stream.start(lambda cb: agent.run(
                    message, session_id=session_id, model=role, on_event=cb,
                    fail_fast=fail_fast,
                ))

                async def gen() -> AsyncIterator[str]:
                    async for event in stream.events():
                        etype = event.get("type", "event")
                        yield f"event: {etype}\n" \
                              f"data: {json.dumps(event, default=str)}\n\n"
                    if stream.error is not None:
                        yield "event: error\n" + \
                              f"data: {json.dumps({'error': str(stream.error)})}\n\n"
                    else:
                        yield "event: done\n" + \
                              f"data: {json.dumps({'response': stream.result})}\n\n"

                return StreamingResponse(gen(), media_type="text/event-stream")
            try:
                text = await asyncio.to_thread(
                    lambda: agent.run(message, session_id=session_id, model=role,
                                      fail_fast=fail_fast)
                )
            except SessionBusy as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            return {
                "response": text,
                "session_id": (agent.last_result or {}).get("session_id"),
                "usage": agent.last_usage,
                "release_id": config.release_id,
                "agent_id": agent.agent_id,
            }

        return app

    # -- OpenAI helpers ------------------------------------------------------

    def _role_for(self, requested: str) -> str | None:
        if not requested:
            return None
        if requested in self.config.role_names():
            return requested
        for role in self.config.role_names():
            if self.config.role(role).get("model") == requested:
                return role
        return None

    def _openai_messages(
        self, messages: list[dict]
    ) -> tuple[str, list[dict], list[dict]]:
        """Return (last user text, prior messages, full client message list)."""
        clean: list[dict] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "")
            content = message.get("content")
            if isinstance(content, list):
                content = "".join(
                    str(part.get("text") or "")
                    for part in content
                    if isinstance(part, dict)
                )
            if role in ("user", "assistant", "system") and content:
                clean.append({"role": role, "content": str(content)})
        last_user = ""
        for message in reversed(clean):
            if message["role"] == "user":
                last_user = message["content"]
                break
        prior = clean[:-1] if clean and clean[-1]["role"] == "user" else clean
        return last_user, prior, clean

    def _run_api_turn(
        self,
        *,
        prior: list[dict],
        text: str,
        session_id: str | None,
        role: str | None,
        on_event: Callable[[dict], None] | None = None,
        fail_fast: bool = False,
    ) -> str:
        agent = self.agent
        if session_id:
            return agent.run(text, session_id=session_id, model=role,
                             on_event=on_event, fail_fast=fail_fast)
        if not prior:
            return agent.run(text, session_id=None, model=role, on_event=on_event,
                             fail_fast=fail_fast)
        ephemeral = f"api-{uuid.uuid4().hex[:12]}"
        session = agent.sessions.create(session_id=ephemeral, name="api-ephemeral")
        try:
            for message in prior:
                if message["role"] in ("user", "assistant"):
                    session.append_message(message["role"], message["content"])
            return agent.run(text, session_id=ephemeral, model=role,
                             on_event=on_event, fail_fast=fail_fast)
        finally:
            import shutil

            shutil.rmtree(session.dir, ignore_errors=True)

    async def _openai_completion(
        self, messages: list[dict], session_id: str | None, role: str | None,
        requested: str,
    ) -> dict[str, Any]:
        text, prior, _ = self._openai_messages(messages)
        if not text:
            raise HTTPException(status_code=400, detail="no user message found")
        try:
            result = await asyncio.to_thread(
                lambda: self._run_api_turn(
                    prior=prior, text=text, session_id=session_id, role=role
                )
            )
        except SessionBusy as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        usage = dict(self.agent.last_usage)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": requested or role or self.agent.default_model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            },
            "session_id": session_id or self.agent.session_id,
        }

    def _openai_stream(
        self, messages: list[dict], session_id: str | None, role: str | None,
        requested: str,
    ) -> StreamingResponse:
        text, prior, _ = self._openai_messages(messages)
        if not text:
            raise HTTPException(status_code=400, detail="no user message found")
        stream = _TurnStream()
        stream.start(lambda cb: self._run_api_turn(
            prior=prior, text=text, session_id=session_id, role=role, on_event=cb
        ))
        model_id = requested or role or self.agent.default_model
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        def chunk(delta: dict, finish: str | None = None,
                  usage: dict | None = None) -> str:
            body: dict[str, Any] = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": ([{"index": 0, "delta": delta, "finish_reason": finish}]
                            if usage is None else []),
            }
            if usage is not None:
                body["usage"] = usage
            return f"data: {json.dumps(body)}\n\n"

        async def gen() -> AsyncIterator[str]:
            yield chunk({"role": "assistant", "content": ""})
            async for event in stream.events():
                if event.get("type") == "delta" and event.get("content"):
                    yield chunk({"content": str(event["content"])})
            if stream.error is not None:
                error = {"error": {"message": str(stream.error),
                                   "type": "canary_error"}}
                yield f"data: {json.dumps(error)}\n\n"
            yield chunk({}, finish="stop")
            usage = dict(self.agent.last_usage)
            yield chunk({}, usage={
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            })
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")


__all__ = ["APIServer", "make_listener", "listener_from_env"]
