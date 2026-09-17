"""Session management: tree of sessions, history, inbox, cross-process locks.

Layout::

    shared/data/sessions/
        {id}/
            state.json      # atomic rewrite: status, turn_open, counters
            history.jsonl   # append-only, one JSON object per line
            inbox.jsonl     # append-only injections
            inbox_state.json# drain offset, evictions, rate counters
            events.jsonl    # delivery receipts / lifecycle events
            .lock           # flock EX: one turn at a time per session
            .inbox.lock     # flock EX: inbox state mutations
        archived/{id}/      # idle sessions are moved here

Rules (spec §4.8, §4.12):

* the server's history is authoritative for a session id; state is rewritten
  atomically at every tool-call boundary;
* sessions form a tree: branching copies history up to (not including) the
  branch point; the branch is an independent session with its own lock;
* a turn that was in flight when the process died is marked
  ``aborted: interrupted`` at load; tools are never replayed;
* injections are ephemeral: they live in the receiver's active context, are
  filtered out of compression summaries unless ``persist: true``, and are
  rate limited per sender→receiver per receiver turn.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from canary.core import util
from canary.core.config import Config
from canary.core.observability import Log

TURN_OPEN = "turn_open"
INTERRUPTED_KEY = "interrupted"


@dataclass
class Message:
    role: str
    content: str
    id: str = ""
    ts: str = ""
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    from_agent: str | None = None
    persist: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        data: dict[str, Any] = {
            "id": self.id or util.new_nonce()[:12],
            "ts": self.ts or util.utc_now(),
            "role": self.role,
            "content": self.content,
        }
        if self.tool_calls:
            data["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.name:
            data["name"] = self.name
        if self.from_agent:
            data["from_agent"] = self.from_agent
        if self.persist:
            data["persist"] = True
        data.update(self.extra)
        return data

    @classmethod
    def from_json(cls, data: dict) -> Message:
        known = {
            "id", "ts", "role", "content", "tool_calls", "tool_call_id",
            "name", "from_agent", "persist",
        }
        return cls(
            id=str(data.get("id") or ""),
            ts=str(data.get("ts") or ""),
            role=str(data.get("role") or "user"),
            content=str(data.get("content") or ""),
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
            from_agent=data.get("from_agent"),
            persist=bool(data.get("persist")),
            extra={k: v for k, v in data.items() if k not in known},
        )

    def to_openai(self) -> dict:
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name and self.role == "tool":
            msg["name"] = self.name
        return msg


class Session:
    """One conversation with its own history, inbox and lock."""

    def __init__(self, manager: SessionManager, session_id: str, directory: Path):
        self.manager = manager
        self.id = session_id
        self.dir = directory
        self.history_path = directory / "history.jsonl"
        self.inbox_path = directory / "inbox.jsonl"
        self.inbox_state_path = directory / "inbox_state.json"
        self.state_path = directory / "state.json"
        self.events_path = directory / "events.jsonl"
        self._state: dict[str, Any] = {}

    # -- state ---------------------------------------------------------------

    @property
    def state(self) -> dict:
        if not self._state:
            loaded = util.read_json(self.state_path) or {}
            self._state = loaded if isinstance(loaded, dict) else {}
        return self._state

    def save_state(self) -> None:
        self.state["updated"] = util.utc_now()
        util.atomic_write_json(self.state_path, self.state)

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.state[key] = value

    @property
    def name(self) -> str:
        return str(self.state.get("name") or self.id)

    @property
    def agent_id(self) -> str:
        return str(self.state.get("agent_id") or "")

    @property
    def visible_to(self) -> str:
        return str(self.state.get("visible_to") or "all")

    @property
    def parent_session(self) -> str | None:
        value = self.state.get("parent_session")
        return str(value) if value else None

    @property
    def branch_point(self) -> str | None:
        value = self.state.get("branch_point")
        return str(value) if value else None

    @property
    def is_worker(self) -> bool:
        return bool(self.state.get("worker"))

    @property
    def last_activity(self) -> str:
        return str(self.state.get("last_activity") or self.state.get("created") or "")

    @property
    def status(self) -> str:
        if self.state.get(TURN_OPEN):
            return "running"
        return str(self.state.get("status") or "idle")

    # -- locking -------------------------------------------------------------

    def lock(self, op: str = "turn", *, wait: bool = True, timeout: float = 0.0) -> util.FileLock:
        return util.FileLock(self.dir / ".lock", op=op, wait=wait, timeout=timeout)

    # -- history -------------------------------------------------------------

    def append(self, message: Message) -> Message:
        if not message.id:
            message.id = util.new_nonce()[:12]
        if not message.ts:
            message.ts = util.utc_now()
        util.append_jsonl(self.history_path, message.to_json())
        self.state["last_activity"] = message.ts
        self.save_state()
        return message

    def append_message(self, role: str, content: str, **kwargs: Any) -> Message:
        return self.append(Message(role=role, content=content, **kwargs))

    def history(self, last_n: int | None = None) -> list[Message]:
        messages = [Message.from_json(item) for item in util.read_jsonl(self.history_path)]
        if last_n is not None and last_n > 0:
            return messages[-last_n:]
        return messages

    def history_ids(self) -> list[str]:
        return [m.id for m in self.history()]

    def transcript(self, last_n: int | None = None, include_tools: bool = False) -> str:
        lines: list[str] = []
        for msg in self.history(last_n):
            if msg.role == "tool" and not include_tools:
                lines.append(f"[tool result: {util.truncate(msg.content, 400)}]")
                continue
            if msg.from_agent:
                lines.append(f"[{msg.role} from @{msg.from_agent}] {msg.content}")
            else:
                lines.append(f"[{msg.role}] {msg.content}")
        return "\n".join(lines)

    # -- turns ---------------------------------------------------------------

    def begin_turn(self) -> None:
        """Mark a turn in flight; reset injection counters for the new turn."""
        state = self.state
        state[TURN_OPEN] = True
        state["status"] = "running"
        state["turn_seq"] = int(state.get("turn_seq", 0)) + 1
        state["injections"] = {}
        state.pop(INTERRUPTED_KEY, None)
        self.save_state()

    def end_turn(self, status: str = "idle") -> None:
        state = self.state
        state[TURN_OPEN] = False
        state["status"] = status
        self.save_state()

    def take_interrupted(self) -> bool:
        """True when the previous turn died mid-flight; clears the flag."""
        if not self.state.get(INTERRUPTED_KEY):
            return False
        self.state[INTERRUPTED_KEY] = False
        self.save_state()
        return True

    def recover(self) -> bool:
        """Called at boot/load: mark an in-flight turn as interrupted."""
        if self.state.get(TURN_OPEN):
            self.state[TURN_OPEN] = False
            self.state[INTERRUPTED_KEY] = True
            self.state["status"] = "aborted"
            self.save_state()
            self.append_event("interrupted", session=self.id)
            return True
        return False

    def set_context_usage(self, tokens: int) -> None:
        self.state["context_usage"] = int(tokens)
        self.save_state()

    # -- cancellation --------------------------------------------------------

    @property
    def cancel_flag(self) -> Path:
        return self.dir / "cancel.flag"

    def request_cancel(self) -> bool:
        """Cross-process cancel request; the turn loop checks the flag."""
        self.cancel_flag.write_text(util.utc_now(), encoding="utf-8")
        self.append_event("cancel_requested", session=self.id)
        return True

    def cancel_requested(self) -> bool:
        return self.cancel_flag.exists()

    def clear_cancel(self) -> None:
        try:
            self.cancel_flag.unlink()
        except FileNotFoundError:
            pass

    # -- branching -----------------------------------------------------------

    def fork(
        self,
        *,
        branch_point: str | None = None,
        name: str | None = None,
        agent_id: str | None = None,
    ) -> Session:
        """Create a branch that copies history up to (excluding) branch_point."""
        new_id = str(uuid.uuid4())
        child = self.manager._new_dir(new_id)
        copy: list[Message] = []
        for msg in self.history():
            if branch_point and msg.id == branch_point:
                break
            copy.append(msg)
        child_state = {
            "id": new_id,
            "name": name or f"{self.name}-branch",
            "agent_id": agent_id or self.agent_id,
            "visible_to": self.visible_to,
            "created": util.utc_now(),
            "updated": util.utc_now(),
            "last_activity": copy[-1].ts if copy else util.utc_now(),
            "parent_session": self.id,
            "branch_point": branch_point,
            "status": "idle",
            "injections": {},
            "turn_seq": 0,
        }
        child = self.manager._session_object(new_id, child_state)
        for msg in copy:
            child.append(msg)
        child.save_state()
        self.append_event("branched", child=child.id, branch_point=branch_point or "")
        return child

    # -- inbox ---------------------------------------------------------------

    def inbox_state(self) -> dict:
        state = util.read_json(self.inbox_state_path) or {}
        state.setdefault("offset", 0)
        state.setdefault("evicted", [])
        return state

    def _inbox_lines(self) -> list[tuple[int, dict]]:
        """(line_number, item) for every raw inbox line."""
        out: list[tuple[int, dict]] = []
        path = self.inbox_path
        if not path.exists():
            return out
        try:
            with open(path, encoding="utf-8") as fh:
                for index, raw in enumerate(fh):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        item = json.loads(raw)
                    except ValueError:
                        continue
                    if isinstance(item, dict):
                        out.append((index, item))
        except OSError:
            return []
        return out

    def pending_inbox(self) -> list[dict]:
        """Not-yet-drained, not-evicted, not-expired items, oldest first."""
        state = self.inbox_state()
        offset = int(state.get("offset", 0))
        evicted = set(state.get("evicted", []))
        ttl_h = float(self.manager.config.get("sessions.inbox_ttl_h", 24))
        now = time.time()
        out: list[dict] = []
        for index, item in self._inbox_lines():
            if index < offset:
                continue
            if item.get("id") in evicted:
                continue
            if not item.get("persist") and ttl_h > 0:
                try:
                    age_h = (now - util.parse_stamp(str(item.get("ts", "")))) / 3600.0
                except ValueError:
                    age_h = 0.0
                if age_h > ttl_h:
                    continue
            out.append(item)
        return out

    def inject(self, item: dict) -> dict:
        """Append an injection, honoring cap and FIFO eviction of old items."""
        with util.FileLock(self.dir / ".inbox.lock", op="inbox", timeout=5.0):
            max_items = int(self.manager.config.get("sessions.inbox_max", 100))
            pending = self.pending_inbox()
            if len(pending) >= max_items:
                oldest = next((p for p in pending if not p.get("persist")), None)
                if oldest is None:
                    return {"status": "rejected", "error": "inbox full (all items persistent)"}
                state = self.inbox_state()
                evicted = list(dict.fromkeys([*state.get("evicted", []), oldest["id"]]))
                state["evicted"] = evicted[-500:]
                util.atomic_write_json(self.inbox_state_path, state)
                pending = self.pending_inbox()
            util.append_jsonl(self.inbox_path, item)
            return {"status": "queued", "position": len(pending)}

    def drain_inbox(self) -> list[dict]:
        """Mark pending items drained, emit receipts, return them oldest first."""
        with util.FileLock(self.dir / ".inbox.lock", op="inbox", timeout=5.0):
            pending = self.pending_inbox()
            state = self.inbox_state()
            lines = self._inbox_lines()
            state["offset"] = (lines[-1][0] + 1) if lines else 0
            state["evicted"] = []
            util.atomic_write_json(self.inbox_state_path, state)
        for item in pending:
            sender = item.get("from_session")
            if sender:
                self.manager.append_event(
                    str(sender),
                    "injection_read",
                    message_id=item.get("id"),
                    receiver=self.id,
                )
        return pending

    # -- events --------------------------------------------------------------

    def append_event(self, event_type: str, **data: Any) -> None:
        util.append_jsonl(
            self.events_path,
            {"ts": util.utc_now(), "type": event_type, **data},
        )

    def events(self, last_n: int | None = None) -> list[dict]:
        items = list(util.read_jsonl(self.events_path))
        if last_n is not None and last_n > 0:
            return items[-last_n:]
        return items

    def had_recent_read(self, receiver_id: str) -> bool:
        """True when the receiver drained an injection from us recently."""
        for event in reversed(self.events(last_n=50)):
            if event.get("type") == "injection_read" and event.get("receiver") == receiver_id:
                return True
        return False

    # -- metadata ------------------------------------------------------------

    def info(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "agent_id": self.agent_id,
            "status": self.status,
            "last_activity": self.last_activity,
            "context_usage": int(self.state.get("context_usage", 0)),
            "visible_to": self.visible_to,
            "worker": self.is_worker,
            "parent_session": self.parent_session,
            "branch_point": self.branch_point,
        }

    def touch(self) -> None:
        self.state["last_activity"] = util.utc_now()
        self.save_state()


class SessionManager:
    """Creates, lists, recovers and archives sessions across processes."""

    def __init__(self, config: Config, log: Log | None = None):
        self.config = config
        self.log = log
        self.root = config.data_path / "sessions"
        self.archive_root = self.root / "archived"
        util.ensure_dir(self.root)

    # -- creation ------------------------------------------------------------

    def _new_dir(self, session_id: str) -> Path:
        directory = self.root / session_id
        util.ensure_dir(directory)
        return directory

    def _session_object(self, session_id: str, state: dict) -> Session:
        session = Session(self, session_id, self.root / session_id)
        session._state = state
        session.save_state()
        return session

    def create(
        self,
        *,
        name: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        visible_to: str = "all",
        worker: bool = False,
        parent_session: str | None = None,
        branch_point: str | None = None,
    ) -> Session:
        session_id = session_id or str(uuid.uuid4())
        directory = self._new_dir(session_id)
        state = {
            "id": session_id,
            "name": name or (worker and f"worker-{util.new_nonce()[:6]}") or session_id,
            "agent_id": agent_id or str(self.config.get("agent.id")),
            "visible_to": visible_to,
            "created": util.utc_now(),
            "updated": util.utc_now(),
            "last_activity": util.utc_now(),
            "status": "idle",
            "worker": worker,
            "parent_session": parent_session,
            "branch_point": branch_point,
            "injections": {},
            "turn_seq": 0,
        }
        session = Session(self, session_id, directory)
        session._state = state
        session.save_state()
        session.append_event("created", name=state["name"], worker=worker)
        if self.log:
            self.log.info("session_created", session=session_id, name=state["name"], worker=worker)
        return session

    def load(self, session_id: str) -> Session | None:
        directory = self.root / session_id
        if not directory.is_dir():
            archived = self.archive_root / session_id
            if archived.is_dir():
                directory = archived
            else:
                return None
        session = Session(self, session_id, directory)
        _ = session.state
        session.recover()
        return session

    def get_or_create(self, session_id: str | None, **kwargs: Any) -> Session:
        if session_id:
            existing = self.load(session_id)
            if existing is not None:
                return existing
        return self.create(session_id=session_id, **kwargs)

    def branch(self, session_id: str, branch_point: str, name: str | None = None) -> Session:
        parent = self.load(session_id)
        if parent is None:
            raise KeyError(f"unknown session {session_id}")
        return parent.fork(branch_point=branch_point, name=name)

    # -- listing -------------------------------------------------------------

    def list(self, *, visible_to: str | None = None, include_archived: bool = False) -> list[dict]:
        out: list[dict] = []
        roots = [self.root]
        if include_archived:
            roots.append(self.archive_root)
        for base in roots:
            if not base.is_dir():
                continue
            for child in sorted(base.iterdir()):
                if child.name in ("archived",) or not child.is_dir():
                    continue
                session = Session(self, child.name, child)
                if visible_to and session.visible_to not in (visible_to, "all"):
                    continue
                out.append(session.info())
        out.sort(key=lambda info: info.get("last_activity") or "", reverse=True)
        return out

    # -- injection (cross-agent) --------------------------------------------

    RATE_KIND = "injections"

    def inject_message(
        self,
        session_id: str,
        message: str,
        *,
        from_agent: str,
        from_session: str | None = None,
        persist: bool = False,
    ) -> dict:
        receiver = self.load(session_id)
        if receiver is None:
            return {"status": "error", "error": f"unknown session {session_id}"}
        if receiver.visible_to == "none":
            return {"status": "error", "error": "session is not visible to other agents"}
        rate_max = int(self.config.get("sessions.inject_rate", 5))
        sender_key = from_agent or (from_session or "unknown")
        state = receiver.state
        turn_seq = int(state.get("turn_seq", 0))
        counts = dict(state.get("injections") or {})
        current = counts.get(sender_key) or {}
        if int(current.get("turn", -1)) != turn_seq:
            current = {"turn": turn_seq, "count": 0}
        if rate_max > 0 and int(current.get("count", 0)) >= rate_max:
            return {
                "status": "error",
                "error": "rate limit: too many injections to this session this turn",
            }
        item = {
            "id": util.new_nonce()[:12],
            "ts": util.utc_now(),
            "from_agent": from_agent,
            "from_session": from_session,
            "message": message,
            "persist": bool(persist),
        }
        result = receiver.inject(item)
        if result.get("status") == "rejected":
            return result
        lock = receiver.lock("inject", wait=False)
        if lock.acquire():
            try:
                state = receiver.state
                count_now = int(state.get("injections", {}).get(sender_key, {}).get("count", 0))
                state.setdefault("injections", {})[sender_key] = {
                    "turn": int(state.get("turn_seq", 0)),
                    "count": count_now + 1,
                }
                receiver.save_state()
            finally:
                lock.release()
        delivered = not bool(receiver.state.get(TURN_OPEN))
        previous_read = False
        if from_session:
            sender = self.load(from_session)
            if sender is not None:
                previous_read = sender.had_recent_read(session_id)
        if self.log:
            self.log.info(
                "session_injected",
                session=session_id,
                from_agent=from_agent,
                persist=persist,
            )
        return {
            "status": "queued" if not delivered else "delivered",
            "position": int(result.get("position", 0)),
            "previous_read": previous_read,
        }

    def append_event(self, session_id: str, event_type: str, **data: Any) -> None:
        directory = self.root / session_id
        if not directory.is_dir():
            directory = self.archive_root / session_id
        if not directory.is_dir():
            return
        util.append_jsonl(
            directory / "events.jsonl",
            {"ts": util.utc_now(), "type": event_type, **data},
        )

    # -- idle archival -------------------------------------------------------

    def sweep(self) -> list[str]:
        """Archive sessions idle longer than sessions.idle_archive_h."""
        limit_h = float(self.config.get("sessions.idle_archive_h", 24))
        if limit_h <= 0:
            return []
        cutoff = time.time() - limit_h * 3600.0
        archived: list[str] = []
        for child in sorted(self.root.iterdir()):
            if child.name == "archived" or not child.is_dir():
                continue
            session = Session(self, child.name, child)
            if session.state.get(TURN_OPEN):
                continue
            try:
                last = util.parse_stamp(session.last_activity)
            except ValueError:
                continue
            if last >= cutoff:
                continue
            _ = session.drain_inbox() if session.pending_inbox() else []
            util.ensure_dir(self.archive_root)
            target = self.archive_root / child.name
            try:
                child.rename(target)
            except OSError as exc:
                if self.log:
                    self.log.warn("session_archive_failed", session=child.name, error=str(exc))
                continue
            archived.append(child.name)
            if self.log:
                self.log.info("session_archived", session=child.name)
        return archived
