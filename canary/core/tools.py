"""Tool registry, built-in tools, and extension hot-reload (spec 4.4, 4.10).

Tools always return strings; errors are returned as error strings and never
raised into the agent loop.  Extensions are Python modules in
``shared/extensions/`` declaring :class:`Tool` subclasses; they are
hot-reloaded with the old version kept live on failure.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

from .config import Config
from .governance import Governance
from .util import atomic_write_text, ensure_dir, truncate

MAX_RESULT_CHARS = 200_000
READ_MAX_CHARS = 100_000
SPILL_HINT = "output truncated"


def _text(result: Any) -> str:
    """Render a tool result: strings pass through, structures become JSON."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, indent=2, default=str)
    except (TypeError, ValueError):
        return str(result)


class Tool:
    """Extension interface (spec 4.4)."""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}
    requires: list[str] = []
    profiles: list[str] = []

    def __init__(self, harness: Any) -> None:
        self.harness = harness

    def __call__(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    # -- helpers shared by built-ins ------------------------------------

    @property
    def config(self) -> Config:
        return self.harness.config

    @property
    def log(self) -> Any:
        return self.harness.log

    def resolve_path(self, path: str) -> Path:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = (self.config.base_dir / p).resolve()
        return p

    def missing_requires(self) -> list[str]:
        return [key for key in self.requires if not os.environ.get(key)]

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
            "profiles": list(self.profiles),
        }


def _clip(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    return truncate(text, limit, marker=f"\n[... {SPILL_HINT}: {len(text)} chars total]")


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------


class ReadTool(Tool):
    name = "read"
    description = "Read a file. Returns the contents; optionally start/end line numbers."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path absolute or relative to the canary root",
            },
            "start": {"type": "integer", "description": "1-based first line"},
            "end": {"type": "integer", "description": "Inclusive last line"},
        },
        "required": ["path"],
    }
    profiles = ["file-editing"]

    def __call__(
        self, path: str, start: int | None = None, end: int | None = None, **_: Any
    ) -> str:
        p = self.resolve_path(path)
        if not p.is_file():
            return f"error: no such file: {p}"
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"error: read failed: {exc}"
        lines = text.splitlines()
        if start or end:
            first = max(1, int(start or 1))
            last = min(len(lines), int(end or len(lines)))
            selected = lines[first - 1 : last]
            body = "\n".join(f"{i}: {line}" for i, line in enumerate(selected, start=first))
        else:
            body = text
        if len(body) > READ_MAX_CHARS:
            body = (
                body[:READ_MAX_CHARS]
                + f"\n[... {SPILL_HINT}: showing first {READ_MAX_CHARS} chars]"
            )
        return body if body else "(empty file)"


class WriteTool(Tool):
    name = "write"
    description = (
        "Write a new file (or overwrite one) after a governance check. "
        "Prefer `edit` for existing files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }
    profiles = ["file-editing"]

    def __call__(self, path: str, content: str, **_: Any) -> str:
        p = self.resolve_path(path)
        allowed, reason = self.harness.governance.check(p)
        if not allowed:
            self.log.event("write_denied", path=str(p), reason=reason)
            return f"error: write denied by governance: {reason}"
        try:
            ensure_dir(p.parent)
            atomic_write_text(p, content)
        except OSError as exc:
            return f"error: write failed: {exc}"
        self.log.event("tool_write", path=str(p), bytes=len(content))
        return f"wrote {p} ({len(content)} bytes)"


class EditTool(Tool):
    name = "edit"
    description = "Replace an exact string in an existing file after a governance check."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old": {"type": "string", "description": "Exact text to replace"},
            "new": {"type": "string"},
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence (default false)",
            },
        },
        "required": ["path", "old", "new"],
    }
    profiles = ["file-editing"]

    def __call__(self, path: str, old: str, new: str, replace_all: bool = False, **_: Any) -> str:
        p = self.resolve_path(path)
        allowed, reason = self.harness.governance.check(p)
        if not allowed:
            self.log.event("write_denied", path=str(p), reason=reason)
            return f"error: write denied by governance: {reason}"
        if not p.is_file():
            return f"error: no such file: {p}"
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as exc:
            return f"error: read failed: {exc}"
        count = text.count(old)
        if count == 0:
            return f"error: old text not found in {p}"
        if count > 1 and not replace_all:
            return (
                f"error: old text occurs {count} times in {p}; "
                "add more context or pass replace_all=true"
            )
        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        try:
            atomic_write_text(p, updated)
        except OSError as exc:
            return f"error: write failed: {exc}"
        self.log.event("tool_edit", path=str(p), occurrences=count if replace_all else 1)
        return f"edited {p} ({count if replace_all else 1} occurrence(s))"


class BashTool(Tool):
    name = "bash"
    description = (
        "Run a shell command (ungoverned by design). Long-running work should "
        "use job_spawn instead."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "cwd": {
                "type": "string",
                "description": "Working directory (default: canary base dir)",
            },
            "timeout": {"type": "integer", "description": "Seconds (default 120)"},
        },
        "required": ["command"],
    }
    profiles = ["shell"]

    def __call__(
        self, command: str, cwd: str | None = None, timeout: int | None = None, **_: Any
    ) -> str:
        workdir = self.resolve_path(cwd) if cwd else self.config.base_dir
        limit = int(timeout or self.config.get("tools.bash_timeout", 120))
        self.log.event("tool_bash", command=command[:500], cwd=str(workdir), timeout=limit)
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=limit,
            )
        except subprocess.TimeoutExpired:
            return f"error: command timed out after {limit}s; use job_spawn for long-running work"
        except OSError as exc:
            return f"error: failed to run command: {exc}"
        parts = [f"exit_code: {proc.returncode}"]
        if proc.stdout:
            parts.append("stdout:\n" + proc.stdout)
        if proc.stderr:
            parts.append("stderr:\n" + proc.stderr)
        return _clip("\n".join(parts))


class ProposeTool(Tool):
    name = "propose"
    description = "Propose a code patch. Runs preflight, canary and publish (spec 4.6)."
    parameters = {
        "type": "object",
        "properties": {
            "patch": {"type": "string", "description": "Unified diff to apply to shared/staging"},
            "motivation": {"type": "string", "description": "Why this change is needed"},
        },
        "required": ["patch"],
    }
    profiles = ["self-modification"]

    def __call__(self, patch: str, motivation: str | None = None, **_: Any) -> str:
        return _text(self.harness.propose(patch, motivation=motivation))


class RevertTool(Tool):
    name = "revert"
    description = "Revert the running copy to a previous green release."
    parameters = {
        "type": "object",
        "properties": {
            "release_id": {
                "type": "string",
                "description": "Release id or green tag (default: previous)",
            }
        },
    }
    profiles = ["self-modification"]

    def __call__(self, release_id: str | None = None, **_: Any) -> str:
        return _text(self.harness.revert(release_id))


class CompressTool(Tool):
    name = "compress"
    description = "Compress older conversation context into a summary, keeping protected units."
    parameters = {
        "type": "object",
        "properties": {
            "focus": {"type": "string", "description": "What to preserve in the summary"}
        },
    }
    profiles = ["context"]

    def __call__(self, focus: str | None = None, **_: Any) -> str:
        return _text(self.harness.compress_current(focus))


class MemorySearchTool(Tool):
    name = "memory_search"
    description = "Search shared memory by query; returns the top matching entries."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "k": {"type": "integer", "description": "Number of results (default from config)"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["query"],
    }
    profiles = ["memory"]

    def __call__(
        self, query: str, k: int | None = None, tags: list[str] | None = None, **_: Any
    ) -> str:
        from .memory import format_hits

        hits = self.harness.memory.search(query, k=k, tags=tags)
        return format_hits(hits) if hits else "no matches"


class MemorySaveTool(Tool):
    name = "memory_save"
    description = "Save a durable fact to shared memory."
    parameters = {
        "type": "object",
        "properties": {
            "body": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "importance": {"type": "number", "description": "0..1 (default 0.5)"},
            "id": {"type": "string", "description": "Stable entry id (default: derived from body)"},
            "relations": {"type": "object", "description": 'e.g. {"supersedes": ["id"]}'},
        },
        "required": ["body"],
    }
    profiles = ["memory"]

    def __call__(
        self,
        body: str,
        tags: list[str] | None = None,
        importance: float | None = None,
        id: str | None = None,
        relations: dict[str, list[str]] | None = None,
        **_: Any,
    ) -> str:
        entry = self.harness.memory.save(
            body,
            tags=tags or [],
            entry_id=id,
            source=f"tool:{self.harness.session_id or 'cli'}",
            importance=importance,
            relations=relations,
        )
        return f"saved memory entry {entry.id}"


class MemoryForgetTool(Tool):
    name = "memory_forget"
    description = "Archive a memory entry (entries are never hard-deleted)."
    parameters = {
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
    }
    profiles = ["memory"]

    def __call__(self, id: str, **_: Any) -> str:
        ok = self.harness.memory.forget(id)
        return f"archived {id}" if ok else f"error: no memory entry {id!r}"


class MemoryConsolidateTool(Tool):
    name = "memory_consolidate"
    description = "Run incremental memory consolidation now (merges near-duplicates)."
    parameters = {"type": "object", "properties": {}}
    profiles = ["memory"]

    def __call__(self, **_: Any) -> str:
        report = self.harness.memory.consolidate(self.harness.models.client_for("compression"))
        if report.get("skipped"):
            return f"skipped: {report['skipped']}"
        return (
            f"merged {report.get('merged', 0)} pair(s) of {report.get('considered', 0)} considered"
        )


class SessionsListTool(Tool):
    name = "sessions_list"
    description = "List sessions visible to this copy."
    parameters = {"type": "object", "properties": {}}
    profiles = ["sessions"]

    def __call__(self, **_: Any) -> str:
        sessions = self.harness.sessions.list(visible_to=self.harness.config.get("agent.id"))
        if not sessions:
            return "no sessions"
        lines = []
        for info in sessions:
            lines.append(
                f"- {info['id']} [{info.get('status')}] name={info.get('name')!r} "
                f"agent={info.get('agent_id')} last={info.get('last_activity')}"
            )
        return "\n".join(lines)


class SessionReadTool(Tool):
    name = "session_read"
    description = "Read the transcript of another session."
    parameters = {
        "type": "object",
        "properties": {
            "session_id": {"type": "string"},
            "last_n": {"type": "integer", "description": "Message count (default 30)"},
        },
        "required": ["session_id"],
    }
    profiles = ["sessions"]

    def __call__(self, session_id: str, last_n: int = 30, **_: Any) -> str:
        try:
            session = self.harness.sessions.load(session_id)
        except FileNotFoundError:
            return f"error: no session {session_id!r}"
        return session.transcript(last_n=last_n) or "(empty session)"


class SessionSendTool(Tool):
    name = "session_send"
    description = "Send a message to another session's inbox (delivery receipts apply)."
    parameters = {
        "type": "object",
        "properties": {
            "session_id": {"type": "string"},
            "message": {"type": "string"},
            "persist": {"type": "boolean", "description": "Survive inbox eviction (default false)"},
        },
        "required": ["session_id", "message"],
    }
    profiles = ["sessions"]

    def __call__(self, session_id: str, message: str, persist: bool = False, **_: Any) -> str:
        result = self.harness.sessions.inject_message(
            session_id,
            message,
            from_agent=self.harness.config.get("agent.id"),
            from_session=self.harness.session_id,
            persist=persist,
        )
        if result.get("status") == "error":
            return f"error: {result.get('error')}"
        previous = " (previous message was read)" if result.get("previous_read") else ""
        return f"{result.get('status')} to {session_id}{previous}"


class TaskTool(Tool):
    name = "task"
    description = "Spawn a depth-1 worker with its own session to handle a self-contained subtask."
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "name": {"type": "string"},
            "tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Restrict the worker to these tools (default: parent's tools)",
            },
        },
        "required": ["prompt"],
    }
    profiles = ["delegation"]

    def __call__(
        self, prompt: str, name: str | None = None, tools: list[str] | None = None, **_: Any
    ) -> str:
        return _text(self.harness.spawn_worker(prompt, name=name, tools=tools))


class EvalsRunTool(Tool):
    name = "evals_run"
    description = "Schedule an eval run as a background job; returns the job id."
    parameters = {"type": "object", "properties": {}}
    profiles = ["evals"]

    def __call__(self, **_: Any) -> str:
        return _text(self.harness.schedule_eval_job())


class DiagnoseRunTool(Tool):
    name = "diagnose_run"
    description = "Schedule a contrastive diagnosis job; returns the job id."
    parameters = {"type": "object", "properties": {}}
    profiles = ["evals"]

    def __call__(self, **_: Any) -> str:
        return _text(self.harness.schedule_diagnose_job())


class JobSpawnTool(Tool):
    name = "job_spawn"
    description = "Spawn a long-running background shell job; returns the job id."
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "name": {"type": "string"},
            "cwd": {"type": "string"},
            "timeout_s": {"type": "integer", "description": "Optional wall-clock limit"},
        },
        "required": ["command"],
    }
    profiles = ["jobs"]

    def __call__(
        self,
        command: str,
        name: str | None = None,
        cwd: str | None = None,
        timeout_s: int | None = None,
        **_: Any,
    ) -> str:
        return _text(self.harness.jobs.spawn(command, name=name, cwd=cwd, timeout_s=timeout_s))


class JobListTool(Tool):
    name = "job_list"
    description = "List background jobs."
    parameters = {"type": "object", "properties": {}}
    profiles = ["jobs"]

    def __call__(self, **_: Any) -> str:
        return _text(self.harness.jobs.list())


class JobStatusTool(Tool):
    name = "job_status"
    description = "Status of one job."
    parameters = {
        "type": "object",
        "properties": {"job_id": {"type": "string"}},
        "required": ["job_id"],
    }
    profiles = ["jobs"]

    def __call__(self, job_id: str, **_: Any) -> str:
        return _text(self.harness.jobs.status(job_id))


class JobTailTool(Tool):
    name = "job_tail"
    description = "Tail the output log of a job."
    parameters = {
        "type": "object",
        "properties": {
            "job_id": {"type": "string"},
            "n": {"type": "integer", "description": "Lines (default 50)"},
        },
        "required": ["job_id"],
    }
    profiles = ["jobs"]

    def __call__(self, job_id: str, n: int = 50, **_: Any) -> str:
        return _text(self.harness.jobs.tail(job_id, n=n))


class JobKillTool(Tool):
    name = "job_kill"
    description = "Kill a running job (process group)."
    parameters = {
        "type": "object",
        "properties": {"job_id": {"type": "string"}},
        "required": ["job_id"],
    }
    profiles = ["jobs"]

    def __call__(self, job_id: str, **_: Any) -> str:
        return _text(self.harness.jobs.kill(job_id))


class JobLogTool(Tool):
    name = "job_log"
    description = "Read the full output log path and last lines of a job."
    parameters = {
        "type": "object",
        "properties": {"job_id": {"type": "string"}},
        "required": ["job_id"],
    }
    profiles = ["jobs"]

    def __call__(self, job_id: str, **_: Any) -> str:
        return _text(self.harness.jobs.log(job_id))


BUILTIN_TOOLS: list[type[Tool]] = [
    ReadTool,
    WriteTool,
    EditTool,
    BashTool,
    ProposeTool,
    RevertTool,
    CompressTool,
    MemorySearchTool,
    MemorySaveTool,
    MemoryForgetTool,
    MemoryConsolidateTool,
    SessionsListTool,
    SessionReadTool,
    SessionSendTool,
    TaskTool,
    EvalsRunTool,
    DiagnoseRunTool,
    JobSpawnTool,
    JobListTool,
    JobStatusTool,
    JobTailTool,
    JobKillTool,
    JobLogTool,
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Holds built-ins and extensions, and watches ``extensions/``."""

    def __init__(self, harness: Any, load_extensions: bool = True) -> None:
        self.harness = harness
        self.config: Config = harness.config
        self.log = harness.log
        self.governance: Governance = harness.governance
        self._lock = threading.RLock()
        self._tools: dict[str, Tool] = {}
        self._builtin_names: set[str] = set()
        self._sources: dict[str, Path] = {}
        self._source_texts: dict[str, str] = {}
        self._mtimes: dict[Path, float] = {}
        self._reload_errors: dict[str, str] = {}
        self._module_tools: dict[str, list[str]] = {}
        self._namespaces: dict[str, types.ModuleType] = {}
        self._disabled: set[str] = set()
        self._watcher: threading.Thread | None = None
        self._stop = threading.Event()
        self._observer: Any = None
        self._pending: set[Path] = set()
        self._quiet_until = 0.0
        self._load_builtins()
        if load_extensions:
            self.scan_extensions(initial=True)

    # -- paths -----------------------------------------------------------

    @property
    def extensions_dir(self) -> Path:
        return self.config.state_path / "extensions"

    @property
    def archive_dir(self) -> Path:
        return self.extensions_dir / "archive"

    @property
    def tests_dir(self) -> Path:
        return self.extensions_dir / "tests"

    # -- registration ----------------------------------------------------

    def _load_builtins(self) -> None:
        for cls in BUILTIN_TOOLS:
            tool = cls(self.harness)
            self._tools[tool.name] = tool
            self._builtin_names.add(tool.name)

    def register(self, tool: Tool, replace: bool = False) -> bool:
        with self._lock:
            if tool.name in self._tools and not replace:
                return False
            self._tools[tool.name] = tool
            return True

    def unregister(self, name: str) -> None:
        with self._lock:
            self._tools.pop(name, None)

    def restrict(self, names: list[str] | None) -> list[str]:
        """Keep only the named tools (plus registered builtins when None).

        Returns the names that were removed.
        """
        if names is None:
            return []
        keep = set(names)
        removed: list[str] = []
        with self._lock:
            for name in list(self._tools):
                if name not in keep:
                    self._tools.pop(name, None)
                    removed.append(name)
        return removed

    def get(self, name: str) -> Tool | None:
        with self._lock:
            return self._tools.get(name)

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._tools)

    def all(self) -> dict[str, Tool]:
        with self._lock:
            return dict(self._tools)

    def disabled(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        with self._lock:
            for name, tool in self._tools.items():
                missing = tool.missing_requires()
                if missing:
                    out[name] = missing
                elif name in self._disabled:
                    out[name] = ["disabled by operator"]
        return out

    def enable(self, name: str) -> bool:
        """Operator enable/disable (``POST /agent/tools/{name}/enable``)."""
        with self._lock:
            if name not in self._tools:
                return False
            self._disabled.discard(name)
        return True

    def disable(self, name: str) -> bool:
        with self._lock:
            if name not in self._tools:
                return False
            self._disabled.add(name)
        return True

    def is_enabled(self, name: str) -> bool:
        with self._lock:
            return name in self._tools and name not in self._disabled

    def available(self) -> dict[str, Tool]:
        enabled: dict[str, Tool] = {}
        for name, tool in self.all().items():
            if not tool.missing_requires() and name not in self._disabled:
                enabled[name] = tool
        return enabled

    def specs(self, only: list[str] | None = None) -> list[dict[str, Any]]:
        """OpenAI tool schemas, sorted for prompt-cache stability."""
        enabled = self.available()
        chosen = sorted(only) if only else sorted(enabled)
        out = []
        for name in chosen:
            tool = enabled.get(name)
            if tool is None:
                continue
            out.append(tool.spec())
        return out

    def profiles(self) -> dict[str, list[str]]:
        return {name: list(tool.profiles) for name, tool in sorted(self.available().items())}

    # -- calling ---------------------------------------------------------

    def call(self, name: str, args: dict[str, Any] | None = None) -> str:
        tool = self.get(name)
        if tool is None:
            return f"error: unknown tool {name!r}"
        if not self.is_enabled(name):
            return f"error: tool {name!r} is disabled"
        missing = tool.missing_requires()
        if missing:
            return f"error: tool {name!r} disabled; missing env: {', '.join(missing)}"
        kwargs = dict(args or {})
        try:
            result = tool(**kwargs)
        except TypeError as exc:
            return f"error: bad arguments for {name}: {exc}"
        except Exception as exc:  # noqa: BLE001 - never raise into the loop
            self.log.warn("tool_error", tool=name, error=f"{type(exc).__name__}: {exc}")
            result = f"error: {type(exc).__name__}: {exc}"
        text = result if isinstance(result, str) else str(result)
        note = self._reload_errors.get(name)
        if note:
            text = f"[tool reload failed: {note}]\n{text}"
        return _clip(text)

    # -- extensions ------------------------------------------------------

    def scan_extensions(self, initial: bool = False) -> list[str]:
        """Discover .py files; returns the tool names currently loaded."""
        directory = self.extensions_dir
        if not directory.is_dir():
            return []
        found: set[Path] = set()
        for path in sorted(directory.glob("*.py")):
            if path.name.startswith("_"):
                continue
            found.add(path)
            mtime = path.stat().st_mtime
            previous = self._mtimes.get(path)
            if previous is None:
                self._load_extension(path, initial=initial)
            elif abs(mtime - previous) > 1e-6:
                self._pending.add(path)
        for path in list(self._mtimes):
            if path not in found:
                self._remove_extension(path)
        self._mtimes = {p: p.stat().st_mtime for p in found}
        self._process_pending()
        return self.names()

    def _process_pending(self) -> None:
        now = time.time()
        due = []
        for path in list(self._pending):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                self._pending.discard(path)
                continue
            if now - mtime >= float(self.config.get("tools.reload_debounce_s", 1.0)):
                due.append(path)
        for path in due:
            self._pending.discard(path)
            self._load_extension(path, initial=False)

    def _load_extension(self, path: Path, initial: bool) -> None:
        name = path.stem
        try:
            source = path.read_text(encoding="utf-8")
            # Compile from source text: module_from_spec + exec_module can serve
            # stale bytecode from __pycache__ for same-second same-size edits.
            module = types.ModuleType(f"canary_extension_{name}")
            module.__file__ = str(path)
            code = compile(source, str(path), "exec")
            exec(code, module.__dict__)  # noqa: S102 - loading trusted local extensions
            tools = self._instantiate(module)
            if not tools:
                raise ImportError(f"{path.name} defines no Tool subclass")
            tests_ok, test_error = self.run_self_tests(name)
            if not tests_ok:
                raise RuntimeError(f"self-tests failed: {test_error}")
            old_names = self._module_tools.get(name, [])
            if not initial and old_names:
                self._archive(name)
            self._remove_module_tools(name)
            for tool in tools:
                if tool.name in self._builtin_names:
                    raise ValueError(f"extension {name!r} may not shadow built-in {tool.name!r}")
                self.register(tool, replace=True)
            self._module_tools[name] = [t.name for t in tools]
            self._sources[name] = path
            self._source_texts[name] = source
            self._namespaces[name] = module
            self._reload_errors.pop(name, None)
            self.log.event(
                "extension_loaded",
                extension=name,
                tools=[t.name for t in tools],
                version="initial" if initial else "reload",
            )
        except Exception as exc:  # noqa: BLE001 - keep the old version live
            error = f"{type(exc).__name__}: {exc}"
            self._reload_errors[name] = error
            if initial:
                self._remove_module_tools(name)
            self.log.warn("extension_reload_failed", extension=name, error=error)
            for tool_name in self._module_tools.get(name, []):
                self._reload_errors[tool_name] = error
            if self.harness is not None:
                try:
                    self.harness.last_tool_error = f"[tool reload failed: {error}]"
                except Exception:  # noqa: BLE001
                    pass

    def _instantiate(self, module: Any) -> list[Tool]:
        tools: list[Tool] = []
        for attr in vars(module).values():
            if (
                isinstance(attr, type)
                and issubclass(attr, Tool)
                and attr is not Tool
                and getattr(attr, "name", "")
            ):
                tools.append(attr(self.harness))
        return tools

    def _remove_module_tools(self, name: str) -> None:
        for tool_name in self._module_tools.pop(name, []):
            if tool_name not in self._builtin_names:
                self.unregister(tool_name)

    def _remove_extension(self, path: Path) -> None:
        name = path.stem
        self._remove_module_tools(name)
        self._sources.pop(name, None)
        self._namespaces.pop(name, None)
        self._mtimes.pop(path, None)
        self._reload_errors.pop(name, None)
        self.log.event("extension_removed", extension=name)

    def _archive(self, name: str) -> None:
        """Archive the previously loaded source as {name}_v{N}.py."""
        previous = self._source_texts.get(name)
        if not previous:
            return
        ensure_dir(self.archive_dir)
        versions = []
        for candidate in self.archive_dir.glob(f"{name}_v*.py"):
            try:
                versions.append(int(candidate.stem.rsplit("_v", 1)[1]))
            except (IndexError, ValueError):
                continue
        version = max(versions, default=0) + 1
        target = self.archive_dir / f"{name}_v{version}.py"
        try:
            atomic_write_text(target, previous)
        except OSError as exc:
            self.log.warn("extension_archive_failed", extension=name, error=str(exc))
            return
        keep = int(self.config.get("tools.archive_keep", 10))
        for candidate in sorted(
            self.archive_dir.glob(f"{name}_v*.py"),
            key=lambda p: int(p.stem.rsplit("_v", 1)[1] or 0),
            reverse=True,
        )[keep:]:
            try:
                candidate.unlink()
            except OSError:
                pass
        self.log.event("extension_archived", extension=name, version=version)

    def rollback(self, name: str) -> str:
        """Swap in the most recent archived version of an extension."""
        versions = []
        for candidate in self.archive_dir.glob(f"{name}_v*.py"):
            try:
                versions.append((int(candidate.stem.rsplit("_v", 1)[1]), candidate))
            except (IndexError, ValueError):
                continue
        if not versions:
            return f"error: no archived versions for {name!r}"
        _, source = max(versions)
        target = self.extensions_dir / f"{name}.py"
        try:
            shutil.copy2(source, target)
        except OSError as exc:
            return f"error: rollback failed: {exc}"
        self._quiet_until = time.monotonic() + 2.0
        self._load_extension(target, initial=False)
        if name in self._reload_errors:
            return f"error: rollback load failed: {self._reload_errors[name]}"
        return f"rolled back {name} to {source.name}"

    # -- self-tests ------------------------------------------------------

    def self_test_path(self, name: str) -> Path:
        return self.tests_dir / f"{name}_test.py"

    def run_self_tests(self, name: str, timeout: int = 60) -> tuple[bool, str]:
        test_path = self.self_test_path(name)
        if not test_path.is_file():
            return True, ""
        import tempfile

        with tempfile.TemporaryDirectory(prefix="canary_extension_test_") as tmp:
            env = dict(os.environ)
            env["CANARY_ROOT"] = tmp
            env["HARNESS_STATE_PATH"] = str(Path(tmp) / "shared")
            env.pop("HARNESS_MODEL_NAME", None)
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", str(test_path), "-x", "-q"],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                    cwd=tmp,
                )
            except subprocess.TimeoutExpired:
                return False, f"self-tests timed out after {timeout}s"
            except OSError as exc:
                return False, f"self-tests could not run: {exc}"
            if proc.returncode != 0:
                tail = (proc.stdout or "") + (proc.stderr or "")
                return False, tail.strip().splitlines()[-1] if tail.strip() else "pytest failed"
            return True, ""

    # -- watching --------------------------------------------------------

    def start_watcher(self) -> None:
        if self._watcher is not None:
            return
        self._stop.clear()
        ensure_dir(self.extensions_dir)
        self._try_watchdog()
        self._watcher = threading.Thread(
            target=self._watch_loop, name="canary-extensions", daemon=True
        )
        self._watcher.start()

    def stop_watcher(self) -> None:
        self._stop.set()
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=2)
            except Exception:  # noqa: BLE001
                pass
            self._observer = None
        if self._watcher is not None:
            self._watcher.join(timeout=3)
            self._watcher = None

    def _try_watchdog(self) -> None:
        try:
            from watchdog.events import FileSystemEventHandler  # noqa: F401
            from watchdog.observers import Observer
        except ImportError:
            return
        registry = self

        class Handler(FileSystemEventHandler):
            def on_any_event(self, event: Any) -> None:
                src = getattr(event, "dest_path", None) or getattr(event, "src_path", "")
                if src.endswith(".py"):
                    registry._pending.add(Path(src))

        try:
            observer = Observer()
            observer.schedule(Handler(), str(self.extensions_dir), recursive=False)
            observer.daemon = True
            observer.start()
            self._observer = observer
        except Exception as exc:  # noqa: BLE001 - fall back to polling
            self.log.warn("extension_watchdog_failed", error=str(exc))

    def _watch_loop(self) -> None:
        interval = 1.0
        while not self._stop.wait(interval):
            if time.monotonic() < self._quiet_until:
                continue
            try:
                self.scan_extensions()
            except Exception as exc:  # noqa: BLE001
                self.log.warn("extension_scan_failed", error=str(exc))

    # -- introspection ---------------------------------------------------

    def extension_modules(self) -> dict[str, types.ModuleType]:
        """Loaded extension namespaces (used by evals for custom checks)."""
        return dict(self._namespaces)

    def info(self) -> dict[str, Any]:
        return {
            "tools": self.names(),
            "builtin": sorted(self._builtin_names),
            "extensions": {name: names for name, names in sorted(self._module_tools.items())},
            "disabled": self.disabled(),
            "profiles": self.profiles(),
            "reload_errors": dict(self._reload_errors),
        }
