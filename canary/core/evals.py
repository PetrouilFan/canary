"""Eval system: effectiveness measurement, canary gate, diagnosis (spec §4.15).

Tests prove correctness; evals measure effectiveness.  The eval set lives in
``shared/evals/*.yaml`` (held out: preflight never reads it), results in
``shared/data/evals.jsonl``, baselines in ``shared/data/eval_cache/``.
"""

from __future__ import annotations

import hashlib
import json
import queue
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import Config
from .observability import Log
from .util import (
    atomic_write_text,
    ensure_dir,
    mono,
    read_json,
    read_jsonl,
    utc_now,
    utc_stamp,
)

CHECK_TYPES = ("contains", "regex", "equals", "file_exists", "file_contains")
#: Prefix marking a check path as relative to the eval agent's *state* directory
#: (its own records: ``logs/harness.log``, ``metrics.jsonl``, ...) rather than
#: the scratch workspace the task is scored in.
AUDIT_PREFIX = "audit:"
DEFAULT_TIMEOUT_S = 300


class EvalCheck:
    """Interface for custom check types provided by extensions."""

    type: str = ""

    def check(self, value: Any, response: str, workspace: Path) -> tuple[bool, str]:
        raise NotImplementedError


@dataclass
class EvalTask:
    id: str
    prompt: str
    tags: list[str] = field(default_factory=list)
    setup: str | None = None
    timeout_s: int = DEFAULT_TIMEOUT_S
    model: str | None = None
    check: list[dict[str, Any]] = field(default_factory=list)
    path: Path | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: Path | None = None) -> EvalTask:
        if not data.get("id"):
            raise ValueError("eval task requires an id")
        if not data.get("prompt"):
            raise ValueError("eval task requires a prompt")
        checks = data.get("check") or []
        if not isinstance(checks, list) or not checks:
            raise ValueError("eval task requires at least one check")
        return cls(
            id=str(data["id"]),
            prompt=str(data["prompt"]),
            tags=[str(t) for t in (data.get("tags") or [])],
            setup=data.get("setup"),
            timeout_s=int(data.get("timeout_s") or DEFAULT_TIMEOUT_S),
            model=data.get("model"),
            check=[dict(c) for c in checks],
            path=path,
        )

    @classmethod
    def from_file(cls, path: Path) -> EvalTask:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"invalid eval task file: {path}")
        return cls.from_dict(data, path=path)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "prompt": self.prompt}
        if self.tags:
            out["tags"] = list(self.tags)
        if self.setup:
            out["setup"] = self.setup
        if self.timeout_s != DEFAULT_TIMEOUT_S:
            out["timeout_s"] = self.timeout_s
        if self.model:
            out["model"] = self.model
        out["check"] = list(self.check)
        return out

    def identity(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "setup": self.setup,
            "check": self.check,
            "model": self.model,
        }


def eval_set_hash(tasks: list[EvalTask]) -> str:
    payload = json.dumps([t.identity() for t in tasks], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def resolve_check_path(
    raw: Any, workspace: Path, state_path: Path | None = None
) -> Path:
    """Resolve a check path, ``audit:`` meaning the eval agent's state dir.

    The agent's own audit records live in its state directory, outside the
    workspace a check can otherwise see, so a check has to say when it wants
    them: ``audit:logs/harness.log``.  Every other path stays workspace-relative.
    """
    text = str(raw)
    if text.startswith(AUDIT_PREFIX):
        base = state_path if state_path is not None else workspace
        return base / text[len(AUDIT_PREFIX):]
    return workspace / text


def _run_check(
    spec: dict[str, Any],
    response: str,
    workspace: Path,
    custom: dict[str, EvalCheck],
    state_path: Path | None = None,
) -> dict[str, Any]:
    """Score one check spec.

    ``file_exists``/``file_contains`` resolve in the scratch workspace, or in the
    eval agent's state directory when the path carries the ``audit:`` prefix.
    Extension checks keep the ``(value, response, workspace)`` interface; they
    can call :func:`resolve_check_path` for the same resolution.
    """
    ctype = str(spec.get("type") or "")
    value = spec.get("value")
    path = spec.get("path")
    try:
        if ctype == "contains":
            ok = str(value) in response
        elif ctype == "regex":
            ok = re.search(str(value), response) is not None
        elif ctype == "equals":
            ok = response.strip() == str(value).strip()
        elif ctype == "file_exists":
            ok = resolve_check_path(path, workspace, state_path).exists()
        elif ctype == "file_contains":
            target = resolve_check_path(path, workspace, state_path)
            ok = target.exists() and str(value) in target.read_text(
                encoding="utf-8", errors="replace"
            )
        elif ctype in custom:
            ok, _ = custom[ctype].check(value, response, workspace)
        else:
            return {"check": spec, "pass": False, "detail": f"unknown check type: {ctype}"}
    except (OSError, ValueError, re.error) as exc:
        return {"check": spec, "pass": False, "detail": f"check error: {exc}"}
    return {"check": spec, "pass": bool(ok), "detail": ""}


class Evals:
    def __init__(self, config: Config, log: Log, registry: Any = None) -> None:
        self.config = config
        self.log = log
        self.registry = registry
        self.data = config.data_path
        self.results_file = self.data / "evals.jsonl"
        self.summary_file = self.data / "evals_summary.jsonl"
        self.cache_dir = ensure_dir(self.data / "eval_cache")
        self.candidates_dir = ensure_dir(self.data / "eval_candidates")

    # -- task loading ------------------------------------------------------

    def evals_dir(self) -> Path:
        return self.config.state_path / "evals"

    def load_tasks(
        self,
        tag: str | None = None,
        limit: int | None = None,
        path: Path | None = None,
    ) -> list[EvalTask]:
        directory = path or self.evals_dir()
        tasks: list[EvalTask] = []
        if directory.is_dir():
            for file in sorted(directory.glob("*.yaml")):
                try:
                    task = EvalTask.from_file(file)
                except (OSError, ValueError, yaml.YAMLError) as exc:
                    self.log.warn("eval_task_invalid", path=str(file), error=str(exc))
                    continue
                if tag and tag not in task.tags:
                    continue
                tasks.append(task)
                if limit and len(tasks) >= limit:
                    break
        return tasks

    def custom_checks(self) -> dict[str, EvalCheck]:
        checks: dict[str, EvalCheck] = {}
        if self.registry is None:
            return checks
        getter = getattr(self.registry, "extension_modules", None)
        modules: Any = getter() if callable(getter) else {}
        if not isinstance(modules, dict):
            return checks
        for module in modules.values():
            for item in vars(module).values():
                if (
                    isinstance(item, type)
                    and item is not EvalCheck
                    and issubclass(item, EvalCheck)
                    and getattr(item, "type", "")
                ):
                    try:
                        checks[item.type] = item()
                    except TypeError:
                        continue
        return checks

    # -- running -----------------------------------------------------------

    def run_task(
        self,
        task: EvalTask,
        *,
        run_id: str,
        model_role: str,
        release_id: str,
        workspace_root: Path | None = None,
    ) -> dict[str, Any]:
        import tempfile

        from .agent import Agent  # lazy: avoids an import cycle at module load

        role = task.model or model_role
        timeout = int(task.timeout_s or DEFAULT_TIMEOUT_S)
        started = mono()
        tokens = 0
        checks: list[dict[str, Any]] = []
        response = ""
        status = "ok"
        reason = ""
        with tempfile.TemporaryDirectory(prefix="canary-eval-") as tmp:
            scratch = Path(tmp)
            workspace = scratch / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            if task.setup:
                try:
                    subprocess.run(
                        task.setup,
                        shell=True,
                        cwd=workspace,
                        capture_output=True,
                        timeout=min(timeout, 60),
                        start_new_session=True,
                    )
                except (subprocess.TimeoutExpired, OSError) as exc:
                    status = "failed"
                    reason = f"setup failed: {exc}"
            if status == "ok":
                agent = Agent(
                    config=self.config,
                    state_path=scratch / "state",
                    ephemeral=True,
                    self_modify=False,
                    workspace=workspace_root or workspace,
                )
                try:
                    response = self._run_with_timeout(agent, task.prompt, role, timeout)
                except TimeoutError:
                    status = "failed"
                    reason = f"timeout after {timeout}s"
                except Exception as exc:  # noqa: BLE001 - eval failures are data
                    status = "failed"
                    reason = f"agent error: {exc}"
                finally:
                    tokens = int(getattr(agent, "last_total_tokens", 0) or 0)
                    try:
                        agent.close()
                    except Exception:  # noqa: BLE001
                        pass
            if status == "ok":
                custom = self.custom_checks()
                # `audit:` reads wherever this agent actually writes its own
                # records.  Agent(state_path=...) is not authoritative (a Config
                # argument wins), so ask the agent instead of the loop.
                audit_state = Path(
                    getattr(agent.config, "state_path", None) or scratch / "state"
                )
                checks = [
                    _run_check(c, response, workspace, custom, state_path=audit_state)
                    for c in task.check
                ]
                if not all(c["pass"] for c in checks):
                    status = "failed"
                    failed = [c for c in checks if not c["pass"]]
                    reason = f"{len(failed)}/{len(checks)} checks failed"
            else:
                checks = [
                    {"check": c, "pass": False, "detail": reason} for c in task.check
                ]
            if not checks:
                checks = [{"check": c, "pass": status == "ok", "detail": reason}
                          for c in task.check]
        latency_ms = int((mono() - started) * 1000)
        passed = int(sum(1 for c in checks if c["pass"]))
        result = {
            "timestamp": utc_now(),
            "run_id": run_id,
            "release_id": release_id,
            "task_id": task.id,
            "pass": status == "ok" and passed == len(checks),
            "checks_passed": passed,
            "checks_total": len(checks),
            "latency_ms": latency_ms,
            "tokens": tokens,
            "model_role": role,
            "agent_id": self.config.get("agent.id"),
            "status": status,
            "reason": reason,
            "response": response[:4000],
            "checks": checks,
        }
        return result

    def _run_with_timeout(self, agent: Any, prompt: str, role: str,
                          timeout: int) -> str:
        box: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def target() -> None:
            try:
                box.put((True, agent.run(prompt, model=role)))
            except Exception as exc:  # noqa: BLE001
                box.put((False, exc))

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        try:
            ok, value = box.get(timeout=timeout)
        except queue.Empty as exc:
            cancel = getattr(agent, "cancel", None)
            if callable(cancel):
                try:
                    cancel()
                except Exception:  # noqa: BLE001
                    pass
            raise TimeoutError(f"turn exceeded {timeout}s") from exc
        if not ok:
            raise value
        return str(value)

    def run(
        self,
        tasks: list[EvalTask] | None = None,
        *,
        tag: str | None = None,
        limit: int | None = None,
        model_role: str | None = None,
        release_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        tasks = tasks if tasks is not None else self.load_tasks(tag=tag, limit=limit)
        role = model_role or str(self.config.get("evals.model", "main"))
        release = release_id or self.config.release_id
        run = run_id or uuid.uuid4().hex[:12]
        results = []
        for task in tasks:
            result = self.run_task(
                task, run_id=run, model_role=role, release_id=release
            )
            results.append(result)
            self.log.eval_result(**{k: result[k] for k in (
                "run_id", "release_id", "task_id", "pass", "checks_passed",
                "checks_total", "latency_ms", "tokens", "model_role", "agent_id",
            )})
        passes = sum(1 for r in results if r["pass"])
        tokens = sum(int(r.get("tokens") or 0) for r in results)
        report = {
            "run_id": run,
            "release_id": release,
            "model_role": role,
            "total": len(results),
            "passes": passes,
            "pass_rate": round(passes / len(results), 4) if results else 0.0,
            "tokens": tokens,
            "results": results,
        }
        self.log.event(
            "evals_run", run_id=run, release_id=release, total=len(results),
            passes=passes, tokens=tokens,
        )
        return report

    # -- baseline cache ----------------------------------------------------

    def cache_path(self, release_id: str, role: str, tasks: list[EvalTask]) -> Path:
        digest = eval_set_hash(tasks)
        return self.cache_dir / f"{release_id}.{digest}.{role}.json"

    def cached_baseline(
        self, release_id: str, role: str, tasks: list[EvalTask]
    ) -> dict[str, Any] | None:
        path = self.cache_path(release_id, role, tasks)
        data = read_json(path, default=None)
        if not isinstance(data, dict):
            return None
        max_age_h = float(self.config.get("evals.cache_max_age_h", 168) or 168)
        age_h = (time.time() - float(data.get("cached_at_unix") or 0)) / 3600.0
        if age_h > max_age_h:
            self.log.info("eval_cache_stale", release_id=release_id, age_h=round(age_h, 1))
            return None
        return data

    def store_baseline(
        self, release_id: str, role: str, tasks: list[EvalTask],
        report: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "release_id": release_id,
            "model_role": role,
            "eval_set_hash": eval_set_hash(tasks),
            "cached_at": utc_now(),
            "cached_at_unix": time.time(),
            "pass_rate": report["pass_rate"],
            "results": [
                {k: r[k] for k in ("task_id", "pass", "checks_passed",
                                   "checks_total", "tokens", "latency_ms")}
                for r in report["results"]
            ],
        }
        atomic_write_text(
            self.cache_path(release_id, role, tasks),
            json.dumps(payload, indent=2),
        )
        return payload

    def baseline(
        self, release_id: str, role: str, tasks: list[EvalTask],
        *, measure: bool = True,
    ) -> dict[str, Any]:
        cached = self.cached_baseline(release_id, role, tasks)
        if cached is not None:
            return cached
        if not measure:
            return {"release_id": release_id, "model_role": role, "pass_rate": None,
                    "results": [], "measured": False}
        report = self.run(tasks, model_role=role, release_id=release_id)
        return {**self.store_baseline(release_id, role, tasks, report),
                "measured": True}

    # -- canary gate -------------------------------------------------------

    def canary_gate(
        self,
        candidate_release_id: str,
        baseline_release_id: str,
        *,
        tasks: list[EvalTask] | None = None,
        role: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        n = int(self.config.get("evals.canary_tasks", 3) or 0)
        if n <= 0 and not force:
            return {"enabled": False, "delta": None, "flagged": False,
                    "tasks": [], "reason": "evals.canary_tasks=0"}
        role = role or str(self.config.get("evals.model", "main"))
        tasks = tasks or self.load_tasks(limit=n)
        if not tasks:
            return {"enabled": False, "delta": None, "flagged": False,
                    "tasks": [], "reason": "no eval tasks"}
        baseline = self.baseline(baseline_release_id, role, tasks)
        candidate = self.run(tasks, model_role=role, release_id=candidate_release_id)
        self.store_baseline(candidate_release_id, role, tasks, candidate)
        delta = None
        if baseline.get("pass_rate") is not None:
            delta = round(candidate["pass_rate"] - float(baseline["pass_rate"]), 4)
        tolerance = float(self.config.get("evals.tolerance", 0.10) or 0.10)
        flagged = bool(delta is not None and delta < -tolerance)
        return {
            "enabled": True,
            "delta": delta,
            "flagged": flagged,
            "pass_rate": candidate["pass_rate"],
            "baseline_pass_rate": baseline.get("pass_rate"),
            "tolerance": tolerance,
            "tasks": [t.id for t in tasks],
            "run_id": candidate["run_id"],
        }

    # -- aggregation -------------------------------------------------------

    def results(self, limit: int = 100) -> list[dict[str, Any]]:
        lines = list(read_jsonl(self.results_file))
        return lines[-max(1, int(limit)):]

    def write_summary(self) -> list[dict[str, Any]]:
        """Daily rollup per (release, agent) with cross-copy divergence flags."""
        groups: dict[tuple[str, str], dict[str, Any]] = {}
        rows = list(read_jsonl(self.results_file))
        for row in rows:
            key = (str(row.get("release_id") or "dev"), str(row.get("agent_id") or "?"))
            group = groups.setdefault(key, {"runs": 0, "passes": 0, "tokens": 0})
            group["runs"] += 1
            group["passes"] += 1 if row.get("pass") else 0
            group["tokens"] += int(row.get("tokens") or 0)
        rates = [
            (g["passes"] / g["runs"]) if g["runs"] else 0.0 for g in groups.values()
        ]
        median = sorted(rates)[len(rates) // 2] if rates else 0.0
        tolerance = float(self.config.get("evals.tolerance", 0.10) or 0.10)
        out = []
        for (release, agent_id), group in sorted(groups.items()):
            rate = (group["passes"] / group["runs"]) if group["runs"] else 0.0
            row = {
                "timestamp": utc_now(),
                "release_id": release,
                "agent_id": agent_id,
                "runs": group["runs"],
                "passes": group["passes"],
                "pass_rate": round(rate, 4),
                "tokens": group["tokens"],
                "divergence": bool(rows and abs(rate - median) > tolerance),
            }
            self.log.eval_summary(**row)
            out.append(row)
        return out

    # -- candidates & diagnosis -------------------------------------------

    def write_candidate(self, data: dict[str, Any], index: int = 0) -> Path | None:
        try:
            task = EvalTask.from_dict(data)
        except ValueError as exc:
            self.log.warn("eval_candidate_invalid", error=str(exc))
            return None
        path = self.candidates_dir / f"{utc_stamp()}-{index}.yaml"
        atomic_write_text(path, yaml.safe_dump(task.to_dict(), sort_keys=False))
        self.log.info("eval_candidate_written", path=str(path), task_id=task.id)
        return path

    def diagnose(
        self,
        *,
        model_client: Any,
        sessions: Any,
        memory: Any = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Contrastive diagnosis job: mine outcome signals, propose candidates."""
        max_candidates = int(self.config.get("diagnosis.max_candidates", 10) or 10)
        signals = self._collect_signals(sessions, limit=limit)
        if not signals:
            return {"candidates": 0, "signals": 0, "reason": "no outcome signals"}
        prompt = _DIAGNOSIS_PROMPT.replace(
            "{signals}", json.dumps(signals, indent=2)[:24000]
        ).replace("{n}", str(max_candidates))
        candidates: list[dict[str, Any]] = []
        try:
            response = model_client.call(
                messages=[{"role": "user", "content": prompt}],
                tools=None,
            )
            text = response.content if hasattr(response, "content") else str(response)
            candidates = _parse_candidates(text)
        except Exception as exc:  # noqa: BLE001 - diagnosis is best effort
            self.log.warn("diagnosis_failed", error=str(exc))
            return {"candidates": 0, "signals": len(signals), "error": str(exc)}
        written = []
        for i, candidate in enumerate(candidates[:max_candidates]):
            path = self.write_candidate(candidate, index=i)
            if path is not None:
                written.append(str(path))
                if memory is not None and candidate.get("failure"):
                    try:
                        memory.save(
                            f"Recurring failure pattern ({candidate.get('id')}): "
                            f"{candidate.get('failure')}",
                            tags=["diagnosis"],
                            source="diagnosis",
                        )
                    except Exception:  # noqa: BLE001
                        pass
        return {"candidates": len(written), "signals": len(signals),
                "paths": written}

    def _collect_signals(self, sessions: Any, limit: int = 20) -> list[dict[str, Any]]:
        signals: list[dict[str, Any]] = []
        try:
            listing = sessions.list()
        except Exception:  # noqa: BLE001
            return signals
        infos = listing.get("sessions", listing) if isinstance(listing, dict) else listing
        for info in list(infos)[:50]:
            session_id = info.get("id") if isinstance(info, dict) else str(info)
            if not session_id:
                continue
            try:
                session = sessions.load(session_id)
                events = session.events(last_n=20)
            except Exception:  # noqa: BLE001
                continue
            for event in events:
                kind = event.get("event") or event.get("type")
                if kind in ("interrupted", "turn_aborted", "tool_error",
                            "context_overflow"):
                    signals.append({"session": session_id, "signal": kind,
                                    "detail": str(event)[:500]})
                if len(signals) >= limit:
                    return signals
            for message in session.history(last_n=20):
                content = message.content or ""
                if message.role == "tool" and content.startswith("error:"):
                    signals.append({"session": session_id, "signal": "tool_error",
                                    "detail": content[:300]})
                elif message.role == "user" and _looks_like_correction(content):
                    signals.append({"session": session_id, "signal": "user_correction",
                                    "detail": content[:300]})
                if len(signals) >= limit:
                    return signals
        for row in self.results(limit=50):
            if not row.get("pass"):
                signals.append({"session": None, "signal": "eval_failure",
                                "detail": f"{row.get('task_id')} on "
                                          f"{row.get('release_id')}"})
        for deploy in self.log.recent_deploys(20):
            if deploy.get("flagged"):
                signals.append({"session": None, "signal": "flagged_deploy",
                                "detail": f"{deploy.get('release_id')} "
                                          f"delta={deploy.get('eval_delta')}"})
        return signals[:limit]


def _looks_like_correction(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "that's wrong", "that is wrong", "incorrect", "no,", "actually,",
        "not what i", "you didn't", "you did not", "try again", "wrong answer",
        "mistake", "fix this", "not correct",
    )
    return any(marker in lowered for marker in markers)


def _parse_candidates(text: str) -> list[dict[str, Any]]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except (ValueError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return [c for c in data if isinstance(c, dict)]


_DIAGNOSIS_PROMPT = """You are analyzing outcome signals from an agent harness.
Below are sessions and runs that carry failure signals (eval failures, flagged
deploys, user corrections, aborted turns, repeated tool errors).

Write at most {n} candidate eval tasks as a single JSON array. Each task:
{{"id": "kebab-case-id", "tags": ["coding"], "prompt": "a task prompt for the agent",
"check": [{{"type": "contains", "value": "expected"}}], "scenario": "",
"failure": "the observed failure this task captures"}}

Only use check types: contains, regex, equals (checked against the response),
file_exists, file_contains (checked in the workspace; prefix the path with
`audit:` to read the eval agent's own state, e.g. `audit:logs/harness.log`).
Prefer few, sharp tasks.

Signals:
{signals}
"""
