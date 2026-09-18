"""Background job system (spec §4.9).

Jobs are fire-and-forget subprocesses. Output goes straight to a log file on
disk (never a pipe) so a job keeps running and keeps logging if the harness
dies. Kill semantics are process-group-wide. Persistent jobs are recorded to
``shared/data/jobs/{id}.json`` and are *observed* after a restart, never
resumed.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import Config
from .observability import Log
from .util import (
    atomic_write_json,
    ensure_dir,
    kill_process_group,
    mono,
    parse_stamp,
    pid_alive,
    pid_start_time,
    proc_usage,
    read_json,
    tail_lines,
    truncate,
    utc_now,
)

_running_lock = threading.Lock()


class JobLimitError(RuntimeError):
    """Raised when ``jobs.max_per_turn`` or ``jobs.max_concurrent`` is hit."""


@dataclass
class Job:
    id: str
    command: str
    shell: bool = True
    type: str = "shell"
    name: str | None = None
    status: str = "pending"
    agent_id: str = ""
    session_id: str | None = None
    exit_code: int | None = None
    pid: int | None = None
    pgid: int | None = None
    pid_start_time: str | None = None
    created: str = ""
    started: str | None = None
    ended: str | None = None
    log_path: str = ""
    persistent: bool = False
    timeout_s: int | None = None
    cwd: str | None = None
    note: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    progress: dict[str, Any] | None = None
    kill_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["command"] = truncate(self.command, 2000)
        return data


class Jobs:
    """Owns this copy's jobs. Shared state lives in ``shared/data/jobs/``."""

    def __init__(self, config: Config, log: Log) -> None:
        self.config = config
        self.logger = log
        self.dir = ensure_dir(config.data_path / "jobs")
        self.agent_id = config.get("agent.id") or ""
        self.max_per_turn = int(config.get("jobs.max_per_turn", 10) or 10)
        self.max_concurrent = int(config.get("jobs.max_concurrent", 32) or 32)
        self._jobs: dict[str, Job] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._turn_spawns = 0
        self.recover()

    # -- lifecycle ---------------------------------------------------------

    def begin_turn(self) -> None:
        """Reset the per-turn spawn counter (called by the agent loop)."""
        self._turn_spawns = 0

    def spawn(
        self,
        command: str | list[str],
        *,
        shell: bool | None = None,
        persistent: bool = False,
        session_id: str | None = None,
        type: str = "shell",
        name: str | None = None,
        cwd: str | None = None,
        timeout_s: int | None = None,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        shell_flag = shell if shell is not None else isinstance(command, str)
        with _running_lock:
            if self._turn_spawns >= self.max_per_turn:
                raise JobLimitError(
                    f"job limit reached: {self._turn_spawns} jobs spawned this turn "
                    f"(jobs.max_per_turn={self.max_per_turn})"
                )
            running = self.running()
            if len(running) >= self.max_concurrent:
                raise JobLimitError(
                    f"job limit reached: {len(running)} running "
                    f"(jobs.max_concurrent={self.max_concurrent})"
                )
            job = self._spawn(command, shell_flag, persistent, session_id, type,
                              name, cwd, timeout_s, env)
            self._turn_spawns += 1
        return {"job_id": job.id, "status": job.status, "log_path": job.log_path,
                "pid": job.pid}

    def _spawn(
        self,
        command: str | list[str],
        shell: bool,
        persistent: bool,
        session_id: str | None,
        type: str,
        name: str | None,
        cwd: str | None,
        timeout_s: int | None,
        env: dict[str, str] | None,
    ) -> Job:
        job_id = uuid.uuid4().hex
        log_path = self.dir / f"{job_id}.log"
        child_env = os.environ.copy()
        if self.config.root is not None:
            child_env["CANARY_ROOT"] = str(self.config.root)
        child_env["HARNESS_STATE_PATH"] = str(self.config.state_path)
        child_env["HARNESS_RELEASE_ID"] = self.config.release_id
        child_env["HARNESS_COMMIT_SHA"] = self.config.commit_sha
        child_env["CANARY_JOB_ID"] = job_id  # so the job can self-report progress
        if env:
            child_env.update({k: str(v) for k, v in env.items()})
        workdir = cwd or str(self.config.base_dir)
        job = Job(
            id=job_id,
            command=command if isinstance(command, str) else " ".join(map(str, command)),
            shell=shell,
            type=type,
            name=name,
            agent_id=self.agent_id,
            session_id=session_id,
            created=utc_now(),
            log_path=str(log_path),
            persistent=persistent,
            timeout_s=int(timeout_s) if timeout_s else None,
            cwd=workdir,
        )
        with open(log_path, "ab") as fh:
            proc = subprocess.Popen(
                command,
                shell=shell,
                cwd=workdir,
                stdin=subprocess.DEVNULL,
                stdout=fh,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
                env=child_env,
            )
        job.status = "running"
        job.pid = proc.pid
        job.started = utc_now()
        job.pid_start_time = pid_start_time(proc.pid)
        try:
            job.pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            job.pgid = proc.pid
        self._jobs[job.id] = job
        self._procs[job.id] = proc
        if persistent:
            self._persist(job)
        threading.Thread(target=self._monitor, args=(job, proc), daemon=True).start()
        self.logger.event(
            "job_spawned",
            job_id=job.id,
            type=job.type,
            name=job.name,
            pid=job.pid,
            persistent=persistent,
            command=truncate(job.command, 400),
        )
        return job

    def _monitor(self, job: Job, proc: subprocess.Popen) -> None:
        timed_out = False
        try:
            rc = proc.wait(timeout=job.timeout_s if job.timeout_s else None)
        except subprocess.TimeoutExpired:
            timed_out = True
            job.note = "timeout"
            kill_process_group(job.pgid or job.pid or 0)
            try:
                rc = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                rc = -9
        time.sleep(0.05)
        self._finalize(job, rc, timed_out=timed_out)

    def _finalize(self, job: Job, rc: int | None, timed_out: bool = False) -> None:
        with _running_lock:
            if job.status not in ("pending", "running"):
                return
            if timed_out or job.note == "killed":
                job.status = "killed"
            elif rc == 0:
                job.status = "done"
            else:
                job.status = "failed"
            job.exit_code = rc
            job.ended = utc_now()
            if job.kill_result is not None:
                job.kill_result["exited_at"] = job.ended
                job.kill_result["exit_code"] = rc
                job.kill_result["status"] = job.status
            self._procs.pop(job.id, None)
            if job.persistent:
                self._persist(job)
        self.logger.event(
            "job_finished",
            job_id=job.id,
            type=job.type,
            status=job.status,
            exit_code=rc,
            note=job.note,
        )

    # -- persistence & recovery -------------------------------------------

    def _persist(self, job: Job) -> None:
        atomic_write_json(self.dir / f"{job.id}.json", job.to_dict())

    def recover(self) -> None:
        """Load persistent records; observe, never resume."""
        for path in sorted(self.dir.glob("*.json")):
            record = read_json(path, default=None)
            if not isinstance(record, dict) or not record.get("id"):
                continue
            if record.get("agent_id") != self.agent_id:
                continue
            job = Job(**{k: v for k, v in record.items() if k in Job.__dataclass_fields__})
            if job.status in ("pending", "running"):
                if pid_alive(job.pid or 0, job.pid_start_time):
                    job.status = "running"
                    threading.Thread(target=self._watch_recovered, args=(job,),
                                     daemon=True).start()
                else:
                    job.status = "failed"
                    job.note = "process gone; observed after restart"
                    job.ended = job.ended or utc_now()
                    job.exit_code = None
                    self._persist(job)
            self._jobs[job.id] = job

    def _watch_recovered(self, job: Job) -> None:
        while pid_alive(job.pid or 0, job.pid_start_time):
            time.sleep(2.0)
        self._finalize(job, None)

    # -- queries -----------------------------------------------------------

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def _load(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        if job is not None:
            return job
        path = self.dir / f"{job_id}.json"
        record = read_json(path, default=None)
        if isinstance(record, dict) and record.get("id"):
            job = Job(**{k: v for k, v in record.items()
                         if k in Job.__dataclass_fields__})
            self._jobs[job.id] = job
        return job

    def running(self, agent_id: str | None = None) -> list[Job]:
        out = []
        for job in self._jobs.values():
            if job.status not in ("pending", "running"):
                continue
            if agent_id and job.agent_id != agent_id:
                continue
            if job.id not in self._procs and not pid_alive(job.pid or 0,
                                                           job.pid_start_time):
                self._finalize(job, None)
                continue
            out.append(job)
        return out

    def _runtime_facts(self, job: Job) -> dict[str, Any]:
        """Wall-clock elapsed and, while the pid lives, its cpu/memory usage.

        Every field is None when it cannot be measured: usage is read from
        ``/proc`` and gated on the pid start time, so a finished job (or a
        recycled pid) reports None instead of another process's numbers.
        """
        usage = proc_usage(job.pid or 0, job.pid_start_time) if job.pid else None
        return {
            "cpu_time_s": usage["cpu_time_s"] if usage else None,
            "max_rss_kb": usage["max_rss_kb"] if usage else None,
            "elapsed_s": self._elapsed_s(job),
        }

    @staticmethod
    def _elapsed_s(job: Job) -> float | None:
        if not job.started:
            return None
        try:
            start = parse_stamp(job.started)
            end = parse_stamp(job.ended) if job.ended else time.time()
        except ValueError:
            return None
        return max(0.0, round(end - start, 3))

    def status(self, job_id: str) -> dict[str, Any]:
        job = self._load(job_id)
        if job is None:
            return {"error": f"unknown job: {job_id}"}
        data = job.to_dict()
        data["alive"] = job.status in ("pending", "running") and (
            job.id in self._procs or pid_alive(job.pid or 0, job.pid_start_time)
        )
        data.update(self._runtime_facts(job))
        progress, source = self._progress_view(job)
        data["progress"] = progress
        data["progress_source"] = source
        data["log_tail"] = tail_lines(job.log_path, 10)
        return data

    def tail(self, job_id: str, n: int = 50) -> dict[str, Any]:
        job = self._load(job_id)
        if job is None:
            return {"error": f"unknown job: {job_id}"}
        return {
            "job_id": job.id,
            "status": job.status,
            "exit_code": job.exit_code,
            "lines": tail_lines(job.log_path, max(1, int(n))),
        }

    def log(
        self,
        job_id: str,
        offset: int | None = None,
        lines: int | None = None,
    ) -> dict[str, Any]:
        job = self._load(job_id)
        if job is None:
            return {"error": f"unknown job: {job_id}"}
        start = max(0, int(offset or 0))
        limit = max(1, int(lines)) if lines else 100
        selected: list[str] = []
        total = 0
        path = Path(job.log_path)
        if path.exists():
            with open(path, encoding="utf-8", errors="replace") as fh:
                for idx, line in enumerate(fh):
                    if idx < start:
                        continue
                    if len(selected) < limit:
                        selected.append(line.rstrip("\n"))
                    total = idx + 1
        else:
            total = 0
        next_offset = start + len(selected)
        return {
            "job_id": job.id,
            "status": job.status,
            "log_path": job.log_path,
            "offset": start,
            "next_offset": next_offset,
            "total_lines": total,
            "eof": next_offset >= total,
            "lines": selected,
        }

    def list(self, status: str | None = None, agent_id: str | None = None) -> dict[str, Any]:
        jobs = sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)
        out = []
        for job in jobs:
            if status and job.status != status:
                continue
            if agent_id and job.agent_id != agent_id:
                continue
            progress, progress_source = self._progress_view(job)
            out.append({
                "job_id": job.id,
                "name": job.name,
                "type": job.type,
                "status": job.status,
                "agent_id": job.agent_id,
                "session_id": job.session_id,
                "pid": job.pid,
                **self._runtime_facts(job),
                "created": job.created,
                "ended": job.ended,
                "exit_code": job.exit_code,
                "persistent": job.persistent,
                "cwd": job.cwd,
                "command": truncate(job.command, 2000),
                "progress": progress,
                "progress_source": progress_source,
                "kill_result": job.kill_result,
            })
        return {"jobs": out, "count": len(out)}

    # -- progress ----------------------------------------------------------

    def report_progress(
        self,
        job_id: str,
        *,
        percent: float | None = None,
        message: str | None = None,
    ) -> dict[str, Any]:
        """Record structured progress for a live job (spec 4.9).

        Reached through this API - progress is never inferred from job output.
        Partial updates merge: an omitted field keeps its previous value.
        """
        job = self._load(job_id)
        if job is None:
            return {"error": f"unknown job: {job_id}"}
        if job.status not in ("pending", "running"):
            return {"error": f"job {job_id} is {job.status}; not running"}
        if percent is not None and (
            isinstance(percent, bool) or not isinstance(percent, (int, float))
        ):
            return {"error": "percent must be a number in [0, 100]"}
        prev = job.progress or {}
        pct = prev.get("percent")
        if percent is not None:
            pct = max(0.0, min(100.0, float(percent)))
        msg = prev.get("message")
        if message is not None:
            msg = truncate(str(message), 200)
        job.progress = {"percent": pct, "message": msg, "updated_at": utc_now()}
        if job.persistent:
            self._persist(job)
        self.logger.event(
            "job_progress", job_id=job.id, percent=pct, message=truncate(str(msg or ""), 80)
        )
        return {"job_id": job.id, "status": job.status, "progress": job.progress}

    def progress_path(self, job_id: str) -> Path:
        """The file a job's own process writes to self-report progress (4.9).

        Transport for subprocesses: no HTTP endpoint, so a job drops
        ``{jobs_dir}/{job_id}.progress.json`` written atomically with
        :func:`canary.core.util.atomic_write_json`. Only :class:`Jobs` reads it.
        """
        return self.dir / f"{job_id}.progress.json"

    def _file_progress(self, job: Job) -> dict[str, Any] | None:
        """The job's self-reported progress file, when it can be trusted.

        Start-time gating, as for every other pid check here: the file counts
        only while the job is live under its recorded pid *and* start time, and
        only if it was written after this incarnation started
        (``mtime >= job.started``). A file left behind by an earlier job - or by
        a job id that came back around - therefore never wins.
        """
        if job.status not in ("pending", "running"):
            return None
        if not (job.id in self._procs or pid_alive(job.pid or 0, job.pid_start_time)):
            return None
        path = self.progress_path(job.id)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        if job.started:
            try:
                if mtime < parse_stamp(job.started):
                    return None
            except ValueError:
                return None
        record = read_json(path, default=None)
        if not isinstance(record, dict) or str(record.get("job_id") or "") != job.id:
            return None
        percent = record.get("percent")
        if percent is not None and (
            isinstance(percent, bool) or not isinstance(percent, (int, float))
        ):
            return None
        message = record.get("message")
        if percent is None and message is None:
            return None
        return {
            "percent": None if percent is None else max(0.0, min(100.0, float(percent))),
            "message": None if message is None else truncate(str(message), 200),
            "updated_at": record.get("updated_at"),
        }

    def _progress_view(self, job: Job) -> tuple[dict[str, Any] | None, str | None]:
        """Progress plus the channel it came from: ``"file"`` or ``"memory"``.

        The file is the job's own word and wins while it is valid, unless the
        in-memory record written through :meth:`report_progress` is newer - a
        fresh API report must not be masked by a stale file.
        """
        reported = self._file_progress(job)
        if reported is None:
            return (job.progress, "memory") if job.progress else (None, None)
        if job.progress:
            mine = str(job.progress.get("updated_at") or "")
            theirs = str(reported.get("updated_at") or "")
            if mine and mine > theirs:
                return job.progress, "memory"
        return reported, "file"

    # -- control -----------------------------------------------------------

    def kill(self, job_id: str) -> dict[str, Any]:
        job = self._load(job_id)
        if job is None:
            return {"error": f"unknown job: {job_id}"}
        if job.status not in ("pending", "running"):
            return {"error": f"job {job_id} is {job.status}; nothing to kill"}
        job.note = "killed"
        job.kill_result = {
            "signal": None,
            "requested_at": utc_now(),
            "exited_at": None,
            "exit_code": None,
            "status": "requested",
        }
        sent: list[str] = []
        proc = self._procs.get(job.id)
        if proc is not None:
            kill_process_group(job.pgid or job.pid or 0, sent=sent)
            try:
                rc = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                rc = -9
            self._finalize(job, rc)
        else:
            alive = pid_alive(job.pid or 0, job.pid_start_time)
            if alive:
                kill_process_group(job.pgid or job.pid or 0, sent=sent)
            self._finalize(job, -15)
        job.kill_result["signal"] = sent[-1] if sent else None
        self.logger.event(
            "job_killed",
            job_id=job.id,
            status=job.status,
            signal=job.kill_result["signal"],
        )
        return {
            "job_id": job.id,
            "status": job.status,
            "exit_code": job.exit_code,
            "kill_result": job.kill_result,
        }

    def info(self) -> dict[str, Any]:
        return {
            "running": len(self.running()),
            "known": len(self._jobs),
            "turn_spawns": self._turn_spawns,
            "max_per_turn": self.max_per_turn,
            "max_concurrent": self.max_concurrent,
        }

    def wait(self, job_id: str, timeout_s: float | None = None) -> dict[str, Any]:
        """Test/CLI helper: block until a job leaves running (monitor does it)."""
        deadline = mono() + timeout_s if timeout_s else None
        while True:
            job = self._load(job_id)
            if job is None:
                return {"error": f"unknown job: {job_id}"}
            if job.status not in ("pending", "running"):
                return job.to_dict()
            if deadline is not None and mono() > deadline:
                return job.to_dict()
            time.sleep(0.1)
