"""test_job_progress_file.py - the file-drop self-report channel (spec 4.9).

A job's own process has no HTTP back door into the harness, so it drops
``{jobs_dir}/{job_id}.progress.json`` (atomically, via
``util.atomic_write_json``). ``status()``/``list()`` merge that file while the
job is live and say which channel won: ``progress_source`` is ``"file"`` or
``"memory"``.

The read is start-time gated like every other pid check: the file counts only
while the job is live under its recorded pid+start time, and only if it was
written after this incarnation started. A stale file can never win.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

from canary.core import util
from canary.core.jobs import Jobs
from canary.core.observability import Log
from tests.conftest import build_config


def _jobs(cfg) -> Jobs:
    return Jobs(cfg, Log(cfg))


def _row(jobs: Jobs, jid: str) -> dict:
    return next(j for j in jobs.list()["jobs"] if j["job_id"] == jid)


def _wait_for(predicate: Callable[[], bool], timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


def _drop(jobs: Jobs, jid: str, **fields) -> Path:
    path = jobs.progress_path(jid)
    util.atomic_write_json(path, {"job_id": jid, **fields})
    return path


def test_live_job_file_is_merged_and_labelled(root: Path) -> None:
    jobs = _jobs(build_config(root))
    jid = jobs.spawn("sleep 30")["job_id"]
    try:
        assert _wait_for(lambda: jobs.status(jid)["alive"])
        _drop(jobs, jid, percent=42, message="half way", updated_at=util.utc_now())
        status = jobs.status(jid)
        assert status["progress"]["percent"] == 42
        assert status["progress"]["message"] == "half way"
        assert status["progress_source"] == "file"
        row = _row(jobs, jid)
        assert row["progress"] == status["progress"]
        assert row["progress_source"] == "file"
        # clamped like the API channel
        _drop(jobs, jid, percent=140, message="x" * 300)
        status = jobs.status(jid)
        assert status["progress"]["percent"] == 100
        assert len(status["progress"]["message"]) == 200
    finally:
        jobs.kill(jid)


def test_file_for_another_job_id_is_ignored(root: Path) -> None:
    jobs = _jobs(build_config(root))
    jid = jobs.spawn("sleep 30")["job_id"]
    try:
        assert _wait_for(lambda: jobs.status(jid)["alive"])
        util.atomic_write_json(  # right file name, wrong identity inside
            jobs.progress_path(jid),
            {"job_id": "not-this-job", "percent": 90},
        )
        status = jobs.status(jid)
        assert status["progress"] is None
        assert status["progress_source"] is None
    finally:
        jobs.kill(jid)


def test_finished_job_file_is_ignored(root: Path) -> None:
    jobs = _jobs(build_config(root))
    jid = jobs.spawn("true")["job_id"]
    assert _wait_for(lambda: jobs.status(jid)["status"] == "done")
    _drop(jobs, jid, percent=99, message="too late", updated_at=util.utc_now())
    status = jobs.status(jid)
    assert status["progress"] is None
    assert status["progress_source"] is None
    assert _row(jobs, jid)["progress_source"] is None


def test_file_older_than_this_incarnation_is_ignored(root: Path) -> None:
    jobs = _jobs(build_config(root))
    jid = jobs.spawn("sleep 30")["job_id"]
    try:
        assert _wait_for(lambda: jobs.status(jid)["alive"])
        job = jobs.get(jid)
        path = _drop(jobs, jid, percent=77, message="stale")
        stale = util.parse_stamp(job.started) - 60.0  # written before we started
        os.utime(path, (stale, stale))
        assert jobs.status(jid)["progress"] is None
        os.utime(path, None)  # fresh again -> merged
        assert jobs.status(jid)["progress_source"] == "file"
    finally:
        jobs.kill(jid)


def test_recovered_job_needs_matching_start_time(root: Path) -> None:
    """After a restart there is no live handle, so pid_start_time decides."""
    jobs = _jobs(build_config(root))
    jid = jobs.spawn("sleep 30")["job_id"]
    try:
        assert _wait_for(lambda: jobs.status(jid)["alive"])
        _drop(jobs, jid, percent=55, message="from the job")
        job = jobs.get(jid)
        jobs._procs.pop(jid, None)  # simulate a recovered job: no handle
        real = job.pid_start_time
        job.pid_start_time = f"999{real}"  # a recycled id, different process
        assert jobs.status(jid)["progress"] is None
        job.pid_start_time = real
        assert jobs.status(jid)["progress"]["percent"] == 55
    finally:
        jobs._procs.pop(jid, None)
        jobs.kill(jid)


def test_memory_channel_still_labelled_and_newer_report_wins(root: Path) -> None:
    jobs = _jobs(build_config(root))
    jid = jobs.spawn("sleep 30")["job_id"]
    try:
        assert _wait_for(lambda: jobs.status(jid)["alive"])
        jobs.report_progress(jid, percent=10, message="via API")
        status = jobs.status(jid)
        assert status["progress_source"] == "memory"
        assert status["progress"]["percent"] == 10
        # a newer API report is not masked by an older file
        _drop(jobs, jid, percent=20, updated_at="2000-01-01T00:00:00Z")
        assert jobs.status(jid)["progress_source"] == "memory"
        _drop(jobs, jid, percent=30, updated_at=util.utc_now())
        status = jobs.status(jid)
        assert status["progress_source"] == "file"
        assert status["progress"]["percent"] == 30
    finally:
        jobs.kill(jid)
