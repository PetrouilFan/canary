"""test_job_kill_audit.py - kill_result: what we did to a job, and to whom.

Item (c): ``Job.kill_result = {signal, requested_at, exited_at, exit_code,
status}``. ``kill()`` opens the record with the request time and the signal
actually delivered (SIGTERM, or SIGKILL after escalation); ``_finalize()``
closes it with the exit facts. A job that exits on its own never gets one, so
"killed by us" and "failed by itself" stay distinguishable.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from canary.core.jobs import Jobs
from canary.core.observability import Log
from tests.conftest import build_config

KILL_RESULT_KEYS = {"signal", "requested_at", "exited_at", "exit_code", "status"}


def _jobs(cfg) -> Jobs:
    return Jobs(cfg, Log(cfg))


def _wait_for(predicate: Callable[[], bool], timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


def _row(jobs: Jobs, jid: str) -> dict:
    return next(j for j in jobs.list()["jobs"] if j["job_id"] == jid)


def test_kill_records_signal_and_both_timestamps(root: Path) -> None:
    jobs = _jobs(build_config(root))
    jid = jobs.spawn("sleep 30")["job_id"]
    try:
        assert _wait_for(lambda: jobs.status(jid)["alive"])
        out = jobs.kill(jid)
        kr = out["kill_result"]
        assert set(kr) == KILL_RESULT_KEYS
        assert kr["signal"] == "SIGTERM"
        assert kr["requested_at"] and kr["exited_at"]
        assert kr["requested_at"] <= kr["exited_at"]  # ISO-8601 UTC sorts lexically
        assert kr["exit_code"] is not None
        assert kr["status"] == "killed"
        # the same record is what status() and list() expose
        assert jobs.status(jid)["kill_result"] == kr
        assert _row(jobs, jid)["kill_result"] == kr
    finally:
        jobs.kill(jid)


def test_self_exit_nonzero_is_distinguishable_from_a_kill(root: Path) -> None:
    jobs = _jobs(build_config(root))
    jid = jobs.spawn("sh -c 'exit 3'")["job_id"]
    jobs.wait(jid, timeout_s=10)
    st = jobs.status(jid)
    assert st["status"] == "failed"
    assert st["exit_code"] == 3
    assert st["kill_result"] is None
    assert _row(jobs, jid)["kill_result"] is None

    killed = jobs.spawn("sleep 30")["job_id"]
    try:
        assert _wait_for(lambda: jobs.status(killed)["alive"])
        out = jobs.kill(killed)
        assert out["status"] == "killed"
        assert out["kill_result"]["status"] == "killed"
        assert out["kill_result"]["signal"] == "SIGTERM"
    finally:
        jobs.kill(killed)


def test_sigterm_ignored_escalates_to_sigkill(root: Path) -> None:
    jobs = _jobs(build_config(root))
    marker = root / "kill-audit-ready"
    # trap '' TERM is installed before the marker is written, so gating on the
    # marker guarantees the TERM-ignoring handler is in place before kill()
    # runs; otherwise the signal can land ahead of the trap (observed race).
    script = f"trap '' TERM; : > {marker}; while true; do sleep 1; done"
    jid = jobs.spawn(script)["job_id"]
    try:
        assert _wait_for(marker.exists), "child never installed its TERM trap"
        started = time.monotonic()
        out = jobs.kill(jid)
        escalated_after = time.monotonic() - started
        kr = out["kill_result"]
        assert set(kr) == KILL_RESULT_KEYS
        assert kr["signal"] == "SIGKILL"
        assert kr["status"] == "killed"
        assert kr["requested_at"] and kr["exited_at"]
        assert kr["requested_at"] <= kr["exited_at"]
        assert kr["exit_code"] == -9
        assert escalated_after >= 5.0  # the grace period ran out first
    finally:
        jobs.kill(jid)
