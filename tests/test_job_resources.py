"""test_job_resources.py - cpu/memory/elapsed facts behind status() and list()."""

from __future__ import annotations

import os
from pathlib import Path

from canary.core.jobs import Jobs
from canary.core.observability import Log
from canary.core.util import pid_start_time, proc_usage
from tests.conftest import build_config


def _jobs(cfg) -> Jobs:
    return Jobs(cfg, Log(cfg))


def test_proc_usage_live_pid_reports_non_negative_values() -> None:
    pid = os.getpid()
    usage = proc_usage(pid, pid_start_time(pid))
    assert usage is not None
    assert usage["cpu_time_s"] >= 0.0
    assert usage["max_rss_kb"] is not None and usage["max_rss_kb"] >= 0


def test_proc_usage_gates_on_start_time() -> None:
    pid = os.getpid()
    assert proc_usage(pid, "not-the-start-time") is None
    assert proc_usage(-1) is None


def test_proc_usage_gone_pid_is_none() -> None:
    assert proc_usage(999999999) is None


def test_status_surfaces_live_job_resources(root: Path) -> None:
    jobs = _jobs(build_config(root))
    jid = jobs.spawn("sleep 30")["job_id"]
    try:
        st = jobs.status(jid)
        assert st["status"] == "running"
        assert st["cpu_time_s"] is not None and st["cpu_time_s"] >= 0.0
        assert st["max_rss_kb"] is not None and st["max_rss_kb"] >= 0
        assert st["elapsed_s"] is not None and st["elapsed_s"] >= 0.0
    finally:
        jobs.kill(jid)


def test_list_surfaces_live_job_resources(root: Path) -> None:
    jobs = _jobs(build_config(root))
    jid = jobs.spawn("sleep 30")["job_id"]
    try:
        row = next(r for r in jobs.list()["jobs"] if r["job_id"] == jid)
        assert row["cpu_time_s"] is not None and row["cpu_time_s"] >= 0.0
        assert row["max_rss_kb"] is not None and row["max_rss_kb"] >= 0
        assert row["elapsed_s"] is not None and row["elapsed_s"] >= 0.0
    finally:
        jobs.kill(jid)


def test_finished_job_resources_are_none(root: Path) -> None:
    jobs = _jobs(build_config(root))
    jid = jobs.spawn("true")["job_id"]
    jobs.wait(jid, timeout_s=10)
    st = jobs.status(jid)
    assert st["status"] == "done"
    assert st["cpu_time_s"] is None
    assert st["max_rss_kb"] is None
    assert st["elapsed_s"] is not None and st["elapsed_s"] >= 0.0
    row = next(r for r in jobs.list()["jobs"] if r["job_id"] == jid)
    assert row["cpu_time_s"] is None
    assert row["max_rss_kb"] is None
    assert row["elapsed_s"] is not None and row["elapsed_s"] >= 0.0
