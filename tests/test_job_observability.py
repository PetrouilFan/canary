"""test_job_observability.py - structured progress via the explicit report API."""

from __future__ import annotations

from pathlib import Path

from canary.core.jobs import Jobs
from canary.core.observability import Log
from tests.conftest import build_config


def _jobs(cfg) -> Jobs:
    return Jobs(cfg, Log(cfg))


def test_report_progress_updates_and_round_trips(root: Path) -> None:
    jobs = _jobs(build_config(root))
    jid = jobs.spawn("sleep 30")["job_id"]
    try:
        out = jobs.report_progress(jid, percent=40, message="warming cache")
        assert out["progress"]["percent"] == 40
        assert out["progress"]["message"] == "warming cache"
        assert out["progress"]["updated_at"]
        assert jobs.status(jid)["progress"]["percent"] == 40
        assert jobs.get(jid).progress["percent"] == 40
        out2 = jobs.report_progress(jid, percent=80)
        assert out2["progress"]["percent"] == 80
        assert out2["progress"]["message"] == "warming cache"
    finally:
        jobs.kill(jid)


def test_report_progress_rejects_unknown_and_finished(root: Path) -> None:
    jobs = _jobs(build_config(root))
    assert "error" in jobs.report_progress("nope", percent=1)
    jid = jobs.spawn("true")["job_id"]
    jobs.wait(jid, timeout_s=10)
    assert jobs.get(jid).status not in ("pending", "running")
    assert "error" in jobs.report_progress(jid, percent=1)


def test_report_progress_clamps_and_truncates(root: Path) -> None:
    jobs = _jobs(build_config(root))
    jid = jobs.spawn("sleep 30")["job_id"]
    try:
        assert jobs.report_progress(jid, percent=150)["progress"]["percent"] == 100
        assert jobs.report_progress(jid, percent=-5)["progress"]["percent"] == 0
        assert "error" in jobs.report_progress(jid, percent="high")
        msg = jobs.report_progress(jid, message="x" * 500)["progress"]["message"]
        assert len(msg) < 500 and "truncated" in msg
    finally:
        jobs.kill(jid)


def test_report_progress_rejects_bool_percent(root: Path) -> None:
    """bool is an int subclass; True is not a percentage."""
    jobs = _jobs(build_config(root))
    jid = jobs.spawn("sleep 30")["job_id"]
    try:
        assert "error" in jobs.report_progress(jid, percent=True)
        assert jobs.report_progress(jid, percent=1)["progress"]["percent"] == 1
    finally:
        jobs.kill(jid)


def test_list_surfaces_progress(root: Path) -> None:
    jobs = _jobs(build_config(root))
    jid = jobs.spawn("sleep 30")["job_id"]
    try:
        jobs.report_progress(jid, percent=10, message="m")
        entry = next(j for j in jobs.list()["jobs"] if j["job_id"] == jid)
        assert entry["progress"]["percent"] == 10
    finally:
        jobs.kill(jid)


def test_progress_survives_restart_for_persistent_job(root: Path) -> None:
    cfg = build_config(root)
    jobs = _jobs(cfg)
    jid = jobs.spawn("sleep 30", persistent=True)["job_id"]
    try:
        jobs.report_progress(jid, percent=25, message="checkpoint")
        reborn = _jobs(cfg)
        rec = reborn.get(jid)
        assert rec is not None and rec.progress["percent"] == 25
        assert rec.progress["message"] == "checkpoint"
    finally:
        jobs.kill(jid)
