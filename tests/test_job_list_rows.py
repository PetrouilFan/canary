"""test_job_list_rows.py - a list() row identifies the job as well as status().

Item (d): ``list()`` rows carried neither ``cwd`` nor the full command -
``status()`` does (``Job.to_dict()`` truncates the command at 2000). A caller
scanning ``job_list`` therefore had to call ``job_status`` per row just to
learn where a job ran and what it actually ran. The row now echoes both, with
the same 2000-char bound, so the two surfaces agree.
"""

from __future__ import annotations

from pathlib import Path

from canary.core.jobs import Jobs
from canary.core.observability import Log
from tests.conftest import build_config


def _jobs(cfg) -> Jobs:
    return Jobs(cfg, Log(cfg))


def _row(jobs: Jobs, jid: str) -> dict:
    return next(j for j in jobs.list()["jobs"] if j["job_id"] == jid)


def test_list_row_carries_cwd_and_full_command(root: Path) -> None:
    cfg = build_config(root)
    jobs = _jobs(cfg)
    workdir = root / "some" / "dir"
    workdir.mkdir(parents=True)
    command = "echo " + "x" * 300  # > 200: the old row lost the tail
    jid = jobs.spawn(command, cwd=str(workdir))["job_id"]
    try:
        row = _row(jobs, jid)
        status = jobs.status(jid)
        assert row["cwd"] == str(workdir)
        assert row["command"] == command
        assert len(row["command"]) > 200
        # list() and status() describe the same job identically
        assert row["cwd"] == status["cwd"]
        assert row["command"] == status["command"]
    finally:
        jobs.kill(jid)


def test_list_row_and_status_agree_on_command_truncation(root: Path) -> None:
    cfg = build_config(root)
    jobs = _jobs(cfg)
    command = "echo " + "y" * 5000  # > 2000: both surfaces must clamp equally
    jid = jobs.spawn(command)["job_id"]
    try:
        row = _row(jobs, jid)
        status = jobs.status(jid)
        assert row["command"] == status["command"]
        assert len(row["command"]) == 2000
        assert row["command"].startswith("echo ")
    finally:
        jobs.kill(jid)
