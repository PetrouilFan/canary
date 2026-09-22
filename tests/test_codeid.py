"""The process/package code identity: measured, never assumed.

``package_identity`` is the *process* identity (every ``.py`` of a package).
It is deliberately not the eval-cache key, which stays
``canary.core.evals.code_identity`` (the two files that decide scoring).
"""

from __future__ import annotations

import json
from pathlib import Path

from canary.core import util
from canary.core.codeid import package_identity
from canary.core.config import Config
from canary.core.health import Health
from canary.core.observability import Log


def test_identity_is_stable_and_tracks_python_content(tmp_path: Path) -> None:
    pkg = tmp_path / "canary"
    (pkg / "core").mkdir(parents=True)
    (pkg / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    (pkg / "core" / "a.py").write_text("y = 2\n", encoding="utf-8")

    first = package_identity(pkg)
    assert isinstance(first, str) and len(first) == 12
    assert package_identity(pkg) == first            # stable for one tree
    assert package_identity(str(pkg)) == first       # str and Path agree

    (pkg / "core" / "a.pyc").write_text("junk", encoding="utf-8")
    assert package_identity(pkg) == first            # only .py counts

    (pkg / "core" / "b.py").write_text("z = 3\n", encoding="utf-8")
    assert package_identity(pkg) != first            # a new module changes it

    second = package_identity(pkg)
    (pkg / "core" / "a.py").write_text("y = 22\n", encoding="utf-8")
    assert package_identity(pkg) not in (first, second)
    assert package_identity(tmp_path / "absent") is None


def test_health_and_log_rows_carry_the_running_identity(cfg: Config) -> None:
    log = Log(cfg)
    health = Health(cfg, log)

    assert health.code_id == package_identity()
    assert health.health_response()["code_id"] == health.code_id
    assert health.status()["code_id"] == health.code_id

    log.event("probe_event", value=1)
    rows = [json.loads(line) for line in
            (cfg.state_path / "logs" / "harness.log").read_text().splitlines()]
    assert rows[-1]["code_id"] == health.code_id
    assert rows[-1]["release_id"] == cfg.release_id


def test_deploy_row_names_both_sides_and_warns_on_a_mismatch(cfg: Config) -> None:
    log = Log(cfg)
    health = Health(cfg, log)

    release_id = "20260101T000000Z-deadbee"
    tree = health.releases_dir / release_id / "canary" / "core"
    tree.mkdir(parents=True)
    (tree / "evals.py").write_text("pass\n", encoding="utf-8")
    release_code_id = package_identity(health.releases_dir / release_id / "canary")
    assert release_code_id and release_code_id != health.code_id

    result = {"op": "publish", "ok": True, "release_id": release_id,
              "duration_ms": 7, "eval": {"delta": 0.0, "flagged": False}}
    health.record_deploy(result, motivation="unit test")

    assert result["supervisor_code_id"] == health.code_id
    assert result["release_code_id"] == release_code_id

    row = list(util.read_jsonl(cfg.data_path / "deploys.jsonl"))[-1]
    assert row["commit_sha"] == cfg.commit_sha          # back-compat: unchanged
    assert row["supervisor_code_id"] == health.code_id
    assert row["release_code_id"] == release_code_id
    assert row["release_id"] == release_id

    events = [json.loads(line) for line in
              (cfg.state_path / "logs" / "harness.log").read_text().splitlines()]
    mismatch = [e for e in events if e.get("event") == "deploy_code_mismatch"]
    assert len(mismatch) == 1
    assert mismatch[0]["level"] == "warning"
    assert mismatch[0]["supervisor_code_id"] == health.code_id
    assert mismatch[0]["release_code_id"] == release_code_id


def test_no_mismatch_warning_when_the_release_matches_the_executor(
    cfg: Config,
) -> None:
    """Same code on both sides: a row, no warning."""
    log = Log(cfg)
    health = Health(cfg, log)

    release_id = "20260101T000000Z-1234abc"
    # A release tree identical to the package this process runs.
    import canary

    src = Path(canary.__path__[0])
    dst = health.releases_dir / release_id
    dst.mkdir(parents=True)
    for path in src.rglob("*.py"):
        target = dst / "canary" / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    assert health.release_code_id(release_id) == health.code_id

    health.record_deploy({"op": "revert", "ok": True, "release_id": release_id,
                          "duration_ms": 3})
    row = list(util.read_jsonl(cfg.data_path / "deploys.jsonl"))[-1]
    assert row["supervisor_code_id"] == row["release_code_id"] == health.code_id
    log_file = cfg.state_path / "logs" / "harness.log"
    events = [json.loads(line) for line in log_file.read_text().splitlines()] \
        if log_file.exists() else []
    assert not [e for e in events if e.get("event") == "deploy_code_mismatch"]
