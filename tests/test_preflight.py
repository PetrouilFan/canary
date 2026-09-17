"""test_preflight.py - classify, publish red/green, crash-resume idempotency."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from canary.cli import main as cli_main
from canary.core.config import Config
from canary.core.health import Health, classify
from canary.core.observability import Log
from canary.core.util import FileLock, atomic_symlink, atomic_write_json

GOOD_PATCH = """diff --git a/canary/extra_module.py b/canary/extra_module.py
new file mode 100644
--- /dev/null
+++ b/canary/extra_module.py
@@ -0,0 +1,2 @@
+VALUE = 1
+DELTA = 2
"""

BAD_PATCH = """diff --git a/canary/does_not_exist_xyz.py b/canary/does_not_exist_xyz.py
--- a/canary/does_not_exist_xyz.py
+++ b/canary/does_not_exist_xyz.py
@@ -1 +1 @@
-OLD
+NEW
"""


def test_classify_paths() -> None:
    assert classify(["canary/core/agent.py"]) == "code"
    assert classify(["shared/extensions/foo.py"]) == "extension"
    assert classify(["shared/governance.yaml"]) == "config"
    assert classify(["shared/SOUL.md"]) == "config"
    assert classify(["shared/evals/task.yaml"]) == "config"
    assert classify(["shared/memory/index.md"]) == "memory"
    assert classify(["shared/workspace/notes.txt"]) == "memory"
    assert classify(["shared/data/x.json"]) == "memory"


def _init_root(root: Path, capsys: pytest.CaptureFixture) -> None:
    rc = cli_main(["init", "--root", str(root), "--no-embedding"])
    capsys.readouterr()
    assert rc == 0


def _health(root: Path) -> Health:
    cfg = Config(root=root)
    handler = Health(cfg, Log(cfg))
    handler.base_gate_override = (True, "test: no running copy")
    return handler


def _patch_gates(health: Health, monkeypatch: pytest.MonkeyPatch) -> None:
    def ruff_only() -> str:
        proc = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "."],
            cwd=health.staging,
            capture_output=True,
            text=True,
        )
        return "" if proc.returncode == 0 else proc.stdout + proc.stderr

    monkeypatch.setattr(health, "_run_static_gate", ruff_only)


def _fake_canary(health: Health, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def fake(release_dir, *, label=None, eval_gate=True):
        calls.append(str(label))
        atomic_symlink(health.current_link, release_dir)
        return {
            "ok": True,
            "release_id": label,
            "port": None,
            "child_pid": None,
            "promoted": True,
            "eval": {"enabled": False, "delta": None, "flagged": False},
        }

    monkeypatch.setattr(health, "canary_release", fake)
    return calls


@pytest.mark.timeout(120)
def test_bad_patch_rejected(root: Path, capsys) -> None:
    _init_root(root, capsys)
    health = _health(root)
    result = health.publish(BAD_PATCH, motivation="bad change")
    assert result["ok"] is False
    assert "does not apply" in result["error"]
    assert health.staging_clean()
    assert health.green_tags() == []
    assert not health.publish_state_path.exists()
    deploys = health.log.recent_deploys(1)
    assert deploys and deploys[-1]["ok"] is False


@pytest.mark.timeout(180)
def test_good_change_published(
    root: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    _init_root(root, capsys)
    health = _health(root)
    _patch_gates(health, monkeypatch)
    calls = _fake_canary(health, monkeypatch)

    result = health.publish(GOOD_PATCH, motivation="add module", session_id="s1")
    assert result["ok"] is True, result
    assert result["release_id"] in calls
    assert result["tag"] in health.green_tags()
    assert health.staging_clean()
    assert not health.publish_state_path.exists()

    release_dir = health.releases_dir / result["release_id"]
    assert (release_dir / "canary" / "extra_module.py").is_file()
    assert health.current_release() == release_dir

    patches = list((health.config.data_path / "patches" / "proposed").glob("*.diff"))
    assert len(patches) == 1

    deploy = health.log.recent_deploys(1)[-1]
    assert deploy["ok"] is True
    assert deploy["motivation"] == "add module"
    assert deploy["source_session"] == "s1"


@pytest.mark.timeout(120)
def test_crash_resume_idempotent(root: Path, capsys) -> None:
    _init_root(root, capsys)
    health = _health(root)
    atomic_write_json(
        health.publish_state_path,
        {"op": "publish", "step": "static", "started": "test"},
    )
    dirty = health.staging / "canary" / "core" / "util.py"
    dirty.write_text(
        dirty.read_text(encoding="utf-8") + "\n# dirty change\n", encoding="utf-8"
    )

    first = health.publish(BAD_PATCH, motivation="resume test")
    assert first["ok"] is False
    assert health.staging_clean()
    assert not health.publish_state_path.exists()
    assert health.green_tags() == []

    second = health.publish(BAD_PATCH, motivation="resume test")
    assert second["ok"] is False
    assert health.staging_clean()
    assert not health.publish_state_path.exists()

    log_text = (health.config.state_path / "logs" / "harness.log").read_text(
        encoding="utf-8"
    )
    assert "publish_state_recovered" in log_text


@pytest.mark.timeout(60)
def test_unlock_refuses_wrong_nonce(root: Path, capsys) -> None:
    _init_root(root, capsys)
    health = _health(root)
    lock_path = health.root / "codebase.lock"
    lock = FileLock(lock_path, op="test-unlock", timeout=5)
    assert lock.acquire()
    try:
        result = health.unlock("not-the-nonce")
        assert result["ok"] is False
        assert result["meta"]["nonce"] == lock.nonce
        assert lock_path.exists()
    finally:
        lock.release()
