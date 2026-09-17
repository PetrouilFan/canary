"""Integration: release revert (spec §15)."""

from __future__ import annotations

from pathlib import Path

import pytest

from canary.core.health import Health
from canary.core.util import atomic_symlink
from tests.integration.conftest import fake_canary, ruff_only_gate

pytestmark = pytest.mark.integration


def _patch(path: str, line: str) -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        "@@ -0,0 +1 @@\n"
        f"+{line}\n"
    )


def test_revert_returns_to_previous_release(
    initialized: Path, root: Path, cfg, log, monkeypatch
) -> None:
    atomic_symlink(root / "current", root / "shared" / "staging")
    health = Health(cfg, log)
    health.base_gate_override = (True, "integration: no running copy")
    ruff_only_gate(health, monkeypatch)
    fake_canary(health, monkeypatch)

    first = health.publish(
        _patch("canary/feature_a.py", "VALUE_A = 1"), motivation="feature a",
        session_id="integration",
    )
    assert first["ok"], first
    release_a = first["release_id"]

    second = health.publish(
        _patch("canary/feature_b.py", "VALUE_B = 2"), motivation="feature b",
        session_id="integration",
    )
    assert second["ok"], second
    release_b = second["release_id"]
    assert (root / "current").resolve().name == release_b

    back = health.revert()
    assert back["ok"], back
    assert back["release_id"] == release_a
    assert (root / "current").resolve().name == release_a

    again = health.revert()
    assert again["ok"]
    assert again["note"] == "already serving this release"
    assert (root / "current").resolve().name == release_a

    forward = health.revert(release_b)
    assert forward["ok"], forward
    assert (root / "current").resolve().name == release_b
