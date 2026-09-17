"""Integration: publish pipeline spawns a real canary copy and promotes it (spec §15)."""

from __future__ import annotations

from pathlib import Path

import pytest

from canary.core.health import Health
from canary.core.util import atomic_symlink, kill_process_group, read_jsonl
from tests.integration.conftest import ruff_only_gate

pytestmark = pytest.mark.integration

_GOOD_PATCH = (
    "diff --git a/canary/e2e_feature.py b/canary/e2e_feature.py\n"
    "new file mode 100644\n"
    "--- /dev/null\n"
    "+++ b/canary/e2e_feature.py\n"
    "@@ -0,0 +1,2 @@\n"
    "+E2E_VALUE = 42\n"
    "+\n"
)


def test_publish_runs_real_canary(
    initialized: Path, root: Path, cfg, log, monkeypatch
) -> None:
    cfg.set("evals.canary_tasks", 0)
    atomic_symlink(root / "current", root / "shared" / "staging")
    health = Health(cfg, log)
    health.base_gate_override = (True, "integration: no running copy")
    ruff_only_gate(health, monkeypatch)

    result = health.publish(
        _GOOD_PATCH, motivation="integration e2e", session_id="it-e2e"
    )
    pid = result.get("child_pid")
    try:
        assert result["ok"], result
        assert result["promoted"] is False

        current = root / "current"
        release_dir = current.resolve()
        assert release_dir.name == result["release_id"]
        assert (release_dir / "canary" / "e2e_feature.py").is_file()
        assert health.green_tags()
        assert health.staging_clean()

        deploy = list(read_jsonl(cfg.data_path / "deploys.jsonl"))[-1]
        assert deploy["ok"] is True
        assert deploy["motivation"] == "integration e2e"
        assert deploy["source_session"] == "it-e2e"
    finally:
        if pid:
            kill_process_group(pid)
