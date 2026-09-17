"""Integration: codebase.lock serializes publishers across processes (spec §15)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from canary.core.util import FileLock
from tests.integration.conftest import clean_env, run_python

pytestmark = pytest.mark.integration

_BAD_PATCH = (
    "diff --git a/canary/does_not_exist_zzz.py b/canary/does_not_exist_zzz.py\n"
    "--- a/canary/does_not_exist_zzz.py\n"
    "+++ b/canary/does_not_exist_zzz.py\n"
    "@@ -1,3 +1,4 @@\n"
    " context that will not match\n"
    "+added\n"
)

_PUBLISH = """
import json
import sys

from canary.core.config import Config
from canary.core.health import Health
from canary.core.observability import Log

cfg = Config(root=sys.argv[1])
cfg.set("releases.lock_timeout_s", 1)
health = Health(cfg, Log(cfg))
health.base_gate_override = (True, "integration: lock test")
print(json.dumps(health.publish(sys.argv[2], motivation="lock test")))
"""


def test_second_publisher_waits_or_fails_cleanly(initialized: Path, root: Path) -> None:
    env = clean_env(initialized)
    lock = FileLock(root / "codebase.lock", op="test-hold", timeout=5)
    assert lock.acquire()
    try:
        proc = run_python(_PUBLISH, env, str(initialized), _BAD_PATCH)
        assert proc.returncode == 0, proc.stderr
        blocked = json.loads(proc.stdout.strip().splitlines()[-1])
        assert blocked["ok"] is False
        assert "codebase.lock held" in blocked["error"]
        assert blocked["holder"]["op"] == "test-hold"
    finally:
        lock.release()

    proc = run_python(_PUBLISH, env, str(initialized), _BAD_PATCH)
    assert proc.returncode == 0, proc.stderr
    after = json.loads(proc.stdout.strip().splitlines()[-1])
    assert after["ok"] is False
    assert "does not apply" in after["error"]
    assert after["error"] != "another copy is publishing (codebase.lock held)"
