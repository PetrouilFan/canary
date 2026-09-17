"""Integration: cross-process session injection and read receipts (spec §15)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from canary.core.session import SessionManager
from tests.integration.conftest import clean_env, run_python

pytestmark = pytest.mark.integration

_INJECT = """
import json
import sys

from canary.core.config import Config
from canary.core.observability import Log
from canary.core.session import SessionManager

cfg = Config(root=sys.argv[1])
manager = SessionManager(cfg, Log(cfg))
result = manager.inject_message(
    sys.argv[2], sys.argv[3], from_agent=sys.argv[4], from_session=sys.argv[5]
)
print(json.dumps(result))
"""


def test_cross_process_injection_and_receipt(initialized: Path, cfg, log) -> None:
    manager = SessionManager(cfg, log)
    target = manager.create(name="target", visible_to="all")
    sender = manager.create(name="sender")
    env = clean_env(initialized)

    proc = run_python(
        _INJECT, env, str(initialized), target.id, "hello from peer",
        "worker-b", sender.id,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["status"] in {"queued", "delivered"}

    reloaded = manager.load(target.id)
    assert reloaded is not None
    items = reloaded.drain_inbox()
    assert any("hello from peer" in (item.get("message") or "") for item in items)

    sender_loaded = manager.load(sender.id)
    assert sender_loaded is not None
    sender_events = sender_loaded.events()
    assert any(event.get("type") == "injection_read" for event in sender_events)


def test_injection_rate_limit_and_visibility(initialized: Path, cfg, log) -> None:
    manager = SessionManager(cfg, log)
    target = manager.create(name="limited", visible_to="all")
    sender = manager.create(name="noisy")
    env = clean_env(initialized)

    statuses = []
    for i in range(6):
        proc = run_python(
            _INJECT, env, str(initialized), target.id, f"message {i}",
            "worker-c", sender.id,
        )
        assert proc.returncode == 0, proc.stderr
        statuses.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    assert all(item["status"] in {"queued", "delivered"} for item in statuses[:5])
    assert statuses[5]["status"] == "error"
    assert "rate limit" in statuses[5]["error"]

    hidden = manager.create(name="hidden", visible_to="none")
    proc = run_python(
        _INJECT, env, str(initialized), hidden.id, "peek", "worker-d", sender.id,
    )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["status"] == "error"
