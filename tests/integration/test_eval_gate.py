"""Integration: canary eval gate (spec §15)."""

from __future__ import annotations

from pathlib import Path

import pytest

from canary.core.evals import Evals
from canary.core.health import Health
from canary.core.util import atomic_symlink, kill_process_group, read_jsonl

pytestmark = pytest.mark.integration


def _state_api_key(state: Path) -> str:
    for line in (state / ".env").read_text().splitlines():
        if line.startswith("HARNESS_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise AssertionError("HARNESS_API_KEY missing from state env")


def test_canary_gate_measures_and_caches_baseline(initialized: Path, cfg, log) -> None:
    evals = Evals(cfg, log)
    result = evals.canary_gate("rel-candidate", "rel-baseline")
    assert result["enabled"] is True
    assert result["delta"] == 0.0
    assert result["flagged"] is False

    cache_files = list((cfg.data_path / "eval_cache").glob("*.json"))
    assert len(cache_files) == 2

    rows = list(read_jsonl(cfg.data_path / "evals.jsonl"))
    assert rows
    assert all(row.get("agent_id") for row in rows)
    assert {row.get("release_id") for row in rows} == {"rel-candidate", "rel-baseline"}


class _FakeGate:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def canary_gate(
        self, candidate, baseline, *, tasks=None, role=None, force=False,
        candidate_code_dir=None, baseline_code_dir=None,
    ):
        self.calls.append((candidate, baseline))
        return {
            "enabled": True,
            "delta": -0.5,
            "flagged": True,
            "pass_rate": 0.0,
            "baseline_pass_rate": 0.5,
            "tolerance": 0.1,
            "tasks": 2,
            "run_id": "fake-gate",
        }


def test_eval_gate_block_and_warn(
    initialized: Path, root: Path, cfg, log, monkeypatch
) -> None:
    monkeypatch.setenv("HARNESS_API_KEY", _state_api_key(root / "shared"))
    atomic_symlink(root / "current", root / "shared" / "staging")
    cfg.set("evals.canary_tasks", 3)
    health = Health(cfg, log)
    fake = _FakeGate()
    health.evals = fake  # type: ignore[assignment]

    cfg.set("evals.gate", "block")
    blocked = health.canary_release(None, label="gate-block", eval_gate=True)
    assert blocked["ok"] is False
    assert "eval gate blocked" in blocked["error"]
    assert fake.calls

    cfg.set("evals.gate", "warn")
    warned = health.canary_release(None, label="gate-warn", eval_gate=True)
    try:
        assert warned["ok"] is True, warned
        assert warned["eval"]["flagged"] is True
    finally:
        pid = warned.get("child_pid")
        if pid:
            kill_process_group(pid)
