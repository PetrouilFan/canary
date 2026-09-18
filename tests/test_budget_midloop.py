"""test_budget_midloop.py - token budget enforced at the top of the loop.

The daily cap is checked before the next model call, using a turn-entry
baseline from tokens_today() plus the turn's in-process counter. A turn that
crosses the cap is stopped, never rolled back; the check only runs when
budget.enforce is true.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from canary.core import util
from canary.core.agent import Agent
from tests.conftest import build_config

CALL = {
    "content": "",
    "tool_calls": [{"name": "bash", "arguments": {"command": "true"}}],
}


def _seed_today(cfg, tokens: int) -> None:
    """Record prior usage for today so log.tokens_today() has a baseline."""
    path = cfg.data_path / "metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": util.utc_now(), "tokens_in": tokens, "tokens_out": 0}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def _rows(cfg) -> list[dict]:
    path = cfg.data_path / "metrics.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@pytest.mark.timeout(60)
def test_over_cap_stops_before_the_next_model_call(root: Path) -> None:
    cfg = build_config(root, scripts={"main": [CALL, "done"]})
    cfg.set("budget.daily_tokens", 100)
    cfg.set("budget.enforce", True)
    _seed_today(cfg, 90)
    agent = Agent(cfg)
    try:
        text = agent.run("keep going", session_id="budget-over")
        assert text == "[turn aborted: token budget]"
        assert len(agent.models.client_for("main").backend.calls) == 1
    finally:
        agent.close()


@pytest.mark.timeout(60)
def test_under_cap_continues(root: Path) -> None:
    cfg = build_config(root, scripts={"main": [CALL, "done"]})
    cfg.set("budget.daily_tokens", 10000)
    cfg.set("budget.enforce", True)
    _seed_today(cfg, 10)
    agent = Agent(cfg)
    try:
        text = agent.run("keep going", session_id="budget-under")
        assert text == "done"
        assert len(agent.models.client_for("main").backend.calls) == 2
    finally:
        agent.close()


@pytest.mark.timeout(60)
def test_warn_row_written_when_cap_configured(root: Path) -> None:
    cfg = build_config(root, scripts={"main": ["ok"]})
    cfg.set("budget.daily_tokens", 100)
    cfg.set("budget.warn_at", 0.8)
    _seed_today(cfg, 90)
    agent = Agent(cfg)
    try:
        text = agent.run("hi", session_id="budget-warn")
        assert text == "ok"
        rows = [r for r in _rows(cfg) if r.get("kind") == "budget_warning"]
        assert rows
        assert rows[-1]["budget_used"] >= 80
        assert rows[-1]["budget_daily"] == 100
    finally:
        agent.close()


@pytest.mark.timeout(60)
def test_no_budget_rows_when_cap_is_zero(root: Path) -> None:
    cfg = build_config(root, scripts={"main": ["ok"]})
    agent = Agent(cfg)
    try:
        assert agent.run("hi", session_id="budget-off") == "ok"
        assert _rows(cfg)
        assert not [r for r in _rows(cfg) if r.get("kind") == "budget_warning"]
    finally:
        agent.close()
