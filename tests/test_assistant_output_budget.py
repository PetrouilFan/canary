"""test_assistant_output_budget.py - per-turn assistant output cap.

A turn whose own assistant text exceeds ``max_assistant_chars_per_turn`` cannot
be recovered by compression: those messages sit in the recent window, which is
kept verbatim, so ``_forced_compress`` can only loop until the turn timeout.
The loop therefore stops at the top, before the next model call, sets
``limit_hit = "assistant output budget"``, appends a [system] notice and logs a
row - without rolling back what the turn already produced.

Tool results have their own budget (``tools.turn_output_budget``) and are not
counted here; ``0`` disables the cap.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from canary.core.agent import Agent
from tests.conftest import build_config

CALL = {
    "content": "",
    "tool_calls": [{"name": "bash", "arguments": {"command": "true"}}],
}
BIG = "x" * 500
BIG_CALL = {
    "content": BIG,
    "tool_calls": [{"name": "bash", "arguments": {"command": "true"}}],
}
MID = "y" * 50
MID_CALL = {
    "content": MID,
    "tool_calls": [{"name": "bash", "arguments": {"command": "true"}}],
}


def _metrics(cfg) -> list[dict]:
    path = cfg.data_path / "metrics.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _events(cfg) -> list[dict]:
    path = Path(cfg.state_path) / "logs" / "harness.log"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _stop_rows(cfg) -> list[dict]:
    return [r for r in _events(cfg) if r.get("event") == "assistant_output_stop"]


@pytest.mark.timeout(60)
def test_under_cap_continues_to_completion(root: Path) -> None:
    cfg = build_config(root, scripts={"main": [MID_CALL, "done"]})
    cfg.set("max_assistant_chars_per_turn", 1000)
    agent = Agent(cfg)
    try:
        assert agent.run("go", session_id="assistant-under") == f"{MID}\ndone"
        assert len(agent.models.client_for("main").backend.calls) == 2
        assert agent.last_result["status"] == "idle"
        assert _stop_rows(cfg) == []
    finally:
        agent.close()


@pytest.mark.timeout(60)
def test_over_cap_stops_before_the_next_model_call(root: Path) -> None:
    cfg = build_config(root, scripts={"main": [BIG_CALL, "done"]})
    cfg.set("max_assistant_chars_per_turn", 100)
    agent = Agent(cfg)
    try:
        text = agent.run("go", session_id="assistant-over")
        # no second model call, no rollback: the text the turn produced is kept
        assert len(agent.models.client_for("main").backend.calls) == 1
        assert agent.last_result["status"] == "aborted: assistant output budget"
        assert text == BIG
        session = agent.sessions.load("assistant-over")
        history = [m.content for m in session.history()] if session else []
        assert BIG in history
        rows = _stop_rows(cfg)
        assert len(rows) == 1
        assert rows[0]["chars"] == len(BIG)
        assert rows[0]["limit"] == 100
        assert rows[0]["model_calls"] == 1
        assert [
            r for r in _metrics(cfg)
            if r.get("status") == "aborted: assistant output budget"
        ]
    finally:
        agent.close()


@pytest.mark.timeout(60)
def test_zero_disables_the_cap(root: Path) -> None:
    cfg = build_config(root, scripts={"main": [BIG_CALL, "done"]})
    cfg.set("max_assistant_chars_per_turn", 0)
    agent = Agent(cfg)
    try:
        assert agent.run("go", session_id="assistant-off") == f"{BIG}\ndone"
        assert len(agent.models.client_for("main").backend.calls) == 2
        assert agent.last_result["status"] == "idle"
        assert _stop_rows(cfg) == []
        assert not [
            r for r in _metrics(cfg)
            if r.get("status") == "aborted: assistant output budget"
        ]
    finally:
        agent.close()


@pytest.mark.timeout(60)
def test_large_tool_output_is_not_counted(root: Path) -> None:
    cfg = build_config(root, scripts={"main": [CALL, "done"]})
    cfg.set("max_assistant_chars_per_turn", 200)
    cfg.set("tools.turn_output_budget", 200000)  # no spill: keep the raw result
    agent = Agent(cfg)
    try:
        agent.tools.call = (  # type: ignore[method-assign]
            lambda name, args: "y" * 60000
        )
        assert agent.run("go", session_id="assistant-tools") == "done"
        backend = agent.models.client_for("main").backend
        assert len(backend.calls) == 2
        tool_payload = [
            m for m in backend.calls[1] if m.get("role") == "tool"
        ]
        assert tool_payload and len(tool_payload[0]["content"]) == 60000
        assert agent.last_result["status"] == "idle"
        assert _stop_rows(cfg) == []
    finally:
        agent.close()
