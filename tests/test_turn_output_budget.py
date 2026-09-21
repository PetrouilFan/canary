"""test_turn_output_budget.py - per-turn tool-output budget (tools.*)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from canary.core.agent import Agent
from canary.core.context import ContextEngine
from canary.core.session import SessionManager
from tests.conftest import build_config

BUDGET = 24576
MARK = "(read it if needed) ...]"


def _lines(tag: str, chars: int) -> str:
    out: list[str] = []
    size = 0
    i = 0
    while size < chars:
        line = f"{tag} line {i:04d}"
        out.append(line)
        size += len(line) + 1
        i += 1
    return "\n".join(out)


def _tool_msg(call_id: str, content: str, name: str = "bash") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}


def _engine(cfg, log) -> ContextEngine:
    return ContextEngine(cfg, log)


def _payload(messages: list[dict]) -> str:
    return json.dumps(messages, sort_keys=True, default=str)


def _run_scripted(root: Path, budget: int, session_id: str = "budget") -> dict:
    """One turn: scripted tool call, then a final text answer."""
    cfg = build_config(
        root,
        scripts={
            "main": [
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "bash",
                            "arguments": {"command": "true"},
                        }
                    ],
                },
                "done",
            ]
        },
    )
    cfg.set("tools.turn_output_budget", budget)
    agent = Agent(cfg)
    try:
        agent.tools.call = (  # type: ignore[method-assign]
            lambda name, args: "small deterministic output"
        )
        text = agent.run("go", session_id=session_id)
        backend = agent.models.client_for("main").backend
        calls = json.loads(json.dumps(backend.calls, default=str))
        session = agent.sessions.load(session_id)
        history = [m.content for m in session.history()] if session else []
        spill_dir = cfg.data_path / "tmp"
        out_files = (
            sorted(p.name for p in spill_dir.glob("*.out")) if spill_dir.exists() else []
        )
        return {
            "text": text,
            "calls": calls,
            "history": history,
            "out_files": out_files,
        }
    finally:
        agent.close()


@pytest.mark.timeout(60)
def test_default_budget_is_24576(cfg) -> None:
    assert cfg.get("tools.turn_output_budget") == BUDGET


@pytest.mark.timeout(60)
def test_short_turn_payload_byte_identical(root: Path) -> None:
    on = _run_scripted(root, BUDGET, "budget-on")
    off = _run_scripted(root, 0, "budget-off")
    assert on["text"] == off["text"] == "done"
    # Same script, same tool result: the provider saw byte-identical payloads.
    assert on["calls"] == off["calls"]
    assert on["out_files"] == [] and off["out_files"] == []
    assert MARK not in json.dumps(on["calls"])
    assert [m["content"] for m in on["calls"][-1] if m["role"] == "tool"] == [
        "small deterministic output"
    ]
    assert [c for c in on["history"] if c == "small deterministic output"]


@pytest.mark.timeout(60)
def test_crossing_budget_spills_largest_first(root: Path, cfg, log) -> None:
    cfg.set("tools.turn_output_budget", 12000)
    ctx = _engine(cfg, log)
    session = SessionManager(cfg, log).create(name="budget-spill")
    big = _lines("A", 10000)
    mid = _lines("B", 8000)
    small = _lines("C", 6000)
    messages = [
        _tool_msg("call_a", big),
        _tool_msg("call_b", mid),
        _tool_msg("call_c", small),
    ]
    spilled = ctx.enforce_turn_output_budget(messages, session=session)
    assert len(spilled) == 2, "the two largest outputs should spill, not the small one"
    assert ctx.tool_chars(messages) <= 12000
    assert Path(spilled[0]).read_text() == big
    assert Path(spilled[1]).read_text() == mid
    assert MARK in messages[0]["content"] and "A line 0000" in messages[0]["content"]
    assert MARK in messages[1]["content"]
    assert messages[2]["content"] == small


@pytest.mark.timeout(60)
def test_second_pass_is_idempotent(root: Path, cfg, log) -> None:
    cfg.set("tools.turn_output_budget", 12000)
    ctx = _engine(cfg, log)
    session = SessionManager(cfg, log).create(name="budget-idem")
    messages = [
        _tool_msg("call_a", _lines("A", 10000)),
        _tool_msg("call_b", _lines("B", 8000)),
        _tool_msg("call_c", _lines("C", 6000)),
    ]
    assert ctx.enforce_turn_output_budget(messages, session=session)
    after = _payload(messages)
    assert ctx.enforce_turn_output_budget(messages, session=session) == []
    assert _payload(messages) == after


@pytest.mark.timeout(60)
def test_zero_budget_disables(root: Path, cfg, log) -> None:
    cfg.set("tools.turn_output_budget", 0)
    ctx = _engine(cfg, log)
    session = SessionManager(cfg, log).create(name="budget-off")
    messages = [_tool_msg("call_a", _lines("A", 40000))]
    before = _payload(messages)
    assert ctx.enforce_turn_output_budget(messages, session=session) == []
    assert _payload(messages) == before
    assert not list(ctx.spill_dir.glob("*.out"))


@pytest.mark.timeout(60)
def test_under_budget_messages_untouched(root: Path, cfg, log) -> None:
    ctx = _engine(cfg, log)
    session = SessionManager(cfg, log).create(name="budget-under")
    messages = [_tool_msg("call_a", "tiny output")]
    before = _payload(messages)
    assert ctx.enforce_turn_output_budget(messages, session=session) == []
    assert _payload(messages) == before


@pytest.mark.timeout(60)
def test_single_huge_line_is_left_alone(root: Path, cfg, log) -> None:
    """A pointer for one giant line is not smaller than the original."""
    cfg.set("tools.turn_output_budget", 1024)
    ctx = _engine(cfg, log)
    session = SessionManager(cfg, log).create(name="budget-oneline")
    content = "z" * 5000
    messages = [_tool_msg("call_a", content)]
    assert ctx.enforce_turn_output_budget(messages, session=session) == []
    assert messages[0]["content"] == content
    assert not list(ctx.spill_dir.glob("*.out"))
