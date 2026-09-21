"""test_turn_output_budget_pointer_floor.py - a reachable per-turn budget.

The ranked spill pass leaves every pointer with a 100-line head capped at 8192
chars, so its floor is ~8300 chars per result: with six or more big results the
turn can stay over budget even after every result has been spilled. The budget
must still be reachable, so already-spilled pointers are compacted to a one-line
path reference.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from canary.core.context import ContextEngine
from canary.core.session import SessionManager

BUDGET = 24576
MARK = "(read it if needed) ...]"
BIG = 50000


def _lines(tag: str, chars: int, width: int = 100) -> str:
    """Many long lines: 100 lines are clipped at 8192 chars (the pointer floor)."""
    out: list[str] = []
    size = 0
    i = 0
    while size < chars:
        line = (f"{tag} line {i:04d} ").ljust(width, "x")
        out.append(line)
        size += len(line) + 1
        i += 1
    return "\n".join(out)


def _tool_msg(call_id: str, content: str, name: str = "bash") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}


def _engine(cfg, log) -> ContextEngine:
    return ContextEngine(cfg, log)


def _session(cfg, log):
    return SessionManager(cfg, log).create(name="pointer-floor")


def _log_events(log) -> list[dict]:
    if not log.log_file.exists():
        return []
    return [
        json.loads(line)
        for line in log.log_file.read_text().splitlines()
        if line.strip()
    ]


@pytest.mark.timeout(60)
def test_six_big_results_reach_budget(root: Path, cfg, log) -> None:
    """(a) 6 x 50k: over budget after spilling, under budget after compaction."""
    cfg.set("tools.turn_output_budget", BUDGET)
    ctx = _engine(cfg, log)
    session = _session(cfg, log)
    originals = [_lines(chr(65 + i), BIG) for i in range(6)]
    messages = [_tool_msg(f"call_{i}", body) for i, body in enumerate(originals)]

    before = ctx.tool_chars(messages)
    assert before >= 300_000
    paths = ctx.enforce_turn_output_budget(messages, session=session)
    after = ctx.tool_chars(messages)

    assert len(paths) == 6, "every result is spilled once"
    assert after <= BUDGET, f"{after} chars still over {BUDGET}"
    # The fix is visible: the spill pass alone cannot reach the budget (~8300
    # chars per pointer), so some pointers must have been compacted.
    assert after < before / 10
    assert any(msg["content"].count("\n") == 0 for msg in messages)
    for i, msg in enumerate(messages):
        pointer = msg["content"]
        assert MARK in pointer, "still marked as spilled (no second spill)"
        path = ctx._pointer_path(msg)
        assert path is not None and path in pointer
        assert Path(path).read_text() == originals[i], "full output readable"


@pytest.mark.timeout(120)
def test_many_results_reach_budget(root: Path, cfg, log) -> None:
    """(b) 30 x 50k: the minimal form reaches the budget for any count."""
    cfg.set("tools.turn_output_budget", BUDGET)
    ctx = _engine(cfg, log)
    session = _session(cfg, log)
    originals = [_lines(f"R{i:02d}", BIG) for i in range(30)]
    messages = [_tool_msg(f"call_{i}", body) for i, body in enumerate(originals)]

    before = ctx.tool_chars(messages)
    paths = ctx.enforce_turn_output_budget(messages, session=session)
    after = ctx.tool_chars(messages)

    assert before >= 1_500_000
    assert len(paths) == 30
    assert after <= BUDGET, f"{after} chars still over {BUDGET}"
    assert any(msg["content"].count("\n") == 0 for msg in messages)
    for i, msg in enumerate(messages):
        assert Path(ctx._pointer_path(msg)).read_text() == originals[i]


@pytest.mark.timeout(60)
def test_under_budget_turn_is_byte_identical(root: Path, cfg, log) -> None:
    """(c) A turn already under budget is untouched."""
    cfg.set("tools.turn_output_budget", BUDGET)
    ctx = _engine(cfg, log)
    session = _session(cfg, log)
    messages = [_tool_msg(f"call_{i}", _lines(f"S{i}", 1000)) for i in range(3)]
    before = json.dumps(messages, sort_keys=True)

    assert ctx.tool_chars(messages) < BUDGET
    assert ctx.enforce_turn_output_budget(messages, session=session) == []
    assert json.dumps(messages, sort_keys=True) == before
    assert not list(ctx.spill_dir.glob("*.out"))
    assert [e for e in _log_events(log) if "budget" in e["event"]] == []


@pytest.mark.timeout(60)
def test_unreachable_budget_stops_and_logs(root: Path, cfg, log) -> None:
    """(d) Minimal pointers over a tiny budget: terminate, log, stay idempotent."""
    cfg.set("tools.turn_output_budget", 64)
    ctx = _engine(cfg, log)
    session = _session(cfg, log)
    originals = [_lines(chr(88 + i), BIG) for i in range(3)]
    messages = [_tool_msg(f"call_{i}", body) for i, body in enumerate(originals)]

    start = time.monotonic()
    ctx.enforce_turn_output_budget(messages, session=session)
    elapsed = time.monotonic() - start

    assert elapsed < 5, "one pass, no loop"
    after = ctx.tool_chars(messages)
    assert after > 64, "the budget really is unreachable here"
    assert all(msg["content"].count("\n") == 0 for msg in messages)
    for i, msg in enumerate(messages):
        assert Path(ctx._pointer_path(msg)).read_text() == originals[i]

    events = [e for e in _log_events(log) if e["event"] == "turn_output_budget_unreachable"]
    assert events, "the unsatisfiable budget must be logged"
    assert events[0]["budget"] == 64 and events[0]["chars_after"] == after
    assert events[0]["level"] == "warning"

    # A second pass must not spill the minimal pointers into new files.
    files = sorted(p.name for p in ctx.spill_dir.glob("*.out"))
    payload = json.dumps(messages, sort_keys=True)
    ctx.enforce_turn_output_budget(messages, session=session)
    assert json.dumps(messages, sort_keys=True) == payload
    assert sorted(p.name for p in ctx.spill_dir.glob("*.out")) == files
