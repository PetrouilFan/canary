"""test_compression.py - threshold prune, spill, pins, summary, nudges."""

from __future__ import annotations

from pathlib import Path

import pytest

from canary.core.context import ContextEngine
from canary.core.models import MockModel
from canary.core.session import SessionManager

FILLER = "x" * 2400


def _unit(index: int, tool: str = "bash", content: str = FILLER) -> list[dict]:
    call_id = f"call_{index}"
    return [
        {"role": "user", "content": f"question {index}"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool, "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "name": tool, "content": content},
    ]


def _messages(count: int = 12, recent: int = 3, tail: str = "latest question") -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": "stable tier"}]
    for i in range(count):
        messages.extend(_unit(i))
    for i in range(recent):
        messages.append({"role": "user", "content": f"{tail} {i}"})
        messages.append({"role": "assistant", "content": "ack"})
    return messages


def _engine(cfg, log, context_length: int = 4096) -> ContextEngine:
    cfg.models["models"]["main"]["context_length"] = context_length
    return ContextEngine(cfg, log)


@pytest.mark.timeout(60)
def test_threshold_prune_sheds_old_units(root: Path, cfg, log) -> None:
    ctx = _engine(cfg, log)
    messages = _messages()
    before = ctx.count_tokens(messages)
    kept, report = ctx.prune(messages, user_message="latest question")
    assert report.pruned_units > 0
    assert report.tokens_after < before == report.tokens_before
    assert any("Pruned" in m.get("content", "") for m in kept)
    assert any(
        m.get("role") == "user" and m["content"].startswith("latest question")
        for m in kept
    )


@pytest.mark.timeout(60)
def test_protected_and_pinned_survive(root: Path, cfg, log) -> None:
    ctx = _engine(cfg, log)
    manager = SessionManager(cfg, log)
    session = manager.create(name="pin-test")
    messages: list[dict] = [{"role": "system", "content": "stable tier"}]
    messages.extend(
        _unit(0, "bash", "REF-OLD output references call_refid here " + FILLER)
    )
    messages.extend(_unit(1, "write", "PROTECTED-MARKER small output"))
    for i in range(2, 12):
        messages.extend(_unit(i))
    messages.append({"role": "user", "content": "recent question"})
    messages.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_write",
                    "type": "function",
                    "function": {"name": "write", "arguments": "{}"},
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": "call_write",
            "name": "write",
            "content": "wrote file; see call_refid for the prior value",
        }
    )
    messages.append({"role": "user", "content": "latest question"})
    kept, report = ctx.prune(messages, session=session, user_message="latest question")
    assert report.pruned_units > 0
    blob = "\n".join(m.get("content") or "" for m in kept)
    assert "PROTECTED-MARKER" in blob
    assert "REF-OLD" in blob
    assert report.released_pins == []


@pytest.mark.timeout(60)
def test_spill_rule(root: Path, cfg, log) -> None:
    ctx = _engine(cfg, log)
    manager = SessionManager(cfg, log)
    session = manager.create(name="spill-test")
    big = "SPILL-MARKER\n" + ("line of output\n" * 3000)
    messages: list[dict] = [{"role": "system", "content": "stable tier"}]
    messages.extend(_unit(0, "write", big))
    for i in range(1, 12):
        messages.extend(_unit(i))
    messages.append({"role": "user", "content": "latest question"})
    kept, report = ctx.prune(messages, session=session, user_message="latest question")
    assert report.spilled, "expected protected oversized output to spill"
    spilled_path = Path(report.spilled[0])
    assert spilled_path.is_file()
    assert "line of output" in spilled_path.read_text()
    blob = "\n".join(m.get("content") or "" for m in kept)
    assert "full output at" in blob
    assert "SPILL-MARKER" in blob


@pytest.mark.timeout(60)
def test_compress_summarizes_old_units(root: Path, cfg, log) -> None:
    ctx = _engine(cfg, log)
    manager = SessionManager(cfg, log)
    session = manager.create(name="compress-test")
    messages: list[dict] = [{"role": "system", "content": "stable tier"}]
    messages.extend(_unit(0, "write", "PROTECTED-KEEP"))
    for i in range(1, 12):
        messages.extend(_unit(i))
    messages.append({"role": "user", "content": "latest question"})
    out, report = ctx.compress(
        messages,
        focus="stay on task",
        session=session,
        model_client=MockModel(["SUMMARY-TEXT condensed decisions"]),
    )
    assert out[0]["role"] == "system"
    assert "[context summary]" in out[1]["content"]
    assert "SUMMARY-TEXT" in out[1]["content"]
    blob = "\n".join(m.get("content") or "" for m in out)
    assert "PROTECTED-KEEP" in blob
    assert "latest question" in blob
    assert report.tokens_after < report.tokens_before


@pytest.mark.timeout(60)
def test_summary_saved_to_memory(root: Path, cfg, log) -> None:
    ctx = _engine(cfg, log)
    from canary.core.memory import Memory

    memory = Memory(cfg, log)
    ctx.memory = memory
    manager = SessionManager(cfg, log)
    session = manager.create(name="summary-test")
    messages = _messages()
    text, report = ctx.summarize(
        messages, model_client=MockModel(["SUMMARY-TEXT saved"]), session=session
    )
    assert "SUMMARY-TEXT" in text
    hits = memory.search("summary", tags=["context-summary"])
    assert hits or any(
        "context-summary" in e.tags for e in memory.all_entries()
    )


@pytest.mark.timeout(60)
def test_nudge_levels(root: Path, cfg, log) -> None:
    ctx = _engine(cfg, log)
    assert ctx.nudge_for(0.5) == ""
    low = ctx.nudge_for(0.81)
    high = ctx.nudge_for(0.96)
    assert low and "80%" in low
    assert high and "95%" in high
    assert low != high


@pytest.mark.timeout(60)
def test_build_assembles_tiers(root: Path, cfg, log) -> None:
    from canary.core.memory import Memory

    ctx = _engine(cfg, log)
    ctx.memory = Memory(cfg, log)
    manager = SessionManager(cfg, log)
    session = manager.create(name="build-test")
    session.append_message("user", "hello")
    session.append_message("assistant", "hi")
    built = ctx.build(
        session,
        user_message="next",
        summary="prior summary",
        tools=[{"type": "function", "function": {"name": "bash"}}],
    )
    assert built[0]["role"] == "system"
    assert "## Tools (deterministic order)" in built[0]["content"]
    assert built[1]["content"].startswith("[context summary]")
    blob = "\n".join(m.get("content") or "" for m in built)
    assert "Today (UTC)" in blob
    assert "hello" in blob
