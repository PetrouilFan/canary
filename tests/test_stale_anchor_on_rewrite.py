"""A provider anchor is ground truth only for the payload it was measured on.

`ContextEngine` may estimate tokens as `anchor.prompt_tokens + ceil(appended/3)`
while the payload *extends* the anchored one.  `compress()` and `prune()` replace
that payload, so the anchor stops describing it: `appended` is clamped at 0 and
`count_tokens` returns the old, larger count for a much smaller list.  The ratio
is then inflated and the next `_maybe_prune` (agent loop, top of every iteration)
runs - and can evict units, including the `[context summary]` message `compress`
just inserted, i.e. partly undoing it - until the next main call re-anchors.

Invariant under test: any rewrite re-anchors the estimate - rescaled to the new
payload so the measured tokens-per-char is preserved, never left stale.  A
rewrite that changed nothing (prune with no eviction and no spill) leaves the
anchor valid.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from canary.core.agent import Agent
from canary.core.context import ContextEngine
from canary.core.models import ContextOverflow
from tests.conftest import build_config

FILLER = "x" * 2400
# context_length 21300 -> usable 19170, threshold 0.5 -> limit 9585.
# The pre-compress payload is legitimately over the limit (the provider charged
# 12000 tokens for it); the compressed payload is ~3.7k by the char heuristic.
CONTEXT_LENGTH = 21300
ANCHOR_TOKENS = 12000


def _unit(index: int, content: str = FILLER) -> list[dict]:
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
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "name": "bash", "content": content},
    ]


def _messages(count: int = 12) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": "stable tier"}]
    for i in range(count):
        messages.extend(_unit(i))
    for i in range(3):
        messages.append({"role": "user", "content": f"latest question {i}"})
        messages.append({"role": "assistant", "content": "ack"})
    return messages


def _has_summary(messages: list[dict]) -> bool:
    return any(
        isinstance(m.get("content"), str)
        and m["content"].startswith("[context summary]")
        for m in messages
    )


def _models(context_length: int = CONTEXT_LENGTH, script: list | None = None) -> dict:
    from canary.core.config import default_models

    models = default_models()
    for role in ("main", "compression", "health"):
        models["models"][role]["provider"] = "mock"
        models["models"][role]["model"] = "mock"
        models["models"][role]["context_length"] = context_length
    if script is not None:
        models["models"]["main"]["script"] = list(script)
    return models


def _engine(root: Path, *, context_length: int = CONTEXT_LENGTH) -> ContextEngine:
    return ContextEngine(build_config(root, models=_models(context_length)))


@pytest.mark.timeout(60)
def test_compress_rescales_the_anchor(root: Path) -> None:
    """Engine-level compress: the estimate rescales, it does not go stale."""
    ctx = _engine(root)
    messages = _messages()
    ctx.note_provider_usage(ANCHOR_TOKENS, messages, "s1")
    assert ctx.usage_ratio(messages) > ctx.threshold  # the anchor is valid here

    out, report = ctx.compress(messages, model_client=None)
    assert report.tokens_after < report.tokens_before
    assert _has_summary(out)

    heuristic = max(1, ctx.chars_of(out) // 4)
    scaled = max(1, round(ANCHOR_TOKENS * ctx.chars_of(out) / ctx.chars_of(messages)))
    # before the fix: ANCHOR_TOKENS (12000), i.e. the pre-compress count for a
    # payload the provider has never seen; a plain clear would report `heuristic`
    assert ctx.count_tokens(out) == max(heuristic, scaled)
    assert ctx.count_tokens(out) < ANCHOR_TOKENS
    assert ctx.usage_ratio(out) <= ctx.threshold

    kept, pruned = ctx.prune(out, user_message="latest question 2")
    # before the fix: pruned_units >= 1 and the summary is gone - the compress
    # this very turn performed is partly undone on the next loop iteration
    assert pruned.pruned_units == 0, "prune ran on a payload below the threshold"
    assert kept == out
    assert _has_summary(kept)


@pytest.mark.timeout(60)
def test_prune_rescales_the_anchor_only_when_it_rewrites(root: Path) -> None:
    """Engine-level prune: a real eviction rescales, a second pass is a no-op."""
    ctx = _engine(root)
    messages = _messages()
    ctx.note_provider_usage(ANCHOR_TOKENS, messages, "s1")

    kept, report = ctx.prune(messages, user_message="latest question 2")
    assert report.pruned_units > 0
    heuristic = max(1, ctx.chars_of(kept) // 4)
    scaled = max(1, round(ANCHOR_TOKENS * ctx.chars_of(kept) / ctx.chars_of(messages)))
    assert ctx.count_tokens(kept) == max(heuristic, scaled)
    assert ctx.count_tokens(kept) < ANCHOR_TOKENS

    kept2, report2 = ctx.prune(kept, user_message="latest question 2")
    # before the fix: the stale anchor keeps the ratio over the threshold and
    # this second pass evicts again (report2.pruned_units > 0)
    assert report2.pruned_units == 0
    assert kept2 == kept


@pytest.mark.timeout(60)
def test_anchor_survives_a_prune_that_changes_nothing(root: Path) -> None:
    """No rewrite, no invalidation: the payload is still the anchored one."""
    ctx = _engine(root)
    messages = [
        {"role": "system", "content": "stable tier"},
        {"role": "user", "content": "hello"},
    ]
    ctx.note_provider_usage(4000, messages, "s1")
    kept, report = ctx.prune(messages, user_message="hello")
    assert report.pruned_units == 0 and report.spilled == []
    assert kept == messages
    assert ctx.count_tokens(messages) == 4000


@pytest.mark.timeout(120)
def test_forced_compress_does_not_leave_a_stale_anchor(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Incident path: ContextOverflow -> _forced_compress -> continue -> prune.

    The next loop iteration evaluates the *compressed* payload with the
    pre-compression anchor; before the fix it prunes again, entering with the
    freshly written `[context summary]` in the payload.
    """
    script = [
        {
            "content": "",
            "tool_calls": [
                {
                    "name": "bash",
                    "arguments": {"command": "python3 -c \"print('a'*2000)\""},
                }
            ],
        },
        "done",
    ]
    cfg = build_config(root, models=_models(script=script))
    cfg.set("tools.turn_output_budget", 0)
    agent = Agent(cfg)
    # Seed real prior turns: compress keeps the last `recent_turns` turns, so a
    # single-user-turn session has nothing to summarize and would early-return.
    session = agent.sessions.create(name="stale")
    for i in range(6):
        session.append_message("user", f"question {i}")
        session.append_message("assistant", "x" * 2400)
    client = agent.models.client_for("main")
    real_call = client.call
    real_prune = ContextEngine.prune
    prunes: list[tuple[int, int, bool, int]] = []
    calls: list[tuple[int, bool]] = []
    counter = {"n": 0}

    def wrapper(messages, tools=None, timeout_s=None):
        counter["n"] += 1
        if counter["n"] == 2:
            raise ContextOverflow("synthetic overflow")
        response = real_call(messages, tools=tools, timeout_s=timeout_s)
        # what the provider charged; the same shape that triggered the review
        response.usage = {
            "prompt_tokens": ANCHOR_TOKENS,
            "completion_tokens": 1,
            "total_tokens": ANCHOR_TOKENS + 1,
        }
        calls.append((counter["n"], _has_summary(messages)))
        return response

    def counting_prune(self, messages, **kwargs):
        prunes.append(
            (
                self.count_tokens(messages),
                len(messages),
                _has_summary(messages),
                self._provider_tokens,
            )
        )
        return real_prune(self, messages, **kwargs)

    monkeypatch.setattr(client, "call", wrapper)
    monkeypatch.setattr(ContextEngine, "prune", counting_prune)
    try:
        text = agent.run("overflow please", session_id=session.id)
    finally:
        agent.close()

    assert text == "done"
    assert counter["n"] == 3, counter
    # the third call happens after the compress, and sees the summary
    assert calls[-1][0] == 3 and calls[-1][1] is True, calls
    # The entry prune (build()) and the iteration-2 prune are both legitimate:
    # the anchor is valid for those payloads.  Before the fix a third row
    # appears, entered with the freshly written summary in the payload
    # (has_summary True) under the stale anchor.
    assert len(prunes) == 2, prunes
    assert not any(row[2] for row in prunes), prunes
