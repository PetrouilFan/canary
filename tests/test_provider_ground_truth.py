"""count_tokens ground truth: the provider's own prompt_tokens, not a guess.

The estimate feeds the same usage_ratio -> prune/nudge path; it may only ever
be *more* pessimistic than the old `len(json.dumps) // 4` heuristic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from canary.core.agent import Agent
from canary.core.context import ContextEngine
from tests.conftest import build_config


def _engine(root: Path, **overrides) -> ContextEngine:
    cfg = build_config(root, **overrides)
    return ContextEngine(cfg)


def test_cold_start_uses_the_char_heuristic(root: Path) -> None:
    ctx = _engine(root)
    messages = [{"role": "user", "content": "hello there"}]
    assert ctx.count_tokens(messages) == max(1, ctx.chars_of(messages) // 4)


def test_anchor_adds_appended_chars_at_three_per_token(root: Path) -> None:
    ctx = _engine(root)
    base = [{"role": "user", "content": "x" * 4000}]
    ctx.note_provider_usage(5000, base, "s1")
    assert ctx.count_tokens(base) == 5000  # anchor, no content appended yet

    grown = base + [{"role": "tool", "tool_call_id": "c1", "name": "bash",
                     "content": "y" * 3000}]
    appended = ctx.chars_of(grown) - ctx.chars_of(base)
    expected = 5000 + -(-appended // 3)
    assert ctx.count_tokens(grown) == expected
    assert expected > ctx.chars_of(grown) // 4  # more pessimistic than the heuristic


def test_estimate_is_never_more_optimistic_than_the_old_heuristic(root: Path) -> None:
    ctx = _engine(root)
    messages = [{"role": "user", "content": "z" * 20000}]
    ctx.note_provider_usage(10, messages, "s1")  # an anchor well below the heuristic
    assert ctx.count_tokens(messages) == max(1, ctx.chars_of(messages) // 4)


def test_estimate_is_capped_at_the_provider_context_length(root: Path) -> None:
    ctx = _engine(root)
    messages = [{"role": "user", "content": "q" * 100}]
    ctx.note_provider_usage(10**7, messages, "s1")
    assert ctx.count_tokens(messages) == ctx.context_length
    assert ctx.usage_ratio(messages) > 1.0  # maximal: prune path must engage


def test_zero_prompt_tokens_does_not_create_an_anchor(root: Path) -> None:
    ctx = _engine(root)
    messages = [{"role": "user", "content": "a" * 400}]
    ctx.note_provider_usage(0, messages, "s1")
    assert ctx.count_tokens(messages) == max(1, ctx.chars_of(messages) // 4)


def test_anchor_from_another_session_is_dropped(root: Path) -> None:
    ctx = _engine(root)
    messages = [{"role": "user", "content": "b" * 8000}]
    ctx.note_provider_usage(9000, messages, "other")
    ctx.begin_turn("mine")
    assert ctx.count_tokens(messages) == max(1, ctx.chars_of(messages) // 4)
    ctx.begin_turn("other")  # same session keeps the anchor
    ctx.note_provider_usage(9000, messages, "other")
    ctx.begin_turn("other")
    assert ctx.count_tokens(messages) == 9000


@pytest.mark.timeout(120)
def test_turn_crosses_threshold_only_with_provider_truth(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One concrete turn: the char heuristic stays under the threshold, the
    provider's own numbers cross it.

    The mock is wrapped so it reports what a real provider did in the incident -
    about 2x the local `len // 4` estimate - and the anchored estimate is that
    `prompt_tokens` plus `chars // 3` for the tool output appended after it.
    """
    payload_chars = 30000
    script = [
        {
            "content": "",
            "tool_calls": [
                {
                    "name": "bash",
                    "arguments": {"command": f"python3 -c \"print('a'*{payload_chars})\""},
                }
            ],
        },
        "done",
    ]
    cfg = build_config(root, models=_models(context_length=21300, script=script))
    # Isolate the estimate: the per-turn tool-output budget would spill the
    # 30000-char output before the second call and hide the crossing.
    cfg.set("tools.turn_output_budget", 0)
    agent = Agent(cfg)
    seen: list[int] = []
    prunes: list[tuple[int, int, int, str | None]] = []
    client = agent.models.client_for("main")
    real_call = client.call
    real_prune = ContextEngine.prune

    def wrapper(messages, tools=None, timeout_s=None):
        raw = len(json.dumps(messages, default=str))
        seen.append(raw)
        response = real_call(messages, tools=tools, timeout_s=timeout_s)
        est = max(1, raw // 4)
        response.usage = {
            "prompt_tokens": 2 * est,
            "completion_tokens": 1,
            "total_tokens": 2 * est + 1,
        }
        return response

    def counting_prune(self, messages, **kwargs):
        prunes.append(
            (
                self.count_tokens(messages),
                len(messages),
                self._provider_tokens,
                self._provider_session,
            )
        )
        return real_prune(self, messages, **kwargs)

    monkeypatch.setattr(client, "call", wrapper)
    monkeypatch.setattr(ContextEngine, "prune", counting_prune)
    try:
        text = agent.run("spill some bytes", session_id="prov")
    finally:
        agent.close()

    assert text == "done"
    assert len(seen) == 2, seen
    ctx = ContextEngine(build_config(root, models=_models(context_length=21300)))
    limit = ctx.usable * ctx.threshold
    old_estimate = seen[1] // 4
    anchored = 2 * (seen[0] // 4) + -(-(seen[1] - seen[0]) // 3)
    # the numbers this test rests on, on the record
    assert old_estimate < limit <= anchored, (old_estimate, limit, anchored, seen)
    # and the crossing happened inside the real agent loop, not on paper only
    crossed = [row for row in prunes if row[2] and row[3] == "prov" and row[0] >= limit]
    assert crossed, prunes
    # the turn is not a single-user-turn-only turn: the crossing prunes real units
    assert max(row[1] for row in crossed) >= 5, prunes


def _models(*, context_length: int, script: list | None = None) -> dict:
    """Mock models with an explicit context_length.

    `scripts=` is ignored by `build_config` whenever `models=` is passed, so the
    script has to be attached here or the mock answers its 'ok' default.
    """
    from canary.core.config import default_models

    models = default_models()
    for role in ("main", "compression", "health"):
        models["models"][role]["provider"] = "mock"
        models["models"][role]["model"] = "mock"
        models["models"][role]["context_length"] = context_length
    if script is not None:
        models["models"]["main"]["script"] = list(script)
    return models
