"""test_governance_workspace.py - the configured workspace is a writable root.

Governance is advisory by design (spec 4.5): it guards ``write``/``edit`` only,
and ``bash`` bypasses it.  Before the change, a caller-supplied workspace
(the eval runner's ``/tmp/canary-eval-*/workspace``) was default-denied: the
only ``shared/workspace/...`` candidate came from the *state store*, which has
nothing to do with where the agent works.  After it, a target inside
``config.workspace`` also renders as ``shared/workspace/<rel>``, so the default
``shared/workspace/**`` allow pattern covers the declared work area.

Each test states the before/after expectation in its docstring; only the first
one fails before the change (it is the behaviour change), the other two pin
that nothing else moved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from canary.core.governance import Governance
from tests.conftest import build_config


@pytest.mark.timeout(60)
def test_configured_workspace_is_writable(root: Path, tmp_path: Path) -> None:
    """before: denied (matches no allow_write pattern) / after: allowed."""
    scratch = tmp_path / "canary-eval-abc123" / "workspace"
    scratch.mkdir(parents=True)
    cfg = build_config(root).derived(workspace=scratch)
    assert cfg.workspace == scratch.resolve()
    gov = Governance(cfg)

    target = scratch / "notes" / "release-token.txt"
    allowed, reason = gov.check(target)
    assert allowed is True, reason
    assert "shared/workspace/notes/release-token.txt" in gov.candidates(target)

    # the real tool path (what an eval agent's write call does) follows
    tool_cfg = build_config(root, scripts={"main": ["done"]}).derived(workspace=scratch)
    from canary.core.agent import Agent

    agent = Agent(tool_cfg)
    try:
        result = agent.tools.call("write", {"path": str(target), "content": "5c1f"})
    finally:
        agent.close()
    assert not result.lower().startswith("error"), result
    assert target.read_text(encoding="utf-8") == "5c1f"


@pytest.mark.timeout(60)
def test_live_agent_without_a_workspace_is_unchanged(root: Path) -> None:
    """before and after: identical - no workspace, no new candidate."""
    cfg = build_config(root)
    assert cfg.workspace is None
    gov = Governance(cfg)

    outside = root / "outside.txt"
    allowed, reason = gov.check(outside)
    assert allowed is False
    assert "no allow_write" in reason
    assert not [c for c in gov.candidates(outside) if "shared/workspace/" in c]

    # a state path inside shared/workspace still resolves as before, exactly once
    in_state = cfg.state_path / "workspace" / "x.txt"
    assert gov.allowed(in_state) is True
    assert gov.candidates(in_state).count("shared/workspace/x.txt") == 1
    assert gov.candidates(in_state)[0] == "shared/workspace/x.txt"

    # a state path outside shared/workspace stays denied
    assert gov.check(cfg.state_path / "secrets" / "s.txt")[0] is False


@pytest.mark.timeout(60)
def test_workspace_prefix_sibling_is_not_matched(root: Path, tmp_path: Path) -> None:
    """before and after: a shared string prefix is not a path prefix."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    evil = tmp_path / "ws-evil" / "x.txt"
    evil.parent.mkdir()
    cfg = build_config(root).derived(workspace=workspace)
    gov = Governance(cfg)

    assert gov.check(evil)[0] is False
    assert not [c for c in gov.candidates(evil) if c.startswith("shared/workspace/")]

    # not vacuous: a target inside the workspace is allowed by the same config
    assert gov.check(workspace / "x.txt")[0] is True
