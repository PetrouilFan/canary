"""test_governance.py - default deny, deny wins, bash bypass (documented)."""

from __future__ import annotations

from pathlib import Path

import pytest

from canary.core.governance import Governance
from tests.conftest import build_config


@pytest.mark.timeout(60)
def test_default_allow_and_deny(root: Path) -> None:
    cfg = build_config(root)
    gov = Governance(cfg)
    assert gov.allowed(cfg.state_path / "memory" / "entries" / "a.md")
    assert gov.allowed(cfg.state_path / "workspace" / "notes.txt")
    assert gov.allowed(cfg.state_path / "governance.yaml")
    assert not gov.allowed(cfg.state_path / "staging" / "canary" / "core" / "agent.py")
    assert not gov.allowed(cfg.state_path / "staging" / "pyproject.toml")
    assert not gov.allowed(root / "releases" / "20260101-000000-abc1234" / "x.py")


@pytest.mark.timeout(60)
def test_unknown_paths_default_deny(root: Path) -> None:
    cfg = build_config(root)
    gov = Governance(cfg)
    allowed, reason = gov.check(root / "secrets" / "private.txt")
    assert allowed is False
    assert "deny" in reason.lower() or "no allow" in reason.lower()


@pytest.mark.timeout(60)
def test_deny_wins_over_allow(root: Path) -> None:
    cfg = build_config(root)
    gov = Governance(cfg)
    gov.allow = ["shared/**"]
    gov.deny = ["shared/memory/**"]
    assert gov.allowed(cfg.state_path / "workspace" / "ok.txt")
    assert not gov.allowed(cfg.state_path / "memory" / "blocked.md")


@pytest.mark.timeout(60)
def test_bash_bypasses_governance(root: Path) -> None:
    """Governance is advisory (spec 4.5): bash is ungoverned by design."""
    cfg = build_config(root, scripts={"main": ["done"]})
    gov = Governance(cfg)
    target = cfg.state_path / "staging" / "canary" / "core" / "agent.py"
    assert not gov.allowed(target)
    from canary.core.agent import Agent

    agent = Agent(cfg)
    try:
        result = agent.tools.call(
            "bash",
            {"command": f"mkdir -p {target.parent} && echo bypassed > {target}"},
        )
        assert "error" not in result.lower()
        assert target.read_text().strip() == "bypassed"
    finally:
        agent.close()


@pytest.mark.timeout(60)
def test_write_tool_respects_governance(root: Path) -> None:
    cfg = build_config(root, scripts={"main": ["done"]})
    from canary.core.agent import Agent

    agent = Agent(cfg)
    try:
        denied = agent.tools.call(
            "write", {"path": "shared/staging/canary/evil.py", "content": "x"}
        )
        assert denied.startswith("error:") or "deny" in denied.lower()
        assert not (cfg.state_path / "staging" / "canary" / "evil.py").exists()
        allowed = agent.tools.call(
            "write", {"path": "shared/workspace/note.txt", "content": "hello"}
        )
        assert not allowed.lower().startswith("error")
        assert (cfg.state_path / "workspace" / "note.txt").read_text() == "hello"
    finally:
        agent.close()
