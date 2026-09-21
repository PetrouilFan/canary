"""test_agent_config_overrides.py - explicit Agent kwargs win over a Config.

``Agent(config=<Config>, state_path=..., ephemeral=..., workspace=...)`` used to
silently drop everything but ``workspace``, so the eval runner's scratch state
directory was ignored and eval agents logged into the live harness state.
"""

from __future__ import annotations

from pathlib import Path

from canary.core.agent import Agent
from tests.conftest import build_config


def test_no_kwargs_keeps_the_passed_config(root: Path) -> None:
    cfg = build_config(root)
    assert Agent(cfg).config is cfg


def test_named_state_path_and_workspace_beat_a_passed_config(
    root: Path, tmp_path: Path
) -> None:
    cfg = build_config(root)
    state = tmp_path / "scratch" / "state"
    workspace = tmp_path / "scratch" / "workspace"
    agent = Agent(config=cfg, state_path=state, ephemeral=True, workspace=workspace)
    assert agent.config.state_path == state.resolve()
    assert agent.config.workspace == workspace.resolve()
    assert agent.config.root == cfg.root  # resolved root is inherited
    assert agent.log.log_file == state.resolve() / "logs" / "harness.log"
    # a named directory is the more specific request, so it beats ephemeral
    assert agent.config.ephemeral is False
    # the passed config is not mutated
    assert cfg.state_path != agent.config.state_path
    assert cfg.ephemeral is False


def test_ephemeral_alone_still_makes_a_temp_dir(root: Path) -> None:
    cfg = build_config(root)
    agent = Agent(config=cfg, ephemeral=True)
    assert agent.config.ephemeral is True
    assert "canary-ephemeral-" in str(agent.config.state_path)


def test_workspace_only_override_keeps_state_and_root(
    root: Path, tmp_path: Path
) -> None:
    cfg = build_config(root)
    workspace = tmp_path / "ws"
    agent = Agent(config=cfg, workspace=workspace)
    assert agent.config.workspace == workspace.resolve()
    assert agent.config.state_path == cfg.state_path
    assert agent.config.root == cfg.root
