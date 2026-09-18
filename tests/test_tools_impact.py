"""test_tools_impact.py - descriptive impact registry (scoped autonomy v1).

Impact classes are observational metadata: surfaced via ToolRegistry.impact()
and GET /agent/tools, never inside spec() (specs() feeds the provider prompt).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from canary.core.agent import Agent
from canary.core.tools import BUILTIN_TOOLS, Tool
from tests.conftest import build_config


@pytest.fixture()
def agent(root: Path):
    cfg = build_config(root, scripts={"main": ["done"]})
    agent = Agent(cfg)
    yield agent
    agent.close()


class _ExtTool(Tool):
    name = "ext_probe"
    description = "extension probe"
    parameters = {"type": "object", "properties": {}}

    def __call__(self, *args, **kwargs) -> str:  # pragma: no cover - stub
        return "ok"


@pytest.mark.timeout(120)
def test_builtin_impact_mapping(agent: Agent) -> None:
    imp = agent.tools.impact()
    assert imp["bash"] == {"impact": "unbounded", "source": "builtin"}
    assert imp["job_spawn"] == {"impact": "unbounded", "source": "builtin"}
    assert imp["read"] == {"impact": "free", "source": "builtin"}
    assert imp["propose"] == {"impact": "high", "source": "builtin"}
    assert imp["session_send"] == {"impact": "high", "source": "builtin"}
    for name in agent.tools.names():
        assert imp[name]["impact"] in {"free", "high", "unbounded"}
        assert imp[name]["source"] == "builtin"
    assert len(imp) == len(agent.tools.names())
    assert json.dumps(agent.tools.info()["impact"]) == json.dumps(imp)


@pytest.mark.timeout(120)
def test_governance_override_wins(agent: Agent) -> None:
    agent.governance.impact = {"bash": "free", "read": "high"}
    imp = agent.tools.impact()
    assert imp["bash"] == {"impact": "free", "source": "governance"}
    assert imp["read"] == {"impact": "high", "source": "governance"}


@pytest.mark.timeout(120)
def test_governance_impact_read_from_file(root: Path) -> None:
    from canary.core.governance import Governance

    cfg = build_config(root)
    cfg.governance_path.write_text(
        "allow_write: []\ndeny_write: []\nimpact:\n  bash: free\n  ext_probe: high\n",
        encoding="utf-8",
    )
    assert Governance(cfg).impact == {"bash": "free", "ext_probe": "high"}
    agent = Agent(cfg)
    try:
        assert agent.tools.impact()["bash"] == {"impact": "free", "source": "governance"}
    finally:
        agent.close()


@pytest.mark.timeout(120)
def test_extension_defaults_to_high(agent: Agent) -> None:
    assert agent.tools.register(_ExtTool(agent))
    assert "ext_probe" not in {cls.name for cls in BUILTIN_TOOLS}
    assert agent.tools.impact()["ext_probe"] == {
        "impact": "high",
        "source": "extension-default",
    }


@pytest.mark.timeout(120)
def test_extension_governance_entry_used(agent: Agent) -> None:
    assert agent.tools.register(_ExtTool(agent))
    agent.governance.impact = {"ext_probe": "free"}
    assert agent.tools.impact()["ext_probe"] == {"impact": "free", "source": "governance"}


@pytest.mark.timeout(120)
def test_specs_byte_identical(agent: Agent) -> None:
    before = json.dumps(agent.tools.specs())
    assert '"impact"' not in before and '"source"' not in before
    agent.governance.impact = {"bash": "high", "read": "unbounded"}
    assert json.dumps(agent.tools.specs()) == before
    for spec in agent.tools.specs():
        assert set(spec) <= {"type", "function", "profiles"}
        assert "impact" not in spec and "impact" not in spec["function"]
