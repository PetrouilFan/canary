"""test_hot_reload.py - extension load, reload, failing self-tests, removal."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from canary.core.agent import Agent
from canary.core.config import Config
from tests.integration.conftest import init_root

pytestmark = pytest.mark.integration

GREET_V1 = '''
from canary.core.tools import Tool


class GreetTool(Tool):
    name = "greet"
    description = "greet someone"
    parameters = {
        "type": "object",
        "properties": {"who": {"type": "string"}},
        "required": [],
    }

    def __call__(self, who="world", **kwargs):
        return f"hello {who}"
'''

GREET_V2 = GREET_V1.replace('f"hello {who}"', 'f"hola {who}"')

FAILING_TEST = "def test_greet_fails():\n    assert False\n"


def _settle(registry) -> None:
    time.sleep(1.2)
    registry.scan_extensions()


@pytest.mark.timeout(120)
def test_extension_lifecycle(root: Path) -> None:
    init_root(root)
    cfg = Config(root=root)
    cfg.set("memory.git", False)
    agent = Agent(config=cfg)
    try:
        assert "greet" not in agent.tools.names()

        ext = cfg.state_path / "extensions" / "greet.py"
        ext.write_text(GREET_V1, encoding="utf-8")
        agent.tools.scan_extensions(initial=True)
        assert "greet" in agent.tools.names()
        assert agent.tools.call("greet", {"who": "canary"}) == "hello canary"

        ext.write_text(GREET_V2, encoding="utf-8")
        _settle(agent.tools)
        _settle(agent.tools)
        assert agent.tools.call("greet", {"who": "canary"}) == "hola canary"

        tests_dir = cfg.state_path / "extensions" / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        (tests_dir / "greet_test.py").write_text(FAILING_TEST, encoding="utf-8")

        ext.write_text(GREET_V1, encoding="utf-8")
        _settle(agent.tools)
        _settle(agent.tools)
        result = agent.tools.call("greet", {"who": "canary"})
        assert "hola canary" in result, "old version must stay live on failed self-test"
        assert agent.tools.info()["reload_errors"].get("greet")

        (tests_dir / "greet_test.py").unlink()
        ext.write_text(GREET_V1, encoding="utf-8")
        _settle(agent.tools)
        _settle(agent.tools)
        assert agent.tools.call("greet", {"who": "canary"}) == "hello canary"

        ext.unlink()
        agent.tools.scan_extensions()
        assert "greet" not in agent.tools.names()
        archives = list((cfg.state_path / "extensions" / "archive").glob("greet_v*.py"))
        assert archives, "previous versions should be archived"
    finally:
        agent.close()
