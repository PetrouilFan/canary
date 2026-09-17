"""test_tools.py - every built-in tool schema plus behaviour smoke."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from canary.core.agent import Agent
from tests.conftest import build_config
from tests.test_boot import EXPECTED_TOOLS


@pytest.fixture()
def agent(root: Path):
    cfg = build_config(root, scripts={"main": ["done"]})
    agent = Agent(cfg)
    yield agent
    agent.close()


@pytest.mark.timeout(120)
def test_every_builtin_has_schema(agent: Agent) -> None:
    assert set(agent.tools.names()) == EXPECTED_TOOLS
    for name in sorted(EXPECTED_TOOLS):
        spec = agent.tools.get(name).spec()
        assert spec["type"] == "function"
        assert spec["function"]["name"] == name
        assert spec["function"]["description"]
        assert spec["function"]["parameters"]["type"] == "object"


@pytest.mark.timeout(120)
def test_unknown_and_disabled_tools(agent: Agent) -> None:
    assert agent.tools.call("nope", {}).startswith("error: unknown tool")
    assert agent.tools.disable("job_log")
    try:
        assert agent.tools.call("job_log", {"job_id": "x"}).startswith("error: tool")
        assert "job_log" in agent.tools.disabled()
    finally:
        assert agent.tools.enable("job_log")


@pytest.mark.timeout(120)
def test_file_tools_round_trip(agent: Agent) -> None:
    write = agent.tools.call(
        "write", {"path": "shared/workspace/t.txt", "content": "alpha\nbeta\n"}
    )
    assert not write.startswith("error")
    read = agent.tools.call("read", {"path": "shared/workspace/t.txt"})
    assert "alpha" in read and "beta" in read
    edit = agent.tools.call(
        "edit",
        {"path": "shared/workspace/t.txt", "old": "beta", "new": "gamma"},
    )
    assert not edit.startswith("error")
    assert "gamma" in agent.tools.call("read", {"path": "shared/workspace/t.txt"})
    missing = agent.tools.call("read", {"path": "shared/workspace/nope.txt"})
    assert missing.startswith("error")


@pytest.mark.timeout(120)
def test_bash_tool(agent: Agent) -> None:
    out = agent.tools.call("bash", {"command": "echo tool-bash-ok"})
    assert "tool-bash-ok" in out
    timed = agent.tools.call("bash", {"command": "sleep 5", "timeout": 1})
    assert "timed out" in timed.lower() or "timeout" in timed.lower()


@pytest.mark.timeout(120)
def test_memory_tools(agent: Agent) -> None:
    saved = agent.tools.call(
        "memory_save", {"body": "the workspace lives on disk", "tags": ["fs"]}
    )
    assert not saved.startswith("error")
    found = agent.tools.call("memory_search", {"query": "workspace disk"})
    assert "workspace lives on disk" in found
    entry_id = json.loads(saved)["entry_id"] if saved.strip().startswith("{") else None
    if entry_id:
        forgotten = agent.tools.call("memory_forget", {"entry_id": entry_id})
        assert not forgotten.startswith("error")
    assert not agent.tools.call("memory_consolidate", {}).startswith("error")


@pytest.mark.timeout(120)
def test_session_tools(agent: Agent) -> None:
    session = agent.sessions.create(name="tools-test")
    listing = agent.tools.call("sessions_list", {})
    assert session.id in listing
    sent = agent.tools.call(
        "session_send",
        {"session_id": session.id, "message": "hi there", "from_agent": "tester"},
    )
    assert "queued" in sent or "delivered" in sent
    read = agent.tools.call("session_read", {"session_id": session.id})
    assert "hi there" in read or "empty" in read


@pytest.mark.timeout(120)
def test_job_tools(agent: Agent) -> None:
    spawned = json.loads(agent.tools.call("job_spawn", {"command": "echo job-ok"}))
    job_id = spawned["job_id"]
    deadline = time.time() + 10
    status = ""
    while time.time() < deadline:
        status = agent.tools.call("job_status", {"job_id": job_id})
        if "done" in status:
            break
        time.sleep(0.2)
    assert "job-ok" in status or "done" in status
    assert "job-ok" in agent.tools.call("job_tail", {"job_id": job_id})
    assert job_id in agent.tools.call("job_list", {})
    log = json.loads(agent.tools.call("job_log", {"job_id": job_id}))
    assert log["total_lines"] >= 1
    long_job = json.loads(agent.tools.call("job_spawn", {"command": "sleep 30"}))
    killed = agent.tools.call("job_kill", {"job_id": long_job["job_id"]})
    assert not killed.startswith("error")


@pytest.mark.timeout(120)
def test_compress_tool_outside_turn(agent: Agent) -> None:
    out = agent.tools.call("compress", {})
    assert out.startswith("error: no active turn")


@pytest.mark.timeout(120)
def test_task_tool_spawns_worker(root: Path) -> None:
    cfg = build_config(root, scripts={"main": ["worker says hi", "parent done"]})
    agent = Agent(cfg)
    try:
        out = agent.tools.call("task", {"prompt": "say hi"})
        assert not out.startswith("error")
        assert "worker says hi" in out
    finally:
        agent.close()
