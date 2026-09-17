"""test_boot.py - Agent starts, /health + /ready pass, tool registry populates."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from canary.core.agent import Agent
from tests.conftest import build_config

EXPECTED_TOOLS = {
    "read",
    "write",
    "edit",
    "bash",
    "propose",
    "revert",
    "compress",
    "memory_search",
    "memory_save",
    "memory_forget",
    "memory_consolidate",
    "sessions_list",
    "session_read",
    "session_send",
    "task",
    "evals_run",
    "diagnose_run",
    "job_spawn",
    "job_list",
    "job_status",
    "job_tail",
    "job_kill",
    "job_log",
}


@pytest.mark.timeout(60)
def test_agent_boots_with_tools(root: Path) -> None:
    cfg = build_config(root, scripts={"main": ["hello"]})
    agent = Agent(cfg)
    try:
        names = set(agent.tools.names())
        assert EXPECTED_TOOLS <= names
        assert agent.tools.available()
        specs = agent.tools.specs()
        assert [s["function"]["name"] for s in specs] == sorted(
            s["function"]["name"] for s in specs
        )
        for spec in specs:
            assert spec["type"] == "function"
            assert spec["function"]["description"]
            assert "parameters" in spec["function"]
    finally:
        agent.close()


@pytest.mark.timeout(60)
def test_run_returns_text_and_records_usage(root: Path) -> None:
    cfg = build_config(root, scripts={"main": ["hello from mock"]})
    agent = Agent(cfg)
    try:
        text = agent.run("hi", session_id="boot")
        assert text == "hello from mock"
        assert agent.last_usage["total_tokens"] > 0
        assert agent.last_result["session_id"] == "boot"
    finally:
        agent.close()


def _asgi_get(server, path: str) -> httpx.Response:
    import asyncio

    async def call() -> httpx.Response:
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://canary"
        ) as client:
            return await client.get(path)

    return asyncio.run(call())


@pytest.mark.timeout(60)
def test_health_and_ready_endpoints(root: Path) -> None:
    cfg = build_config(root, scripts={"main": ["ok"]})
    agent = Agent(cfg)
    assert agent.health is not None
    try:
        from canary.api.server import APIServer

        server = APIServer(agent, allow_insecure_local=True)
        health = _asgi_get(server, "/health")
        assert health.status_code == 200
        assert health.json()["release_id"] == cfg.release_id
        assert health.json()["agent_id"] == cfg.get("agent.id")
        ready = _asgi_get(server, "/ready")
        assert ready.status_code == 200
        assert ready.json()["ready"] is True
    finally:
        agent.close()


@pytest.mark.timeout(60)
def test_auth_rejects_without_key(root: Path) -> None:
    cfg = build_config(root, scripts={"main": ["ok"]})
    agent = Agent(cfg)
    try:
        from canary.api.server import APIServer

        server = APIServer(agent, allow_insecure_local=False)
        assert _asgi_get(server, "/health").status_code == 401
    finally:
        agent.close()
