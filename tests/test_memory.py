"""test_memory.py - write/read/recall/forget, dedupe, merge, lock deferral."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from canary.core.memory import Memory
from canary.core.models import MockModel
from canary.core.util import FileLock, read_json
from tests.conftest import build_config


@pytest.mark.timeout(60)
def test_write_read_recall_forget(root: Path) -> None:
    cfg = build_config(root)
    memory = Memory(cfg)
    entry = memory.save(
        "Deployment port for service atlas is 9090",
        tags=["infra", "atlas"],
        entry_id="atlas-port",
        source="test",
        importance=0.8,
    )
    assert entry.id == "atlas-port"
    assert memory.get("atlas-port") is not None
    hits = memory.search("atlas port", tags=["infra"])
    assert hits and hits[0].entry.id == "atlas-port"
    assert memory.forget("atlas-port") is True
    assert memory.get("atlas-port").archived
    assert memory.search("atlas port") == []
    assert memory.search("atlas port", include_archived=True)


@pytest.mark.timeout(60)
def test_index_and_active_index(root: Path) -> None:
    cfg = build_config(root)
    memory = Memory(cfg)
    memory.save("alpha fact " * 40, entry_id="alpha", importance=0.9)
    memory.save("beta fact", entry_id="beta")
    index = (cfg.state_path / "memory" / "index.md").read_text()
    assert "alpha" in index and "beta" in index
    assert len(index.encode()) <= 5120
    active = read_json(cfg.data_path / "memory_index" / "active.json")
    assert active["model"]


@pytest.mark.timeout(60)
def test_index_cap_by_selection(root: Path) -> None:
    cfg = build_config(root)
    memory = Memory(cfg)
    for i in range(80):
        memory.save(
            f"entry {i} " + ("detail " * 30),
            entry_id=f"cap-{i:03d}",
            importance=(i % 10) / 10,
        )
    index = (cfg.state_path / "memory" / "index.md").read_text()
    assert len(index.encode()) <= 5120
    assert "entries; showing most important" in index


@pytest.mark.timeout(60)
def test_extract_dedupe_and_secret_rejection(root: Path) -> None:
    cfg = build_config(root)
    memory = Memory(cfg)
    memory.save("The database is postgres 16", entry_id="db", tags=["db"])
    script = json.dumps(
        [
            {"body": "The database is postgres 16", "tags": ["db"]},
            {"body": "The cache is redis 7", "tags": ["cache"], "importance": 0.7},
            {"body": "api key is sk-live-abc1234567890abcdef", "tags": ["secret"]},
        ]
    )
    saved = memory.extract(
        "conversation text",
        model_client=MockModel([script]),
        source="test",
    )
    ids = {e.id for e in saved}
    assert len(saved) == 1 and "redis" in saved[0].body.lower()
    assert "db" not in ids
    assert not any("sk-live" in e.body for e in memory.all_entries())


@pytest.mark.timeout(60)
def test_consolidation_merges_and_archives(root: Path) -> None:
    cfg = build_config(root)
    memory = Memory(cfg)
    memory.save("The deploy port is 8080", entry_id="dup-a", importance=0.8)
    memory.save("The deploy port is 8080", entry_id="dup-b", importance=0.4)
    result = memory.consolidate(idle_after_s=0)
    assert result["merged"] == 1
    keep = memory.get("dup-a")
    drop = memory.get("dup-b")
    assert drop.archived is True
    assert "dup-b" in keep.relations.get("supersedes", [])


@pytest.mark.timeout(60)
def test_consolidation_defers_when_not_idle(root: Path) -> None:
    cfg = build_config(root)
    memory = Memory(cfg)
    memory.save("The deploy port is 8080", entry_id="idle-a")
    memory.save("The deploy port is 8080", entry_id="idle-b")
    result = memory.consolidate(idle_after_s=3600)
    assert result["skipped"] == "not_idle"
    assert memory.get("idle-b").archived is False


@pytest.mark.timeout(60)
def test_consolidation_defers_under_write_contention(root: Path) -> None:
    cfg = build_config(root)
    memory = Memory(cfg)
    memory.save("The deploy port is 8080", entry_id="lock-a")
    memory.save("The deploy port is 8080", entry_id="lock-b")
    lock = FileLock(memory.lock_path, op="test-hold", wait=False)
    assert lock.acquire()
    try:
        result = memory.consolidate(idle_after_s=0)
        assert result["skipped"] == "lock_busy"
        assert memory.get("lock-b").archived is False
    finally:
        lock.release()


@pytest.mark.timeout(60)
def test_index_invalidation_on_backend_change(root: Path) -> None:
    cfg = build_config(root)
    memory = Memory(cfg)
    memory.save("the sky is blue", entry_id="sky")
    active_path = cfg.data_path / "memory_index" / "active.json"
    data = read_json(active_path)
    first = data["model"]
    data["model"] = "some-other-model"
    active_path.write_text(json.dumps(data))
    fresh = Memory(cfg)
    hits = fresh.search("sky")
    assert hits and hits[0].entry.id == "sky"
    rebuilt = read_json(active_path)
    assert rebuilt["model"] == first
