"""test_memory_relations.py - relations round-trip, conflicts, dangling refs."""

from __future__ import annotations

from pathlib import Path

import pytest

from canary.core.memory import Entry, Memory, format_hits
from tests.conftest import build_config


@pytest.mark.timeout(60)
def test_relations_round_trip(root: Path) -> None:
    cfg = build_config(root)
    memory = Memory(cfg)
    memory.save("service atlas runs on port 8080", entry_id="atlas-8080")
    memory.save(
        "service atlas now runs on port 9090",
        entry_id="atlas-9090",
        tags=["atlas"],
        relations={"supersedes": ["atlas-8080"], "supports": ["atlas-8080"]},
    )
    fresh = Memory(cfg)
    entry = fresh.get("atlas-9090")
    assert entry.relations["supersedes"] == ["atlas-8080"]
    assert entry.relations["supports"] == ["atlas-8080"]
    assert entry.relations.get("contradicts", []) == []

    markdown = entry.to_markdown()
    parsed = Entry.from_markdown(entry.id, markdown)
    assert parsed.relations == entry.relations
    assert parsed.id == entry.id
    assert parsed.tags == entry.tags
    assert parsed.importance == entry.importance


@pytest.mark.timeout(60)
def test_conflicts_surface_in_recall(root: Path) -> None:
    cfg = build_config(root)
    memory = Memory(cfg)
    memory.save("service atlas runs on port 8080", entry_id="atlas-8080")
    memory.save(
        "service atlas runs on port 9090",
        entry_id="atlas-9090",
        relations={"contradicts": ["atlas-8080"]},
    )
    hits = memory.search("atlas port", k=1)
    ids = [h.entry.id for h in hits]
    assert set(ids) == {"atlas-8080", "atlas-9090"}
    assert len(hits) <= 3
    assert all(h.components.get("conflict") for h in hits)
    text = format_hits(hits)
    assert "[conflict]" in text
    assert "atlas-8080" in text and "atlas-9090" in text


@pytest.mark.timeout(60)
def test_supersedes_flagged_and_ordering(root: Path) -> None:
    cfg = build_config(root)
    memory = Memory(cfg)
    memory.save("legacy endpoint is /v1/run", entry_id="old-endpoint")
    memory.save(
        "current endpoint is /agent/run",
        entry_id="new-endpoint",
        tags=["endpoint"],
        relations={"supersedes": ["old-endpoint"]},
    )
    hits = memory.search("agent run endpoint", tags=["endpoint"], k=1)
    ids = [h.entry.id for h in hits]
    assert ids[0] == "new-endpoint"
    assert "old-endpoint" in ids


@pytest.mark.timeout(60)
def test_dangling_relations_are_ignored(root: Path) -> None:
    cfg = build_config(root)
    memory = Memory(cfg)
    memory.save(
        "orphan fact with dangling refs",
        entry_id="orphan",
        relations={"supersedes": ["does-not-exist"], "contradicts": ["nope"]},
    )
    hits = memory.search("orphan fact")
    assert hits and hits[0].entry.id == "orphan"
    assert not hits[0].components.get("conflict")
    assert len(hits) == 1
