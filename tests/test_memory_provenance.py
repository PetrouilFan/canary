"""test_memory_provenance.py - provenance v1: schema, population, redaction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from canary.core.memory import REDACTED, Memory, normalize_provenance
from canary.core.models import MockModel, ModelResponse
from tests.conftest import build_config


def _text(memory: Memory, entry_id: str) -> str:
    return memory.entry_path(entry_id).read_text()


@pytest.mark.timeout(60)
def test_save_persists_provenance_and_round_trips(root: Path) -> None:
    cfg = build_config(root)
    memory = Memory(cfg)
    memory.save(
        "Deployment port for service atlas is 9090",
        entry_id="atlas-port",
        source="user",
        provenance={
            "origin": "session:s-1#42",
            "author": "user",
            "derivation": "stated",
            "confidence": 0.95,
            "lineage": ["job:j-7"],
            "status": "confirmed",
            "scope": "project",
        },
    )
    fresh = Memory(cfg).get("atlas-port")
    assert fresh is not None
    provenance = fresh.provenance
    assert provenance["origin"] == "session:s-1#42"
    assert provenance["author"] == "user"
    assert provenance["derivation"] == "stated"
    assert provenance["confidence"] == 0.95
    assert provenance["lineage"] == ["job:j-7"]
    assert provenance["status"] == "confirmed"
    assert provenance["scope"] == "project"
    assert fresh.status == "confirmed"

    # unknown origin is honest, not invented: it degrades to uncertain
    memory.save("orphan fact about atlas retries", entry_id="atlas-orphan")
    orphan = Memory(cfg).get("atlas-orphan")
    assert orphan is not None
    assert orphan.provenance == {}
    assert orphan.status == "uncertain"


@pytest.mark.timeout(60)
def test_extract_stamps_derivation_without_second_model_call(root: Path) -> None:
    cfg = build_config(root)
    memory = Memory(cfg)
    payload = json.dumps(
        [{"body": "the atlas service prefers UTC timestamps", "tags": ["atlas"],
          "importance": 0.7, "derivation": "inferred", "confidence": 0.4,
          "status": "uncertain"}]
    )
    client = MockModel([ModelResponse(content=payload, model="mock")])
    saved = memory.extract("user: we always store UTC", origin="session:s-1#9", model_client=client)
    assert len(client.calls) == 1  # provenance costs no extra model call
    assert len(saved) == 1
    assert saved[0].provenance["derivation"] == "inferred"
    assert saved[0].provenance["confidence"] == 0.4
    assert saved[0].provenance["origin"] == "session:s-1#9"
    assert saved[0].provenance["lineage"] == ["session:s-1#9"]
    assert saved[0].status == "uncertain"


@pytest.mark.timeout(60)
def test_consolidate_preserves_lineage_and_never_deletes(root: Path) -> None:
    cfg = build_config(root)
    memory = Memory(cfg)
    memory.save("atlas port is 9090", entry_id="a1", importance=0.9,
                provenance={"origin": "session:s-1#1", "author": "user",
                            "derivation": "stated", "status": "confirmed"})
    memory.save("atlas port is 9090", entry_id="a2", importance=0.2,
                provenance={"origin": "session:s-2#5", "author": "agent",
                            "derivation": "observed", "status": "confirmed"})
    result = memory.consolidate(idle_after_s=0)
    keep, drop = memory.get("a1"), memory.get("a2")
    assert result["merged"] == 1
    assert drop is not None and drop.archived is True          # never deleted
    assert drop.status == "superseded"                          # status preserved on the loser
    assert "a2" in keep.provenance["lineage"]                    # history cited
    assert keep.provenance["derivation"] == "summarized"
    assert keep.provenance["origin"] == "session:s-1#1"          # origin of the survivor kept
    assert "a2" in keep.relations.get("supersedes", [])          # non-destructive link


@pytest.mark.timeout(60)
def test_provenance_redacts_secret_values(root: Path) -> None:
    cfg = build_config(root)
    memory = Memory(cfg)
    memory.save("rotate the atlas deploy credential next week", entry_id="rot",
                provenance={"origin": "external", "author": "user",
                            "derivation": "stated",
                            "scope": "api_key=sk-abcdef123456"})
    text = _text(memory, "rot")
    assert REDACTED == "[redacted]"
    assert "sk-abcdef123456" not in text
    assert "[redacted]" in text


@pytest.mark.timeout(60)
def test_search_surfaces_superseded_and_uncertain(root: Path) -> None:
    cfg = build_config(root)
    memory = Memory(cfg)
    memory.save("atlas port is 9090", entry_id="old", importance=0.9,
                provenance={"origin": "session:s-1#1", "derivation": "stated",
                            "status": "superseded"})
    memory.save("atlas port is 8080", entry_id="new", importance=0.9,
                provenance={"origin": "session:s-3#2", "derivation": "stated",
                            "status": "confirmed"})
    memory.save("atlas port might sleep", entry_id="maybe", importance=0.9,
                provenance={"origin": "session:s-4#2", "derivation": "inferred",
                            "status": "uncertain"})
    hits = {h.entry.id: h for h in memory.search("atlas port", k=5)}
    assert hits["new"].components["status"] == "confirmed"
    assert hits["old"].components["status"] == "superseded"
    assert hits["maybe"].components["status"] == "uncertain"
    assert hits["old"].score < hits["new"].score          # superseded is demoted

    from canary.core.memory import format_hits
    rendered = format_hits([hits["old"], hits["maybe"], hits["new"]])
    assert "[superseded] old" in rendered
    assert "[uncertain] maybe" in rendered
    assert "provenance" in hits["new"].to_dict()


def test_normalize_provenance_is_total() -> None:
    assert normalize_provenance(None)["status"] == "uncertain"
    assert normalize_provenance(None)["origin"] == "unknown"
    assert normalize_provenance({"derivation": "nonsense"})["derivation"] == "stated"
    assert normalize_provenance({"confidence": "nope"})["confidence"] == 0.5
    assert normalize_provenance({"confidence": 5})["confidence"] == 1.0
