"""test_memory_redaction.py - direct save() body secrets are redacted at the one funnel."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from canary.core.memory import REDACTED, Memory, redact_secrets
from canary.core.models import MockModel
from tests.conftest import build_config


@pytest.mark.timeout(60)
def test_save_redacts_secret_in_body(root: Path) -> None:
    cfg = build_config(root)
    memory = Memory(cfg)
    memory.save("deploy key is api_key=sk-live-abc1234567890abcdef", entry_id="dep")
    raw = memory.entry_path("dep").read_text()
    assert "sk-live-abc1234567890abcdef" not in raw
    assert "[redacted]" in raw
    entry = memory.get("dep")
    assert "sk-live" not in entry.body
    assert REDACTED in entry.body
    assert entry.text == entry.safe_text


@pytest.mark.timeout(60)
def test_redaction_preserves_ordinary_prose(root: Path) -> None:
    cfg = build_config(root)
    memory = Memory(cfg)
    prose = "rotate the atlas password policy quarterly and tell the on-call about token budgets"
    memory.save(prose, entry_id="prose")
    assert memory.get("prose").body == prose
    raw = memory.entry_path("prose").read_text()
    assert prose in raw
    assert REDACTED not in raw


@pytest.mark.timeout(60)
def test_redaction_survives_index_and_reload(root: Path) -> None:
    cfg = build_config(root)
    memory = Memory(cfg)
    memory.save(
        "the prod token=ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 is rotated weekly",
        entry_id="idx",
    )
    index = (cfg.state_path / "memory" / "index.md").read_text()
    assert "ghp_" not in index
    assert REDACTED in index
    fresh = Memory(cfg).get("idx")
    assert "ghp_" not in fresh.body
    assert fresh.safe_text == fresh.body


@pytest.mark.timeout(60)
def test_redaction_is_idempotent_and_covers_bearer() -> None:
    once = redact_secrets("api_key=sk-live-abc1234567890abcdef and password: hunter2")
    assert REDACTED in once
    assert "hunter2" not in once
    assert redact_secrets(once) == once
    assert redact_secrets("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abcdefghij.klmnopqrst") == (
        "Authorization: Bearer [redacted]"
    )
    assert redact_secrets("we discussed bearer token strategy") == (
        "we discussed bearer token strategy"
    )


@pytest.mark.timeout(60)
def test_extract_still_drops_secret_bodies(root: Path) -> None:
    cfg = build_config(root)
    memory = Memory(cfg)
    script = json.dumps([{"body": "api key is sk-live-abc1234567890abcdef", "tags": ["secret"]}])
    saved = memory.extract("conversation", model_client=MockModel([script]), source="test")
    assert saved == []
    assert not any("sk-live" in e.body for e in memory.all_entries())
