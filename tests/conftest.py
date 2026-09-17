"""Shared test fixtures.

Isolation mandate (spec §15): every test runs with ``CANARY_ROOT`` and
``HARNESS_STATE_PATH`` pointed at a fresh temp directory, uses the mock
model, and never touches the network, real state, or the embedding cache.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from canary.core.config import Config, default_models

_ENV_CLEAR = (
    "HARNESS_MODEL_NAME",
    "HARNESS_MODEL_BASE_URL",
    "HARNESS_MODEL_CONTEXT_LENGTH",
    "HARNESS_MODEL_API_KEY",
    "HARNESS_API_KEY",
    "HARNESS_EMBEDDING_MODEL",
    "HARNESS_SELF_MODIFY",
    "HARNESS_MEMORY_GIT",
    "HARNESS_EVAL_GATE",
    "HARNESS_RELEASE_ID",
    "HARNESS_COMMIT_SHA",
)


@pytest.fixture()
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    root_path = tmp_path / "canary-root"
    state = root_path / "shared"
    state.mkdir(parents=True)
    monkeypatch.setenv("CANARY_ROOT", str(root_path))
    monkeypatch.setenv("HARNESS_STATE_PATH", str(state))
    monkeypatch.setenv("HARNESS_EMBEDDING_BACKEND", "hash")
    for name in _ENV_CLEAR:
        monkeypatch.delenv(name, raising=False)
    return root_path


def build_config(
    root_path: Path,
    *,
    scripts: dict[str, list[Any]] | None = None,
    models: dict[str, Any] | None = None,
    **overrides: Any,
) -> Config:
    """Create a Config on ``root_path`` with mock model scripts."""
    state = Path(root_path) / "shared"
    state.mkdir(parents=True, exist_ok=True)
    if models is None:
        models = default_models()
        for role in ("main", "compression", "health"):
            models["models"][role]["provider"] = "mock"
            models["models"][role]["model"] = "mock"
            models["models"][role]["context_length"] = 32768
        for role, script in (scripts or {}).items():
            entry = models["models"].setdefault(
                role, {"provider": "mock", "model": "mock", "context_length": 32768}
            )
            entry["provider"] = "mock"
            entry["script"] = list(script)
    (state / "models.yaml").write_text(yaml.safe_dump(models), encoding="utf-8")
    config = Config(root=Path(root_path), overrides=overrides or None)
    config.set("memory.git", False)
    return config


@pytest.fixture()
def cfg(root: Path) -> Config:
    return build_config(root)


@pytest.fixture()
def log(cfg: Config):
    from canary.core.observability import Log

    return Log(cfg)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail fast if a test accidentally tries the network."""
    monkeypatch.setenv("NO_PROXY", "*")
    os.environ.setdefault("CANARY_TEST_OFFLINE", "1")
