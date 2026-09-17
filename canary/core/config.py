"""Configuration: built-in defaults, harness.yaml, models.yaml, .env, env vars.

Precedence: explicit constructor overrides > environment variables >
``harness.yaml`` > built-in defaults (spec 9). ``.env`` files are loaded into
the environment at boot without overriding what is already set.
"""

from __future__ import annotations

import atexit
import copy
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

from canary.core import util

DEFAULTS: dict[str, Any] = {
    "port": 8080,
    "agent": {"id": None},
    "canary": {"port_range": [9000, 9100]},
    "max_model_calls_per_turn": 64,
    "max_tool_calls_per_turn": 128,
    "model_timeout_s": 120,
    "turn_timeout_s": 1800,
    "compression": {
        "threshold": 0.50,
        "spill_threshold": 32768,
        "nudge_levels": [0.80, 0.90, 0.95],
    },
    "pruning": {"recent_turns": 3, "pin_max": 50},
    "memory": {
        "recall_k": 10,
        "recall_weights": {"similarity": 0.6, "importance": 0.3, "recency": 0.1},
        "relations": True,
        "extract": "gated",
        "extract_max_per_turn": 3,
        "dedupe_similarity": 0.92,
        "consolidate_idle_min": 15,
        "git": None,
    },
    "embedding": {
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "backend": "auto",
        "download": False,
        "onnx_file": "onnx/model_quantized.onnx",
        "cache_dir": None,
        "dim": None,
        "max_length": 512,
        "openai": {"base_url": None, "api_key_env": "HARNESS_MODEL_API_KEY", "model": None},
    },
    "sessions": {
        "idle_archive_h": 24,
        "inject_rate": 5,
        "inbox_ttl_h": 24,
        "inbox_max": 100,
    },
    "jobs": {"max_per_turn": 10, "max_concurrent": 32},
    "tools": {"reload_debounce_s": 1.0, "archive_keep": 10, "bash_timeout_s": 120},
    "evals": {
        "model": "main",
        "canary_tasks": 3,
        "gate": "warn",
        "tolerance": 0.10,
        "cache_max_age_h": 168,
    },
    "diagnosis": {"max_candidates": 10},
    "releases": {"keep": 10, "drain_timeout_s": 60, "lock_timeout_s": 300},
    "budget": {"daily_tokens": 0, "warn_at": 0.80, "enforce": False},
    "context": {
        "workspace_files": ["README.md", "AGENTS.md", "CANARY.md"],
        "max_file_bytes": 20000,
        "cache_s": 5.0,
    },
    "state_path": None,
    "self_modify": None,
}

_ENV_MAP = [
    ("HARNESS_PORT", "port", int),
    ("HARNESS_AGENT_ID", "agent.id", str),
    ("HARNESS_SELF_MODIFY", "self_modify", util.env_bool),
    ("HARNESS_MEMORY_GIT", "memory.git", util.env_bool),
    ("HARNESS_EMBEDDING_BACKEND", "embedding.backend", str),
    ("HARNESS_EMBEDDING_MODEL", "embedding.model", str),
    ("HARNESS_EMBEDDING_DOWNLOAD", "embedding.download", util.env_bool),
    ("HARNESS_EVAL_GATE", "evals.gate", str),
]

MODEL_ENV_KEYS = {
    "model": "HARNESS_MODEL_NAME",
    "base_url": "HARNESS_MODEL_BASE_URL",
    "context_length": "HARNESS_MODEL_CONTEXT_LENGTH",
}

DEFAULT_MODEL_KEY_ENV = "HARNESS_MODEL_API_KEY"

BUILTIN_SOUL = """# SOUL

You are Canary: a personal agent that runs, maintains, and improves itself.

## Purpose
- Be genuinely useful to your operator: do the work, not the theater of work.
- Keep what you learn; memory is shared across all your copies.
- Improve deliberately: evidence decides what to change.

## Values
- Honesty over comfort: never claim a result you did not verify.
- Continuity over novelty: durable state beats clever one-offs.
- Ownership: you have root access by design; act like it matters.
"""

BUILTIN_PERSONALITY = """# PERSONALITY

- Direct and concise. Lead with the result; details only when they help.
- Plain language, no filler, no flattery.
- Code and commands are exact; prose may be brief.
- State uncertainty plainly instead of hedging.
"""

BUILTIN_INSTRUCTIONS = """# INSTRUCTIONS

Operating contract. Precedence: harness enforcement > explicit user or operator
instruction > this file > PERSONALITY.md.

1. Verify before claiming: never report a result you did not run; quote the
   actual command output.
2. Read before edit; prefer `edit` to rewriting; `write` only for new files.
3. Memory hygiene: `memory_search` before work that may have been done before;
   `memory_save` durable facts as they are learned; never store credentials or
   secrets.
4. Use `task` for independent subtasks; use `job_spawn` for long work and do
   not block a turn waiting on it.
5. `[system]` messages are harness facts (nudges, interruptions, injections) -
   not user requests.
6. When a request is ambiguous and guessing is irreversible, ask; otherwise act.
7. Self-modification goes through `propose` with a motivation citing the
   observed failure; never write to `shared/staging` directly.
8. Cross-agent messages are work, not small talk; respect `visible_to` and the
   injection rate.
"""


def default_governance() -> dict:
    return {
        "allow_write": [
            "shared/memory/**",
            "shared/extensions/**",
            "shared/workspace/**",
            "shared/data/**",
            "shared/SOUL.md",
            "shared/PERSONALITY.md",
            "shared/INSTRUCTIONS.md",
            "shared/.env",
            "shared/governance.yaml",
            "shared/models.yaml",
            "shared/harness.yaml",
        ],
        "deny_write": [
            "shared/staging/**",
            "shared/evals/**",
            "core/**",
            "api/**",
            "tests/**",
            "pyproject.toml",
            "releases/**",
        ],
    }


def default_models() -> dict:
    return {
        "models": {
            "main": {
                "provider": "mock",
                "model": "mock",
                "base_url": None,
                "api_key_env": DEFAULT_MODEL_KEY_ENV,
                "context_length": 32768,
            },
            "compression": {
                "provider": "mock",
                "model": "mock",
                "base_url": None,
                "api_key_env": DEFAULT_MODEL_KEY_ENV,
                "context_length": 32768,
            },
            "health": {"provider": "mock", "model": "mock", "context_length": 8192},
        }
    }


def yaml_dump(obj: Any) -> str:
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)


class Config:
    """Resolved runtime configuration for one Agent."""

    def __init__(
        self,
        values: dict | None = None,
        *,
        state_path: str | os.PathLike[str] | None = None,
        root: str | os.PathLike[str] | None = None,
        ephemeral: bool = False,
        overrides: dict | None = None,
        load_files: bool = True,
        workspace: str | os.PathLike[str] | None = None,
    ):
        self.overrides = copy.deepcopy(overrides or {})
        self.ephemeral = ephemeral
        self._ephemeral_dir: str | None = None
        self.pre_errors: list[str] = []
        self.workspace: Path | None = None
        if workspace is not None:
            ws = Path(workspace).expanduser()
            if not ws.is_absolute():
                ws = Path.cwd() / ws
            self.workspace = ws.resolve()

        env_root = os.environ.get("CANARY_ROOT")
        if root is not None:
            self.root = Path(root).expanduser().resolve()
        elif env_root:
            self.root = Path(env_root).expanduser().resolve()
        else:
            self.root = None

        if state_path is not None:
            resolved = Path(state_path).expanduser()
            if not resolved.is_absolute():
                resolved = Path.cwd() / resolved
            self.state_path = resolved.resolve()
        elif os.environ.get("HARNESS_STATE_PATH"):
            self.state_path = Path(os.environ["HARNESS_STATE_PATH"]).expanduser().resolve()
        elif self.root is not None:
            self.state_path = self.root / "shared"
        else:
            self.state_path = Path.cwd() / ".canary"

        if ephemeral:
            self._ephemeral_dir = tempfile.mkdtemp(prefix="canary-ephemeral-")
            atexit.register(shutil.rmtree, self._ephemeral_dir, ignore_errors=True)
            self.state_path = Path(self._ephemeral_dir) / "state"

        self.values: dict[str, Any] = copy.deepcopy(DEFAULTS)
        self.values["state_path"] = str(self.state_path)
        if self.root is not None:
            self.values["root"] = str(self.root)

        if load_files:
            self._load_dotenvs()
            self._merge_yaml()

        self._apply_env()
        self.values = util.deep_merge(self.values, self.overrides)
        if load_files:
            self.models = self._load_models()
        else:
            self.models = copy.deepcopy(overrides.get("models", {})) if overrides else {}

        if load_files:
            self._apply_model_env()
        self._fill_model_defaults()

        if self.get("memory.git") is None:
            self.set("memory.git", self.root is not None and not ephemeral)
        if self.get("self_modify") is None:
            self.set("self_modify", self.root is not None and not ephemeral)
        if self.get("embedding.cache_dir") is None:
            self.set("embedding.cache_dir", str(self.data_path / "models"))
        if not self.get("agent.id"):
            self.set("agent.id", f"{util.hostname()}-{self.get('port')}")

        self.errors: list[str] = list(self.pre_errors)
        self.errors.extend(validate_harness(self.values))
        self.errors.extend(validate_models(self.models))

    # -- construction helpers ------------------------------------------------

    @classmethod
    def from_env(cls) -> Config:
        return cls()

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Config:
        p = Path(path).expanduser().resolve()
        state = p if p.is_dir() else p.parent
        return cls(state_path=state)

    # -- file loading --------------------------------------------------------

    def _load_dotenvs(self) -> None:
        candidates = [self.state_path / ".env"]
        if self.root is not None:
            candidates.append(self.root / ".env")
        cwd_env = Path.cwd() / ".env"
        candidates.append(cwd_env)
        seen: set[Path] = set()
        for candidate in candidates:
            try:
                key = candidate.resolve()
            except OSError:
                continue
            if key in seen:
                continue
            seen.add(key)
            util.load_dotenv(candidate, override=False)

    def _merge_yaml(self) -> None:
        harness = self.state_path / "harness.yaml"
        if harness.exists():
            try:
                loaded = yaml.safe_load(harness.read_text(encoding="utf-8")) or {}
                if isinstance(loaded, dict):
                    self.values = util.deep_merge(self.values, loaded)
            except (yaml.YAMLError, OSError) as exc:
                self.pre_errors.append(f"harness.yaml: {exc}")

    def _apply_env(self) -> None:
        for env_name, path, caster in _ENV_MAP:
            raw = os.environ.get(env_name)
            if raw is None:
                continue
            try:
                if caster is util.env_bool:
                    value = util.env_bool(env_name, False)
                else:
                    value = caster(raw)
            except (TypeError, ValueError):
                continue
            self.set(path, value)

    def _apply_model_env(self) -> None:
        have_any = any(os.environ.get(env) for env in MODEL_ENV_KEYS.values())
        if not have_any:
            return
        for role in ("main", "compression"):
            entry = self.models.setdefault("models", {}).setdefault(role, {})
            if not isinstance(entry, dict):
                entry = {}
                self.models["models"][role] = entry
            for field, env_name in MODEL_ENV_KEYS.items():
                raw = os.environ.get(env_name)
                if raw is None:
                    continue
                if field == "context_length":
                    try:
                        entry[field] = int(raw)
                    except ValueError:
                        continue
                else:
                    entry[field] = raw
            entry["provider"] = "openai"
            entry.setdefault("api_key_env", DEFAULT_MODEL_KEY_ENV)

    def _load_models(self) -> dict:
        models_path = self.state_path / "models.yaml"
        if models_path.exists():
            try:
                loaded = yaml.safe_load(models_path.read_text(encoding="utf-8")) or {}
                if isinstance(loaded, dict) and isinstance(loaded.get("models"), dict):
                    return loaded
            except (yaml.YAMLError, OSError):
                pass
        return default_models()

    def _fill_model_defaults(self) -> None:
        models = self.models.setdefault("models", {})
        main = models.get("main") or {}
        for role in ("main", "compression"):
            entry = models.setdefault(role, {})
            if not entry and main:
                models[role] = copy.deepcopy(main)
        models.setdefault(
            "health", {"provider": "mock", "model": "mock", "context_length": 8192}
        )
        for name, entry in models.items():
            if name == "providers":
                continue
            if isinstance(entry, dict):
                entry.setdefault("provider", "openai")
                entry.setdefault("api_key_env", DEFAULT_MODEL_KEY_ENV)
                entry.setdefault("context_length", 32768)

    # -- access --------------------------------------------------------------

    def get(self, path: str, default: Any = None) -> Any:
        if path == "state_path":
            return self.values.get("state_path", str(self.state_path))
        return util.dot_get(self.values, path, default)

    def set(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node = self.values
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    @property
    def data_path(self) -> Path:
        return self.state_path / "data"

    @property
    def workspace_path(self) -> Path:
        if self.workspace is not None:
            return self.workspace
        return self.state_path / "workspace"

    @property
    def base_dir(self) -> Path:
        if self.workspace is not None:
            return self.workspace
        return self.root if self.root is not None else Path.cwd()

    @property
    def serve_mode(self) -> bool:
        return self.root is not None

    @property
    def release_id(self) -> str:
        return os.environ.get("HARNESS_RELEASE_ID", "dev")

    @property
    def commit_sha(self) -> str:
        return os.environ.get("HARNESS_COMMIT_SHA", "unknown")

    def role(self, name: str) -> dict:
        entry = self.models.get("models", {}).get(name)
        if not isinstance(entry, dict):
            raise KeyError(f"unknown model role: {name}")
        return entry

    def role_names(self) -> list[str]:
        return sorted(self.models.get("models", {}).keys())

    @property
    def governance_path(self) -> Path:
        return self.state_path / "governance.yaml"

    def to_dict(self) -> dict:
        out = copy.deepcopy(self.values)
        out["models"] = self.models.get("models", {})
        out["state_path"] = str(self.state_path)
        out["root"] = str(self.root) if self.root else None
        out["ephemeral"] = self.ephemeral
        return out

    def with_overrides(self, overrides: dict) -> Config:
        return Config(
            state_path=self.state_path,
            root=self.root,
            ephemeral=self.ephemeral,
            overrides=util.deep_merge(self.overrides, overrides),
            load_files=True,
            workspace=self.workspace,
        )


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def validate_harness(values: dict) -> list[str]:
    errors: list[str] = []
    port = values.get("port")
    if not isinstance(port, int) or not (1 <= port <= 65535):
        errors.append(f"port must be an integer in 1..65535, got {port!r}")
    for key in ("max_model_calls_per_turn", "max_tool_calls_per_turn"):
        v = values.get(key)
        if not isinstance(v, int) or v <= 0:
            errors.append(f"{key} must be a positive integer, got {v!r}")
    threshold = util.dot_get(values, "compression.threshold", 0.5)
    if not (0 < float(threshold) <= 1):
        errors.append(f"compression.threshold must be in (0, 1], got {threshold!r}")
    gate = util.dot_get(values, "evals.gate", "warn")
    if gate not in ("off", "warn", "block"):
        errors.append(f"evals.gate must be one of off|warn|block, got {gate!r}")
    extract = util.dot_get(values, "memory.extract", "gated")
    if extract not in ("gated", "always", "never"):
        errors.append(f"memory.extract must be gated|always|never, got {extract!r}")
    return errors


def validate_models(models: dict) -> list[str]:
    errors: list[str] = []
    entries = models.get("models")
    if not isinstance(entries, dict) or "main" not in entries:
        errors.append("models.yaml must define at least a 'main' role")
        return errors
    for name, entry in entries.items():
        if not isinstance(entry, dict):
            errors.append(f"model role {name!r} must be a mapping")
            continue
        provider = entry.get("provider")
        if provider == "mock":
            continue
        for field in ("provider", "model", "base_url", "api_key_env", "context_length"):
            if not entry.get(field):
                errors.append(f"model role {name!r} missing required field {field!r}")
        length = entry.get("context_length")
        if length is not None and (not isinstance(length, int) or length <= 0):
            errors.append(f"model role {name!r} context_length must be a positive integer")
    return errors
