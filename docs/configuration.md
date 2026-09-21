# Configuration

Canary reads configuration from four places, applied in this order (later wins):

1. Built-in `DEFAULTS` (`canary/core/config.py`).
2. `{state_path}/.env`, `{CANARY_ROOT}/.env`, `{cwd}/.env` (dotenv, **never
   overrides variables already present in the process environment**).
3. `{state_path}/harness.yaml`.
4. Environment variable overrides.
5. Constructor `overrides` (CLI flags, library arguments).

`models.yaml`, `governance.yaml`, `SOUL.md`, `PERSONALITY.md` and
`INSTRUCTIONS.md` are separate files in the state store, loaded at boot and
hot-applied through the config-change pipeline.

## `harness.yaml`

A full dump of the tree below is written by `canary init`. Unknown keys are kept;
dot-path access is used everywhere.

| Key | Default | Meaning |
| --- | --- | --- |
| `port` | `8080` | API port for a copy |
| `agent.id` | `{hostname}-{port}` | Stable copy identity in logs, evals, messaging |
| `canary.port_range` | `[9000, 9100]` | Port range for canary children |
| `max_model_calls_per_turn` | `64` | Model calls allowed in one turn |
| `max_tool_calls_per_turn` | `128` | Tool calls allowed in one turn |
| `model_timeout_s` | `120` | Per model call timeout |
| `turn_timeout_s` | `1800` | Whole-turn timeout |
| `compression.threshold` | `0.50` | Silent prune trigger (fraction of usable context) |
| `compression.spill_threshold` | `32768` | Bytes above which protected tool output is spilled to disk |
| `compression.nudge_levels` | `[0.80, 0.90, 0.95]` | Context usage nudge levels |
| `pruning.recent_turns` | `3` | Recent user turns never pruned |
| `pruning.pin_max` | `50` | Maximum pinned dependency units |
| `memory.recall_k` | `10` | Recall results per query |
| `memory.recall_weights` | `{similarity: 0.6, importance: 0.3, recency: 0.1}` | Retrieval blend |
| `memory.relations` | `true` | Track supersedes/contradicts/supports and surface conflicts |
| `memory.extract` | `"gated"` | Extraction mode (`gated` = only when the turn merits it) |
| `memory.extract_max_per_turn` | `3` | Extraction cap per turn |
| `memory.dedupe_similarity` | `0.92` | Merge threshold for duplicate memories |
| `memory.consolidate_idle_min` | `15` | Idle minutes before consolidation may run |
| `memory.git` | `null` → `true` in serve mode, `false` embedded | Commit memory changes to the state repo |
| `embedding.model` | `Qwen/Qwen3-Embedding-0.6B` | Embedding model name |
| `embedding.backend` | `"auto"` | `auto`, `onnx`, `openai`, `fastembed`, `hash`, `none` |
| `embedding.download` | `false` | Allow downloading the ONNX model |
| `embedding.onnx_file` | `onnx/model_quantized.onnx` | File inside the ONNX repo |
| `embedding.cache_dir` | `null` → `{data}/models` | Model cache location |
| `embedding.dim` | `null` | Expected dimension (inferred when unset) |
| `embedding.max_length` | `512` | Tokenizer truncation |
| `embedding.openai.base_url` | `null` | Base URL for the OpenAI-compatible backend |
| `embedding.openai.api_key_env` | `HARNESS_MODEL_API_KEY` | Key env var |
| `embedding.openai.model` | `null` | Model name for that backend |
| `sessions.idle_archive_h` | `24` | Archive sessions idle this long |
| `sessions.inject_rate` | `5` | Max injections per sender→receiver per receiver turn |
| `sessions.inbox_ttl_h` | `24` | Pending inbox item TTL |
| `sessions.inbox_max` | `100` | Pending inbox cap (FIFO eviction) |
| `jobs.max_per_turn` | `10` | Job spawns per turn |
| `jobs.max_concurrent` | `32` | Concurrent jobs |
| `tools.reload_debounce_s` | `1.0` | Extension reload debounce |
| `tools.archive_keep` | `10` | Archived extension versions per name |
| `tools.bash_timeout_s` | `120` | Default `bash` timeout |
| `tools.turn_output_budget` | `24576` | Chars of tool output a turn may append before the largest results spill (`0` disables) |
| `evals.model` | `"main"` | Role used for eval runs |
| `evals.canary_tasks` | `3` | Tasks per canary gate (`0` disables) |
| `evals.gate` | `"warn"` | `off`, `warn` (mark flagged), `block` |
| `evals.tolerance` | `0.10` | Allowed pass-rate drop before flagging/blocking |
| `evals.cache_max_age_h` | `168` | Baseline cache age |
| `diagnosis.max_candidates` | `10` | Candidate eval tasks per diagnosis run |
| `releases.keep` | `10` | Release dirs retained |
| `releases.drain_timeout_s` | `60` | Graceful drain before parent exits |
| `releases.lock_timeout_s` | `300` | Publish lock wait |
| `budget.daily_tokens` | `0` | Daily budget (`0` = unlimited) |
| `budget.warn_at` | `0.80` | Warning fraction |
| `budget.enforce` | `false` | Block turns when exceeded (warn-only by default) |
| `context.workspace_files` | `[README.md, AGENTS.md, CANARY.md]` | Files injected as context tier |
| `context.max_file_bytes` | `20000` | Per-file cap |
| `context.cache_s` | `5.0` | Context tier cache |
| `state_path` | `null` | State store override (embedded default `./.canary/`) |
| `self_modify` | `null` | `true` in serve mode, `false` embedded |

## Environment variables

| Variable | Effect |
| --- | --- |
| `CANARY_ROOT` | Copy root (CLI/library default when `--root` absent) |
| `HARNESS_STATE_PATH` | State store path (default `{root}/shared`) |
| `HARNESS_PORT` | `port` |
| `HARNESS_AGENT_ID` | `agent.id` |
| `HARNESS_SELF_MODIFY` | Enable/disable the publish tools |
| `HARNESS_MEMORY_GIT` | Commit memory changes |
| `HARNESS_EMBEDDING_BACKEND` / `_MODEL` / `_DOWNLOAD` | Embedding overrides |
| `HARNESS_EVAL_GATE` | `off`/`warn`/`block` |
| `HARNESS_MODEL_NAME` | Model id for `main` and `compression`, forces provider `openai` |
| `HARNESS_MODEL_BASE_URL` | Base URL for both roles |
| `HARNESS_MODEL_CONTEXT_LENGTH` | Context window for both roles |
| `HARNESS_MODEL_API_KEY` | Default key env for model roles |
| `HARNESS_API_KEY` | Bearer token clients must present (and children use for probes) |
| `HARNESS_RELEASE_ID` | Stamped by the release launcher (default `dev`) |
| `HARNESS_COMMIT_SHA` | Stamped by the release launcher (default `unknown`) |
| `HARNESS_CANARY_PORT` | Canary child's private port |
| `LISTEN_FDS` / `LISTEN_PID` | systemd socket activation |

Because dotenv loading never overrides existing environment variables, an
exported variable always beats the `.env` file. Keep `HARNESS_API_KEY` in an
`EnvironmentFile=` outside `CANARY_ROOT` for deployments.

## `models.yaml`

```yaml
models:
  main:
    provider: openai          # "openai" (any OpenAI-compatible endpoint) or "mock"
    model: gpt-4.1-mini
    base_url: https://api.openai.com/v1
    api_key_env: HARNESS_MODEL_API_KEY
    context_length: 200000    # required, no fallback
    temperature: 0.7          # optional per-role generation params
    top_p: 1.0
    max_tokens: null
  compression:
    provider: openai
    model: gpt-4.1-mini
    base_url: https://api.openai.com/v1
    api_key_env: HARNESS_MODEL_API_KEY
    context_length: 200000
    temperature: 0.0          # compression defaults to deterministic
  health:
    provider: mock            # used by /agent/smoke and canary probes
providers:                    # optional shared limits/keys by provider name
  openai:
    limits: {rpm: 0, tpm: 0, concurrency: 0}
```

- `context_length` is required for non-mock roles; boot fails readiness without
  it.
- Roles: `main` (turns), `compression` (summaries, extraction, diagnosis),
  `health` (smoke turn; `mock` recommended so probes stay cheap and offline).
- Generation params fall back to provider defaults.

## `governance.yaml`

```yaml
allow_write:
  - shared/memory/**
  - shared/extensions/**
  - shared/workspace/**
  - shared/data/**
  - shared/SOUL.md
  - shared/PERSONALITY.md
  - shared/INSTRUCTIONS.md
  - shared/.env
  - shared/governance.yaml
  - shared/models.yaml
  - shared/harness.yaml
deny_write:
  - shared/staging/**
  - shared/evals/**
  - core/**
  - api/**
  - tests/**
  - pyproject.toml
  - releases/**
```

Semantics: **default deny, deny wins**. Patterns are matched against paths
relative to both `CANARY_ROOT` and the state store, so `shared/...` and
`core/...` forms both work. Only the `write` and `edit` tools consult governance;
`bash` does not, by design. The file is a workflow contract, not a sandbox — the
agent can edit it (which is itself allowed and audited).

## Identity files

| File | Purpose |
| --- | --- |
| `SOUL.md` | Identity: purpose, values, standing commitments |
| `PERSONALITY.md` | Voice: language, tone, verbosity, formatting |
| `INSTRUCTIONS.md` | Operating contract (eight default rules, see below) |

All three are injected into the stable context tier, changes go through the config
pipeline (validated, smoke-booted, committed), and the agent may rewrite them with
`write`/`edit`. Precedence: harness enforcement > explicit user/operator >
`INSTRUCTIONS.md` > `PERSONALITY.md`; the honesty rules are not overridable.

Default `INSTRUCTIONS.md` rules (shipped in `config.BUILTIN_INSTRUCTIONS`):

1. Verify before claiming; quote actual output.
2. Read before editing; prefer `edit`, use `write` only for new files.
3. Memory hygiene: search first, save durable facts, never store secrets.
4. Use `task` for independent subtasks and `job_spawn` for long work.
5. `[system]` messages are harness facts, not user requests.
6. Ask when ambiguous and irreversible; otherwise act.
7. Self-modification goes through `propose` with a motivation citing the observed
   failure; never write `shared/staging` directly.
8. Cross-agent messages are work; respect `visible_to` and injection rate.

When a file is missing (embedded mode), the built-in template text is used.

## Configuration changes at runtime

There is no "config server": edits to `harness.yaml`, `models.yaml`, identity
files or governance ride the same publish pipeline as code but with a lighter
tier — validate schema → smoke-boot the candidate config → commit to the state
repo (`state: config …`). Config changes do not create a new release directory;
the same release runs with new config after the child boots.

## Embedded vs. serve defaults

| Setting | Embedded (`Agent()`) | Serve (`Config(root=...)`) |
| --- | --- | --- |
| State path | `./.canary/` | `{root}/shared` |
| `self_modify` | `false` | `true` |
| `memory.git` | `false` | `true` |
| Health/probes | not constructed | yes |
| Publish tools | return an error string | active |
