# Implementation notes and deviations from the spec

The authoritative specification is `/home/petrouil/Obsidian/10-Projects/canary/spec.md`
(v2.3, "Canary — Agent Harness Spec"). This implementation follows it closely;
this page records every intentional difference and the reasoning behind it, so a
future maintainer does not "fix" a deviation back into a bug.

## Deviations

### 1. Canary port lease file name

- **Spec**: `shared/data/.ports.json`.
- **Implementation**: `shared/data/canary_ports.json` (+ `canary_ports.lock`).
- **Why**: the file sits next to other data files whose names are already
  descriptive (`jobs/`, `sessions/`, `patches/`); a visible name is easier to
  reason about during incident response. Function and semantics are unchanged.

### 2. Release stamp uniqueness + green tag ordering by creation time

- **Spec**: zero-padded UTC stamps are enough to order `green/*` tags.
- **Implementation**: `Health._unique_stamp()` waits for a UTC second with no
  existing `green/{stamp}-*` tag before starting a publish, so release ids stay
  chronologically sortable as plain strings. `Health.green_tags()` additionally
  orders by `git for-each-ref --sort=creatordate`, and `_do_revert` picks the
  newest green *strictly older than* the current release.
- **Why**: two publishes can otherwise land within the same UTC second, so the
  stamp would not be unique and lexicographic ordering would fall back to the
  sha, pointing a revert at the wrong release. Tag creation times also have
  second resolution, so stamps — not tag timestamps — are the ordering truth;
  creation order is kept as a defense for legacy tags. The "strictly older" rule
  also prevents a revert from ping-ponging between two releases.

### 3. `canary init` writes model env vars only for non-mock providers

- **Spec**: init seeds `shared/.env` from the main model role.
- **Implementation**: `HARNESS_MODEL_BASE_URL/NAME/CONTEXT_LENGTH` are written
  only when `main.provider not in (None, "mock")`, plus the API key env when set.
- **Why**: `HARNESS_MODEL_NAME` forces `provider: openai` at boot. Writing
  `HARNESS_MODEL_NAME=mock` made a fresh mock root fail readiness with
  `model role 'main' missing required field 'base_url'`.

### 4. Releases are built from the staging snapshot

- **Spec**: implicitly, staging is the single source of code.
- **Implementation detail**: `canary init` copies the project into
  `shared/staging`. After that, repository edits do not affect a root until they
  arrive through a publish patch (or a fresh `init`).
- **Why**: this is what makes the pipeline auditable; it also means a developer
  changing harness code must publish that change (or re-init a scratch root) to
  see it in a release.

### 5. Publish drains asynchronously

- **Spec**: the parent sets `/ready` 503, drains, exits.
- **Implementation**: `Health.canary_release` performs the drain on a daemon
  thread after a short delay instead of calling the drain callback inline.
- **Why**: publishing from inside the serving process (API `/agent/propose` or a
  tool call) otherwise deadlocks: the drain waits for the in-flight request while
  the request waits for the publish to return. Live E2E reproduced this as an
  HTTP 500 after the 60s uvicorn graceful timeout, even though the publish had
  succeeded.

### 6. Commit sha is stamped into the child environment

- **Spec**: revision identity everywhere.
- **Implementation**: `canary_release` sets `HARNESS_COMMIT_SHA` from the release
  tag (`Health._release_sha`), falling back to the release name's sha suffix.
- **Why**: without it, promoted children reported `commit_sha: unknown`, which
  broke `/health`-based revision tracking and audit correlation.

### 7. Canary children sync health state before serving

- **Spec**: a promoted child can itself publish/revert.
- **Implementation**: `APIServer._main()` calls `_sync_health()` in canary mode
  (setting `listener_fd`, `port`, `drain_callback`).
- **Why**: otherwise a promoted child's publish/revert ran without the production
  fd, returning `promoted: false` and leaving the old listener alive plus a stray
  un-promoted child.

### 8. ONNX embeddings are computed one text at a time

- **Spec**: silent per-entry embedding.
- **Implementation**: `Memory._embed_each` loops single-text forward passes when
  the backend is ONNX.
- **Why**: the Qwen3-Embedding-0.6B quantized ONNX export is not
  batch-composition-invariant: the same text embedded alone vs. padded next to a
  longer sibling differs (cosine ~0.95–0.97). Embedding each text alone makes
  index and query geometry self-consistent. Ranking quality is still good;
  throughput is acceptable for memory-sized workloads.

### 9. Extended configuration keys

The spec table is a minimum, not a maximum. Added keys (all defaulted, all
documented in [configuration.md](configuration.md)):

| Key | Why |
| --- | --- |
| `tools.bash_timeout_s` | The `bash` tool needs a concrete timeout; 120s |
| `embedding.onnx_file` | Which ONNX file to fetch from the HF repo |
| `embedding.max_length` | Tokenizer truncation for long memories |
| `embedding.openai.*` | The OpenAI-compatible embedding backend needs its own model/base URL |
| `context.cache_s` | Context tier files are re-read on a 5s cadence, not every call |
| `providers` in `models.yaml` | Shared rate limits per provider instead of per role |

The `.env` keys `HARNESS_MODEL_NAME/BASE_URL/CONTEXT_LENGTH` are an addition (the
spec only describes `models.yaml`); they exist for twelve-factor deployments and
force the openai provider because their presence implies a live HTTP model.

### 10. Readiness gate override for offline reverts

- **Spec**: every pipeline starts with the base gate (`/ready` 200).
- **Implementation**: `Health.base_gate_override` lets the CLI perform a local
  revert when no copy is running; the override is logged as a warning.
- **Why**: `canary revert` must work on a stopped copy. All safety checks after
  the gate (patch application, canary boot, promotion) are unchanged. The in-process
  API never sets the override.

### 11. `--canary-mode --canary-port` instead of `--canary-probe`

- **Spec** names a `--canary-probe` flag on `canary serve`.
- **Implementation**: `--canary-mode` marks the child; `--canary-port` selects the
  private port; the parent always probes afterward.
- **Why**: the probe is the parent's responsibility, not a mode of the child;
  this keeps the child's CLI explicit about what it serves.

### 12. Eval candidate file naming and gate `warn` semantics

- Candidates are `data/eval_candidates/{UTC-stamp}-{i}.yaml` (spec suggests a
  stamp); never auto-executed.
- `warn` mode marks the deploy `flagged: true` when the delta is below
  `-evals.tolerance`. "Below" means a *drop*, which is what the operator cares
  about; improvements and ties never flag.

## Intentional non-goals (same as the spec)

- **Governance is advisory.** `bash` bypasses it, and the agent can edit
  `governance.yaml`. This is a workflow contract plus an audit trail, not a
  sandbox. The implementation does not pretend otherwise.
- **No artificial limiters.** Turn/tool/model caps exist to bound runaway loops,
  not to constrain capability.
- **Local filesystem only.** `flock` and atomic rename guarantees do not hold on
  NFS; Canary assumes a local POSIX filesystem.
- **Single host.** Cross-host coordination (leases, fencing tokens, distributed
  locks) is out of scope; the shared store is for copies on the same machine.
- **No automatic eval promotion.** Diagnosis produces candidates; a human (or the
  operator's git flow) promotes them into `shared/evals/`.

## Verification status

- Unit suite: 46 tests covering boot, memory, relations, governance, tools,
  compression, evals, preflight.
- Integration suite: 12 tests covering publish e2e, codebase lock race,
  cross-process memory, hot reload, injection, revert, port GC, fd handover and
  drain, eval gate.
- Live E2E performed against a real OpenAI-compatible gateway and the real
  Qwen3-Embedding-0.6B ONNX model: init → serve → live turns → publish →
  promotion → nested publish → nested revert, all through inherited fds.
- `ruff check canary/ tests/` clean.
