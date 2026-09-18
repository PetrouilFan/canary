# Architecture

This document describes how Canary is put together: the stores it owns, how code
is released, how processes hand over, which concurrency primitives protect what,
and how a request flows through the system.

## 1. The three stores

Everything Canary owns lives in one directory, `CANARY_ROOT`. Inside it there are
three stores with different lifecycles. Keeping them separate is what makes
rollback safe: rolling code back never rolls memory back, and wiping data never
touches identity.

| Store | Location | Git | Lifecycle |
| --- | --- | --- | --- |
| **Code repo** | `shared/staging/` | yes (`shared/staging/.git`) | one mutable checkout; every publish commits and tags it |
| **State store** | `shared/` (everything except `staging/`) | yes (`shared/.git`, staging is gitignored) | durable, shared by all copies, never rolled back by a code revert |
| **Runtime data** | `shared/data/`, `shared/logs/`, `shared/workspace/` | no (gitignored) | ephemeral by nature: logs, sessions, jobs, caches, temp |

The code repo is copied at `canary init` from the project that contains the
`pyproject.toml`. After that, the only things that touch it are the publish
pipeline and the agent's own `propose` tool.

Releases are built from the code repo with `git archive`, so a release contains
exactly the tree at a committed sha — no `.git`, no build junk.

## 2. Releases and the `current` symlink

```
CANARY_ROOT/
  current -> releases/20260917T212942Z-e0eb64d/
  releases/
    20260917T212942Z-e0eb64d/   # immutable snapshot (the code repo tree at its root)
    20260917T213146Z-a0f0054/
  codebase.lock
  shared/...
```

- Release ids are `{UTC stamp}-{sha7}`, e.g. `20260917T212942Z-e0eb64d`.
  `Health._unique_stamp()` waits for a stamp that no `green/*` tag has used, so
  ids sort chronologically as plain strings.
- Each successful publish creates an annotated tag `green/{release_id}` in the
  staging repo. Tag ordering is **creation time**
  (`git for-each-ref --sort=creatordate`) for legacy tags, but the unique stamp
  is the authoritative order.
- `current` is swapped atomically: a temporary symlink is created next to it and
  `os.replace()`d over the old one (`rename(2)`). `ln -sfn` is never used because
  it unlinks first and leaves a window where the path does not exist.
- `releases.keep` (default 10) controls garbage collection of old release dirs.

The running process always serves from `current`'s target. If `current` is not
the newest green tag at boot (crash during publish), a fresh copy reverts to the
newest green before serving.

## 3. Process model

One copy = one long-lived server process plus whatever children it spawns.

- **Listener owner.** The API socket is created *above* the process (systemd
  socket activation, or by a parent that passes `--fd N`). It is inherited by
  children, never closed, rebound, or opened with `SO_REUSEPORT`. This is what
  makes handover lossless: the accept queue lives in the kernel, not the process.
- **Canary mode.** A release child boots with `--canary-mode --canary-port P
  [--port PROD] [--fd FD]`. It holds the production fd but does **not** accept on
  it. It serves a private canary port so the parent can probe it.
- **Promotion.** The parent sends `SIGUSR1`; the child registers a uvicorn server
  on the inherited production socket and starts accepting. Then the parent flips
  `/ready` to 503, drains for `releases.drain_timeout_s` (default 60s), and exits.
- **Workers.** `task` spawns an in-process `Agent` (depth 1) that shares the state
  store, has its own context window and session, never spawns further workers,
  and reports its final text back to the parent turn. Worker tokens are
  attributed to the parent turn.
- **Jobs.** Long-running shell commands run as detached children in their own
  session (`start_new_session=True`), with stdout/stderr redirected **directly to
  a log file** (never pipes, so a dead parent cannot EPIPE them). Kill uses
  `killpg`, liveness checks use PID + process start time to avoid PID reuse.

## 4. Concurrency primitives

All coordination is local-POSIX and advisory. The kernel releases `flock` locks
automatically when the holder dies, including `SIGKILL`, so there are no stale
lock files to clean up after a crash.

| Lock | Protects | Notes |
| --- | --- | --- |
| `CANARY_ROOT/codebase.lock` | The whole publish pipeline (patch → preflight → canary → swap) | Held for the entire pipeline; holder metadata `{pid, host, op, started, nonce}`; force-unlock requires the nonce and refuses if the holder is alive |
| `shared/memory/.lock` | Memory git commits and consolidation batches | Consolidation acquires it non-blocking per small batch and defers on contention |
| `shared/data/sessions/{id}/.lock` | One session's turn | Second request on the same session queues (FIFO) or gets 409 with `fail_fast` |
| `shared/data/sessions/{id}/.inbox.lock` | Inbox injection/drain | Short critical sections only |
| `shared/data/ratelimit/{provider}.json` (+`.lock`) | Cross-process provider rate limits | Fixed window + concurrency slots, leaked slots GC'd by PID liveness |
| `shared/data/canary_ports.json` (+`.lock`) | Canary port allocation | Leases carry PID + start time; `gc()` frees dead leases; occupied ports are skipped |

Atomic file writes everywhere are `temp file + fsync + os.replace + directory
fsync` (`util.atomic_write_*`). JSONL appends are a single `O_APPEND` write.
Never edit state files in place.

## 5. Data flows

### 5.1 A turn

1. Request arrives (`/agent/run`, `/v1/chat/completions`, or `agent.run()`).
2. `Agent.run` acquires the per-session lock (queue or 409), opens the turn
   (`begin_turn`), appends the user message to history, drains the inbox.
3. `ContextEngine.build` assembles: stable tier (SOUL/PERSONALITY/INSTRUCTIONS +
   deterministic tool list) + context tier (workspace files) + history, then
   prunes if above the compression threshold, then volatile tier (date, memory
   recall, injections, nudge).
4. The model loop runs up to `max_model_calls_per_turn` (64) calls and
   `max_tool_calls_per_turn` (128) tool calls, with cancellation checks between
   each step, a per-call timeout (`model_timeout_s`, 120s) and a whole-turn
   timeout (`turn_timeout_s`, 1800s). Context overflow triggers one forced
   compaction and a retry.
5. Tool calls are validated against governance (for `write`/`edit`), dispatched,
   and their results appended atomically with the call (never split, so pruning
   cannot orphan a result).
6. After the final assistant message: extraction (gated), consolidation when
   idle, metrics, deploy-audit-friendly logging, `end_turn`.

### 5.2 Self-modification (publish)

`propose(patch, motivation)` is the only path that touches code:

1. Base gate: the running copy must be `/ready` 200.
2. Acquire `codebase.lock` (wait up to `releases.lock_timeout_s`, 300s).
3. Recover any crashed publish from `publish.state.json`, reset dirty staging.
4. Write the patch to `shared/data/patches/proposed/{stamp}.diff`, `git apply
   --check`, then apply.
5. Static gate: `ruff check .` and `pytest tests/ -x --timeout=60` in staging
   with a scratch `CANARY_ROOT` (integration tests are excluded via marker).
6. Commit with explicit pathspecs, tag `green/{stamp}-{sha7}`.
7. Build the release with `git archive` into `releases/{id}/`, immutable.
8. Canary: spawn the release child on a spare port (9000–9100), probe `/ready`
   every 2s for up to 30s, run a mock-model smoke turn, run the eval gate.
9. Red: kill the child group, free the port, delete the release dir, delete the
   tag, reset staging to the last green, keep serving. Green: swap `current`,
   `SIGUSR1` the child, flip this process to draining, exit after the drain.
10. `publish.state.json` is written after each step, so a crash resumes or rolls
    back deterministically. Every outcome is appended to `data/deploys.jsonl`
    with motivation, eval delta and a `flagged` bit.

### 5.3 Canary fd handover

Classic zero-downtime handover without `SO_REUSEPORT`:

```
systemd/parent owns listener fd
        │
        ├── parent process accepts (serves /ready 200)
        │
        └── canary child inherits fd, serves only canary port (not accepting prod)
                    │
        probe OK ───┘
                    │
        swap current symlink + SIGUSR1
                    │
        child accepts on inherited fd → parent /ready 503 → drain 60s → parent exits
```

Connections already queued in the kernel accept queue survive, because the socket
itself is never closed. `SO_REUSEPORT` is explicitly avoided: closing one listener
in a reuseport group RSTs connections still in its queue.

### 5.4 Cross-agent messages (injection)

`session_send` (tool) and `POST /agent/sessions/{id}/inject` share one code path:
under the receiver's inbox lock, the message is appended with a TTL
(`sessions.inbox_ttl_h`) and a cap (`sessions.inbox_max`); the sender gets a
receipt (`queued` or `delivered`, plus `previous_read` if the earlier message was
already drained). The receiver drains pending inbox items at the start of its next
turn and they appear as `[system] Message from @sender` in the volatile tier. A
per-sender→receiver rate limit (`sessions.inject_rate`, 5) applies per receiver
turn.

### 5.5 Jobs

`job_spawn` creates a detached child, records `{id, agent_id, session_id, type,
command, status, exit_code, pgid, timestamps, log_path, pid_start_time}`, and
returns immediately. `job_tail`/`job_log` read the file; no pipes exist. On
restart, **persistent** jobs are *observed*: running children keep running and
their records are attached; dead ones are marked failed with a note. Nothing is
ever resumed automatically.

### 5.6 Evals and diagnosis

- Tasks live in `shared/evals/*.yaml` (held out; preflight never reads them, eval
  tasks never import preflight code).
- A run creates one ephemeral `Agent` per task in a scratch workspace and executes
  one turn, then scores `contains/regex/equals/file_exists/file_contains` checks
  (plus extension-provided check types).
- Results append to `data/evals.jsonl` and roll up per `(release, agent)` into
  `data/evals_summary.jsonl` with a cross-copy divergence flag.
- The canary gate compares candidate vs. cached baseline (keyed by release, eval
  set hash and model role; `evals.cache_max_age_h` 168h) and can warn (default,
  marks the deploy `flagged`) or block.
- The `diagnose` job mines failures (failed evals, flagged deploys, user
  corrections, aborted turns, repeated tool errors), contrasts failed vs. succeeded
  trajectories on the compression role and writes candidate eval tasks to
  `data/eval_candidates/`. Candidates are never auto-promoted; the operator moves
  good ones into `shared/evals/` via git.

## 6. Module boundaries

```
canary/
  __init__.py       lazy re-exports (Agent, Config, __version__)
  __main__.py       python -m canary
  cli.py            argparse entrypoint (may import api)
  core/             all harness logic; NEVER imports canary.api (ruff TID251)
  api/server.py     FastAPI app + listener lifecycle
```

Two evolution loops follow from this split:

- **Inner loop** (fast): memory, extensions, identity files, config — hot paths
  with atomic writes, auditable git history, no release needed.
- **Outer loop** (slow): core code changes — patch, preflight, canary, eval gate,
  handover.

## 7. Recovery

| Situation | Behavior |
| --- | --- |
| Crash mid-publish | `publish.state.json` records the step; next publish rolls back staging/release/tag deterministically |
| Lock holder died | `flock` released by the kernel; next acquisition proceeds with no cleanup |
| Stale lock held by a live process | `canary unlock --nonce N` refuses unless the nonce matches and the holder is dead |
| Interrupted turn | Marked `aborted`; replay is never attempted (tool side effects must not run twice) |
| Promoted child cannot serve | Parent keeps the listener and `/ready`; child is killed and the port freed |
| `current` off green at boot | Copy reverts to newest green before serving |
| Job parent died | Child keeps running (fd redirect); persistent records are observed, never resumed |

## 8. Retention

| Artifact | Policy |
| --- | --- |
| Releases | `releases.keep` (10) newest kept |
| Archived sessions | 30 days |
| Job logs | 7 days |
| Proposed patches | 100 newest |
| Eval cache | expires after `evals.cache_max_age_h` (168h), GC daily |
| Eval candidates | discarded after 30 days |
| `evals.jsonl` | rotatable after summaries (weekly rollups kept 12) |
| Metrics/health/deploys | external logrotate |
| Extension archive | `tools.archive_keep` (10) per extension |
