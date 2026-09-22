# Operations

How to create, run, publish to, recover and maintain a Canary copy.

## CLI reference

All commands accept `--root PATH` (default: `CANARY_ROOT`, else cwd) except pure
library mode.

### `canary init`

```bash
canary init --root /srv/canary [--no-embedding] [--force]
```

Creates:

- `CANARY_ROOT/releases/`, `codebase.lock`, `current -> releases/` (dangling).
- `shared/` with `staging/` (copy of this project, less `.git`, `.venv`, caches,
  `shared/`, `releases/`, `e2e-root/`, `dist/`, `build/`, `*.egg-info`), plus
  `memory/entries/`, `extensions/{archive,tests}/`, `evals/`, `workspace/`,
  `logs/`, and `data/{jobs,sessions,patches/proposed,ratelimit,tmp,memory_index,
  eval_cache,eval_candidates}`.
- `SOUL.md`, `PERSONALITY.md`, `INSTRUCTIONS.md` (built-in templates, only if not
  present).
- `governance.yaml`, `models.yaml` (`main`/`compression`/`health` mock by
  default), `harness.yaml` (full defaults dump).
- `shared/.env` (mode 0600): `HARNESS_API_KEY` (from env or generated) plus live
  model values when `main` is not a mock provider.
- Four seeded eval tasks: `memory-knowledge-update`,
  `memory-conflict-resolution`, `response-form-artifacts`, `write-artifact`.
  (The last one is setup + agent-produced artifact: `setup` writes
  `notes/source.txt`, the prompt makes the agent write `notes/result.txt`,
  and the checks read that file plus the run's own `audit:logs/harness.log`.)
- Git repos: `git init` in `shared/staging/` (commit "bootstrap: staging code
  repo") and `shared/` ("bootstrap: state store"), with a `.gitignore` covering
  `data/ logs/ workspace/ staging/ .env`.
- Best-effort ONNX embedding download into `shared/data/models/` unless
  `--no-embedding`.

Seeded eval tasks are written only when absent, so `canary init` never
overwrites an operator-edited task. A seed fix therefore does not reach an
existing root by itself: to refresh a seed, delete the file and re-run init
(`rm shared/evals/<id>.yaml && canary init --root /srv/canary`), or edit the
installed copy in place. The same no-clobber rule covers `SOUL.md`,
`PERSONALITY.md` and `INSTRUCTIONS.md`; `governance.yaml`, `models.yaml` and
`harness.yaml` are rewritten on every init.

`init` never starts a server and never creates a green tag.

### `canary run`

```bash
canary run "prompt" [--root PATH] [--state-dir PATH] [--ephemeral]
                    [--model ROLE] [--json]
```

One-shot embedded turn, no server. Streams deltas to stdout unless `--json`
(`{response, usage, agent_id, release_id}`).

### `canary serve`

```bash
canary serve [--root PATH] [--host 127.0.0.1] [--port N] [--fd N]
             [--allow-insecure-local] [--canary-mode] [--canary-port P]
```

- Normal: binds (or inherits) the listener and serves. `--fd N` uses an inherited
  socket; systemd socket activation works without `--fd`.
- Canary mode (used by the pipeline): holds the production fd without accepting,
  serves `--canary-port`, and promotes on `SIGUSR1`. `--port` must match the
  parent's port so `/ready` probes and promotion work.

### `canary eval` / `canary diagnose`

Run the held-out set or the diagnosis job. `--tag`, `--limit`, `--json`. These are
also what the `evals_run`/`diagnose_run` tools schedule as jobs.

### `canary revert [RELEASE_ID]`

With a running copy, delegates through `POST /agent/revert` (found via
`--port`, default `port` config). Without a running copy, runs the pipeline
locally with the readiness gate bypassed (logged as such). Reverts to the newest
green release strictly older than the current one, or to an explicit release id.

### `canary status`

With a running copy, prints `/agent/status`. Otherwise a local summary: release
id, commit sha, agent id, green tags, last deploy and whether it was flagged.

### `canary unlock --nonce N`

Force-unlocks `codebase.lock`. The nonce is printed by the lock error message
(`holder` metadata includes `nonce`). Refuses when the nonce does not match or
the holder process is still alive.

## Running as a service

The recommended deployment is systemd with socket activation or an inherited fd;
naive `Type=simple` loses the listener on handover because the promoted child is a
new process tree.

```ini
[Unit]
Description=Canary copy
After=network-online.target

[Service]
Type=notify
WorkingDirectory=/srv/canary
Environment=CANARY_ROOT=/srv/canary
EnvironmentFile=/etc/canary/env            # HARNESS_API_KEY etc., outside CANARY_ROOT
ExecStart=/srv/canary/.venv/bin/python -m canary serve --port 8080
Restart=on-failure
RestartSec=2
KillMode=mixed
TimeoutStopSec=90

[Install]
WantedBy=multi-user.target
```

Socket activation variant:

```ini
# canary.socket
[Socket]
ListenStream=127.0.0.1:8080
Service=canary.service

# canary.service
[Service]
ExecStart=/srv/canary/.venv/bin/python -m canary serve
```

Canary detects `LISTEN_FDS`/`LISTEN_PID` and serves on fd 3. Because the socket
is owned by systemd, handover survives even a full process restart.

## Publishing a change

The agent's `propose(patch, motivation)` tool and `POST /agent/propose` run the
full pipeline synchronously (can take tens of seconds). Under the hood
(`Health._do_publish`):

1. Base gate: the running copy must answer `/ready` 200.
2. Acquire `codebase.lock` (`releases.lock_timeout_s`, default 300s).
3. Recover a previous crashed run from `publish.state.json`, reset dirty staging.
4. Save the patch to `shared/data/patches/proposed/{stamp}.diff`; `git apply`.
5. Static gate in staging: `ruff check .` (30s) and
   `pytest tests/ -x --timeout=60` with a scratch root; integration tests are
   skipped (`-m 'not integration'`). Exit code 5 (no tests collected) is tolerated.
6. Commit the changed paths only, tag `green/{stamp}-{sha7}`.
7. `git archive` into `releases/{release_id}/`.
8. Boot a canary child on a free port (9000–9100), probe `/ready` for up to 30s,
   run a mock-model smoke turn, then the eval gate (which scores the candidate's
   own code — see below).
9. **Red**: kill the child group, free the port, delete the release dir and tag,
   reset staging to the last green, keep serving. **Green**: atomically swap
   `current`, `SIGUSR1` the child, flip `/ready` to 503, drain
   (`releases.drain_timeout_s`), exit.
10. Append the outcome to `data/deploys.jsonl` (`ok`, motivation, `source_session`,
    eval delta, `flagged`, duration).

Patches can be unified diffs or full Git-style diffs; new files are supported.

### Reverting

`POST /agent/revert` or `canary revert`. Reverts are themselves published through
the same canary/probe/promotion pipeline (with the eval gate skipped), so a revert
is as safe as a publish. Reverting twice is a no-op ("already serving this
release").

### Eval gate modes

| `evals.gate` | Behavior |
| --- | --- |
| `off` | Never measure |
| `warn` (default) | Deploy proceeds; `flagged: true` in `deploys.jsonl` and `/agent/status` when the drop exceeds `evals.tolerance` |
| `block` | Candidate is rejected and the old release keeps serving |

Candidates are compared against a cached baseline keyed by
`(release_id, eval_set_hash, model_role)`, valid for `evals.cache_max_age_h`
(168h) and only while the code that measured it is the code that would measure
it now (the entry stores a `code_identity`, a digest of the measuring
`canary/core/evals.py` + `canary/core/agent.py`; an entry from another code
tree, or one written before identities existed, is re-measured). The first run
measures and caches the candidate as the next baseline.

The gate scores the **candidate's** code, not the running copy's: each task runs
in a subprocess with `releases/<id>` first on `PYTHONPATH`, so a release is
measured by the code it would actually serve (and a fix reaches the gate as soon
as it is in the release, without restarting the running copy). If that subprocess
cannot produce a report — for example a release tree without
`canary/core/evals.py` — the gate logs `eval_gate_measure_failed` and reports
`measure_failed: true` with `delta: null` rather than guessing a pass or fail.

## Locks and concurrency

| Situation | What to do |
| --- | --- |
| Publish says "another copy is publishing" | Wait; the holder metadata in the error shows `pid`, `host`, `op`, `started`, `nonce` |
| Holder process is dead | None — `flock` is released by the kernel; retry |
| You must break a live lock | `canary unlock --nonce <nonce>` refuses while the holder is alive; stop the holder first |
| Session answers 409 | Another turn is running and the client used `fail_fast`; retry later |

## Jobs

- Spawn with `job_spawn` (agent) or inspect with `/agent/jobs`.
- Logs are files under `shared/data/jobs/{id}.log`; tail with `job_tail`.
- `job_kill` signals the whole process group.
- Persistent jobs (`persistent=true`) survive restarts as *observed* records:
  running children are reattached, dead ones marked failed. Nothing is resumed
  automatically.
- Limits: 10 spawns per turn, 32 concurrent.

## Extensions

Drop a Python file into `shared/extensions/`:

```python
from canary.core.tools import Tool


class WeatherTool(Tool):
    name = "weather"
    description = "Return the weather for a city."
    parameters = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    requires = []          # env var names that must be set for the tool to load
    profiles = ["network"] # capability tags shown in GET /agent/tools

    def __call__(self, city: str) -> str:
        return f"{city}: sunny, 22C"
```

Behavior:

- New/modified files are picked up within ~1s (watchdog) or by a 1s poll, after a
  `tools.reload_debounce_s` quiet period.
- The previous source is archived to `extensions/archive/{name}_v{N}.py` (keep 10).
- If a reload fails, the old version stays live and calls to that tool return the
  error prefixed with `[tool reload failed: ...]`.
- Declared self-tests in `extensions/tests/{name}_test.py` run in a scratch root
  before the new version is admitted; failing tests keep the old version.
- Modules with same-second/same-size edits are still reloaded correctly (sources
  are compiled, not imported via bytecode cache).

## Memory operations

- Entries are markdown files in `shared/memory/entries/` with YAML frontmatter.
- The agent manages them through `memory_search`/`memory_save`/`memory_forget`/
  `memory_consolidate`; operators can edit files directly (atomic writes are the
  agent's concern; plain editor writes are seen on next read).
- `shared/memory/index.md` is regenerated (≤5KB, most important first).
- When `memory.git` is true, changes are committed to the state repo:
  `git -C shared log -- memory/`.
- Embedding index lives in `shared/data/memory_index/`; swapping the embedding
  model builds the new index before `active.json` flips, so recall never has a
  gap. A rebuild log line (`memory_index_rebuild_pending`) means the active index
  is stale but serving.
- Consolidation only runs when the agent has been idle ≥ 15 minutes and acquires
  the memory lock non-blocking; under contention it defers to a later turn.

## Evals

- Add structured tasks as `shared/evals/{id}.yaml`:

```yaml
id: math-basic
tags: [reasoning, seeded]
prompt: "What is 17 * 23? Reply with just the number."
setup: ""                 # optional bash run in the scratch workspace
timeout_s: 300
model: main               # role override
check:
  - type: contains
    value: "391"
```

- Run: `canary eval --tag reasoning` or `POST /agent/evals/run`.
- Results: `shared/data/evals.jsonl`; rollups: `evals_summary.jsonl`.
- Candidates produced by `canary diagnose` land in
  `shared/data/eval_candidates/`, are never auto-executed, and should be reviewed
  and promoted by moving them into `shared/evals/` (via git).
- Keep eval tasks disjoint from preflight tests: do not import test helpers, do
  not rely on test fixtures.

## Observability

| File | Contents |
| --- | --- |
| `shared/logs/harness.log` | JSONL events (info/warn/error + custom events) |
| `shared/data/metrics.jsonl` | One row per turn: tokens, duration, tool counts |
| `shared/data/health.log` | Probe and drain records |
| `shared/data/deploys.jsonl` | Every publish/revert outcome with motivation, eval delta, flagged |
| `shared/data/evals.jsonl` | Per-task eval results |
| `shared/data/evals_summary.jsonl` | Per-(release, agent) rollups with divergence flag |

Every row carries `timestamp`, `release_id`, `commit_sha`, `agent_id`, so
cross-copy analysis is a `jq` away:

```bash
jq -c 'select(.flagged == true)' shared/data/deploys.jsonl
jq -r '[.release_id, .pass_rate] | @tsv' shared/data/evals_summary.jsonl
```

## Recovery playbook

| Symptom | Action |
| --- | --- |
| Publish failed halfway, staging dirty | Publish again; `_recover` rolls back by `publish.state.json` step |
| `current` points at a non-green release | Restart the copy; boot recovery reverts to the newest green |
| Old child still serving after promotion | Check `ps` for `canary serve`; the parent should exit within `drain_timeout_s` |
| Stuck lock, holder dead | Retry; lock auto-releases. If a live process holds it, `canary unlock --nonce N` |
| Session left `running` after a crash | Next load marks it `interrupted`, `aborted`; it is never replayed |
| Job parent crashed | Children keep running; records observed after restart |
| Embedding model missing | Degrades to tag+recency search and logs `embedding_not_cached`; run `canary init` or set `embedding.download` |
| `/ready` 503 | Read the body's `reason`; usually config errors (missing `base_url`/`context_length`) or missing tools |

## Retention / housekeeping

`releases.keep` (10) prunes old release dirs during publish. Archived sessions
(30d), job logs (7d), proposed patches (100), eval cache (168h) and eval
candidates (30d) have documented retention policies; today GC is performed
opportunistically by the components that own each directory (release GC during
publish, port GC on allocation, session sweep on session access). For long-lived
deployments add cron entries for the log files (`metrics.jsonl`, `health.log`,
`deploys.jsonl`) via logrotate and rotate `evals.jsonl` after `write_summary()`
has produced rollups.
