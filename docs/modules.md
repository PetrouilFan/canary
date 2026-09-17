# Module reference

Every module in `canary/`, what it owns, its public surface and the invariants you
must not break. File paths are relative to the repository root.

---

## `canary/__init__.py`

Lazy re-exports for library users.

- `Config` is imported eagerly from `canary.core.config`.
- `Agent` is exposed through PEP 562 `__getattr__` so `canary.core.agent` is only
  imported when actually used (keeps the import graph acyclic).
- `__version__ = "0.1.0"`.

## `canary/__main__.py`

`python -m canary` → `canary.cli.main()`.

## `canary/cli.py`

Argparse entrypoint (allowed to import `canary.api`). See
[operations.md](operations.md#cli-reference) for behavior.

Helpers:

| Name | Purpose |
| --- | --- |
| `_root(args)` | `--root` > `CANARY_ROOT` > cwd |
| `_write_text(path, text, mode=None)` | mkdir + atomic write + optional chmod |
| `_git(repo, *args)` | `subprocess.run(["git", "-C", repo, ...])` |
| `_auth_headers()` | Bearer header from `HARNESS_API_KEY` |
| `_probe_ready(port, timeout)` | GET `/ready`; accepts 200 or 503 (process alive) |
| `_http_json(method, url, payload)` | httpx request returning parsed JSON |
| `_print_json(obj)` | pretty JSON output |
| `_project_root()` | walks up from `cli.py` for `pyproject.toml` |

Commands: `init`, `run`, `serve`, `eval`, `diagnose`, `revert`, `status`,
`unlock`. `main(argv=None) -> int`.

---

## `canary/core/util.py`

Foundation primitives. No dependencies on other core modules.

**Time/identity**: `utc_now()`, `utc_day()` (`YYYY-MM-DD`), `utc_stamp()`
(zero-padded `YYYYMMDDTHHMMSSZ`), `parse_stamp(text)`, `mono()`, `new_nonce()`,
`hostname()`, `short_sha(text)`.

**Files/dirs**: `ensure_dir(path)`, `atomic_write_bytes/text/json` (temp file in
the same directory + `fsync` + `os.replace` + dir `fsync`), `read_json`,
`append_line` (single `O_APPEND` write), `append_jsonl`, `read_jsonl` (tolerates a
torn last line), `tail_lines(path, n)`, `atomic_symlink(link_path, target)`
(temp symlink + `os.replace`).

**Locking**: `FileLock(path, op, timeout=..., wait=True)` — `flock(LOCK_EX)`
polled every 50ms; after acquiring it writes holder metadata
`{pid, host, op, started, nonce}` next to the lock; `release()` truncates the
metadata before unlocking; `acquire() -> bool` (no exception on contention).
`LockTimeout`, `read_lock_meta(path)`, `force_unlock(path, nonce) -> (ok, message)`
(refuses unless the nonce matches and the holder is dead). `locked(path)`.

**Processes**: `pid_start_time(pid)` (field 22 of `/proc/{pid}/stat`),
`pid_alive(pid, start_time)` (guards against PID reuse), `kill_pid`,
`kill_process_group(pgid, grace=5.0)` (SIGTERM → 5s → SIGKILL),
`port_available(port, host="127.0.0.1")` (bind probe).

**Env/config helpers**: `parse_dotenv(text)`/`load_dotenv(path, override=False)`,
`secret_values()`, `redact(text)`, `env_bool(name, default)`, `dot_get`,
`deep_merge`.

**Globs/strings**: `glob_match(pattern, path)` (own `**` matcher used by
governance and context protection), `slugify`, `is_loopback(host)`,
`parse_size`, `truncate(text, limit, marker)`.

**Git**: `git_commit(repo, message, paths, *, log=None, lock_path=None,
lock_timeout=30.0) -> bool` — `git add -A -- <paths>` then
`git commit -q -m msg -- <paths>`, explicit pathspecs only, bounded retries on
`index.lock`, returns `False` on "nothing to commit". Requires `repo/.git`.

**Invariants**: every state file write goes through an atomic helper; none of
these functions import other core modules (so everything can use them).

---

## `canary/core/config.py`

Configuration resolution and defaults.

- `DEFAULTS` — the full config tree with defaults (see
  [configuration.md](configuration.md)).
- `BUILTIN_SOUL`, `BUILTIN_PERSONALITY`, `BUILTIN_INSTRUCTIONS` — template text
  used by `canary init` and as fallbacks when the identity files are absent.
  `BUILTIN_INSTRUCTIONS` contains the eight default operating rules.
- `default_governance()` / `default_models()` — used by `canary init` and as
  fallbacks.
- `Config(values=None, *, state_path=None, root=None, ephemeral=False,
  overrides=None, load_files=True, workspace=None)`.

Resolution order for the root: constructor `root` > `CANARY_ROOT` > `None`
(embedded). State path: constructor `state_path` > `HARNESS_STATE_PATH` >
`{root}/shared` > `{cwd}/.canary`; `ephemeral=True` uses a temp dir removed at
exit. Files loaded (`load_files=True`): `.env` candidates (state, root, cwd; no
override), `harness.yaml`, then environment overrides, then constructor
overrides, then `models.yaml`, then model env overrides.

Environment overrides (`_ENV_MAP`): `HARNESS_PORT`, `HARNESS_AGENT_ID`,
`HARNESS_SELF_MODIFY`, `HARNESS_MEMORY_GIT`, `HARNESS_EMBEDDING_BACKEND`,
`HARNESS_EMBEDDING_MODEL`, `HARNESS_EMBEDDING_DOWNLOAD`, `HARNESS_EVAL_GATE`.
Model overrides (`_apply_model_env`): `HARNESS_MODEL_NAME`,
`HARNESS_MODEL_BASE_URL`, `HARNESS_MODEL_CONTEXT_LENGTH` overwrite **both**
`main` and `compression` and force `provider: openai`; the key env comes from
`DEFAULT_MODEL_KEY_ENV` unless the role declares `api_key_env`.

Properties: `data_path` (`state_path/data`), `workspace_path`, `base_dir`
(workspace > root > cwd), `serve_mode` (root is not None), `release_id`
(`HARNESS_RELEASE_ID` or `"dev"`), `commit_sha` (`HARNESS_COMMIT_SHA` or
`"unknown"`), `governance_path` (`state_path/governance.yaml`).

Methods: `get(path, default=None)` / `set(path, value)` (dot paths), `role(name)`,
`role_names()`, `to_dict()`, `with_overrides(**kw)`. Validation results are in
`errors` (blocking) and `pre_errors`; `validate_harness()` / `validate_models()`.

`serve_mode and not ephemeral` ⇒ `self_modify` defaults to `True`; embedded mode
defaults to `False`.

---

## `canary/core/observability.py`

Structured JSONL logging with revision identity on every line.

`Log(config)` writes to `{state_path}/logs/harness.log` and creates:
`data/metrics.jsonl`, `data/health.log`, `data/deploys.jsonl`,
`data/evals.jsonl`, `data/evals_summary.jsonl`.

Every line gets `timestamp`, `release_id`, `commit_sha`, `agent_id` from
`_base()`. Methods: `event(event, level="info", **fields)`, `info`, `warn`,
`error` (warn/error also go to stderr), `metric(**fields)`,
`health_probe(probe, result, duration_ms, detail=None)`, `deploy(...)`,
`eval_result(**row)`, `eval_summary(**row)`, `tokens_today()`,
`recent_deploys(n)`, `last_deploy_flagged()`. Secrets are redacted via
`util.redact`; long strings are truncated.

---

## `canary/core/models.py`

OpenAI-compatible model client with retries, rate limits and a deterministic mock.

- `ModelError(retryable, status, retry_after)` and `ContextOverflow`.
- `is_context_overflow(status, body)` — heuristic over provider messages.
- `ToolCall` (`id`, `name`, `arguments` JSON string) with `from_openai`/`to_openai`.
- `ModelResponse` (`content`, `tool_calls`, `usage`, `finish_reason`, `model`,
  `raw`) with `prompt_tokens`/`completion_tokens` and `message()`.
- `RateLimiter(config, log, provider)` — fixed window rpm/tpm plus concurrency
  slots persisted in `data/ratelimit/{provider}.json`; `acquire()`, `record_usage()`,
  `release()`; leaked slots GC'd by PID liveness; raises a retryable `ModelError`
  429 with `retry_after`.
- `MockModel(script)` — deterministic script entries: `str`, `dict`
  (`{content, tool_calls}`), or `ModelResponse`. The last entry repeats. Empty
  script answers `"ok"`. Synthesizes token usage when absent. `stream()` emits
  OpenAI-shaped chunks. Used by tests, the health role, smoke turns and offline
  demo roots.
- `HTTPModel(role, entry, config, log)` — `POST {base_url}/chat/completions`;
  `api_key` read from `entry["api_key_env"]` (error if missing); adds tools and
  `tool_choice: auto`; streaming with `stream_options.include_usage`; retries 3
  attempts with 2/4/8s backoff, honors `Retry-After` capped at 60s, retries 5xx
  and transport errors, surfaces 4xx as non-retryable, converts overflow markers
  to `ContextOverflow`; `stream()` retries only before the first chunk.
- `ModelClient` — facade (`call`, `stream`, `close`, `context_length`) that
  dispatches to mock or HTTP.
- `Models(config, log)` — registry with `client_for(role)`, `context_length(role)`,
  `mock(script)`, `close()`.

Generation parameters (per role in `models.yaml`): `temperature`, `top_p`,
`max_tokens`; defaults come from provider config; the `compression` role defaults
to `temperature: 0.0`.

---

## `canary/core/embeddings.py`

Pluggable embedding backends with graceful degradation.

- `EmbeddingUnavailable` — raised when no backend can run.
- Backends: `HashBackend` (deterministic 64-dim, for tests), `OnnxBackend`,
  `OpenAIBackend`, `FastEmbedBackend`, `UnavailableBackend`.
- `resolve_backend(config, log, allow_download)` honors `embedding.backend`
  (`auto|onnx|openai|fastembed|hash|none`), preferring a cached ONNX model,
  downloading only when `embedding.download` or `allow_download` is set; logs
  `embedding_not_cached`, `embedding_download_failed`, `embedding_degraded`.
- `onnx_repo_for(model)` → `onnx-community/{basename}-ONNX`; `model_slug(model)`;
  `download_onnx_model(config, log, force=False)` via `huggingface_hub`;
  `onnx_ready(config)`; `can_import_onnx()`.
- `cosine(a, b)` helper.

`OnnxBackend` details: CPU-only ONNX Runtime, tokenizer from `tokenizer.json`
(truncation + padding), last-token pooling by attention sum, L2 normalization.
Exports with `past_key_values` inputs are supported by feeding empty KV arrays
and `position_ids`. **Embeddings are computed one text per forward pass on ONNX**
because quantized batches are not composition-invariant; the `_embed_each` helper
in `memory.py` enforces this.

---

## `canary/core/memory.py`

Durable agent memory: markdown entries with YAML frontmatter, an embedding index,
retrieval with conflict surfacing, gated extraction and incremental consolidation.

Layout under `{state_path}/memory/`: `entries/{id}.md`, `index.md` (≤5KB, by
importance/updated selection), `.lock`. Index data in
`{data}/memory_index/{model_slug}/` with `meta.json` + `vectors.dat`, and
`{data}/memory_index/active.json` naming the active model.

- `Entry` — `id`, body, `tags`, `created`, `updated`, `source`, `importance`,
  `relations` (`supersedes`/`contradicts`/`supports`), `archived`; markdown
  frontmatter round-trip (`to_markdown`/`from_markdown`).
- `SearchHit` — entry, score, component breakdown; `format_hits` renders
  `- [conflict] {id} [tags]: text`.
- `EmbeddingIndex` — per-model dir, `update(entries, force=False)`, `remove(ids)`,
  `scores(query)`; tolerant loading.
- `Memory(config, log, backend=None, model_client=None, allow_download=None)`:
  - `save(body, *, tags, entry_id=None, source="", importance=0.5, relations=None)`
    — atomic write, index update, `index.md` refresh, git commit under the memory
    lock when `memory.git` is on (message `state: memory {ts} {tags}`).
  - `search(query, *, k=None, tags=None, include_archived=False)` — score =
    `0.6·similarity + 0.3·tag/lexical + 0.1·recency`, modulated by importance;
    `_add_conflicts` marks `[conflict]` and appends up to two related entries.
  - `get(id)`, `all_entries(include_archived=False)`, `forget(id)` (archives,
    never deletes), `write_index()`.
  - `extract(conversation, *, max_entries=None, source="", model_client=None)` —
    gated extraction on the compression role, dedupes at
    `memory.dedupe_similarity`, rejects secrets.
  - `consolidate(*, model_client=None, idle_after_s=None)` — requires idle time
    (`memory.consolidate_idle_min`), acquires `.lock` non-blocking per bounded
    batch and defers on contention; merges near-duplicates (keep higher
    importance, archive the other, add a `supersedes` relation); embeddings are
    computed outside the lock; never deletes content.
- Model swap: the old model's index stays active until the new one is fully
  built (`active.json` is swapped atomically), so retrieval never has a gap.

---

## `canary/core/session.py`

Sessions, history, inbox, events, branching and cross-process coordination.

Layout: `{data}/sessions/{id}/` with `state.json`, `history.jsonl`,
`inbox.jsonl`, `inbox_state.json`, `events.jsonl`, `.lock`, `.inbox.lock`;
archived sessions move to `sessions/archived/{id}/`.

- `Message` — role, content, id, ts, tool_calls, tool_call_id, name, from_agent,
  persist; `to_json`/`from_json`/`to_openai`.
- `Session` — `save_state`, properties (`name`, `agent_id`, `visible_to`,
  `parent_session`, `branch_point`, `is_worker`, `last_activity`, `status`),
  `lock(op, wait=True, timeout=0)` (FileLock), `append_message`,
  `history(last_n)`, `transcript(last_n, include_tools)`, `begin_turn()` /
  `end_turn(status)`, `take_interrupted()` / `recover()`, `set_context_usage()`,
  `fork(branch_point, name, agent_id)`, `inject(item)`, `pending_inbox()`,
  `drain_inbox()`, `append_event(type, **data)`, `events(last_n)`,
  `had_recent_read(receiver_id)`, `info()`, `touch()`, cancel flags
  (`cancel_flag` path, `request_cancel()`, `cancel_requested()`, `clear_cancel()`).
- `SessionManager(config, log)` — `create(...)`, `load(id)` (falls back to
  archived; recovers interrupted turns), `get_or_create(id)`, `branch(...)`,
  `list(visible_to=None, include_archived=False)`, `inject_message(session_id,
  message, *, from_agent, from_session=None, persist=False) -> dict`,
  `append_event`, `sweep()` (archives sessions idle beyond
  `sessions.idle_archive_h`, drains inbox first, skips open turns).

Turns are serialized per session by the cross-process `.lock`; interrupted turns
are marked aborted and are **never replayed**.

---

## `canary/core/context.py`

Context assembly, pruning, spill, compression and nudges.

- `PROTECTED_PATTERNS = (task, memory_*, write, edit, propose, revert, compress,
  evals_run)`; `TOOL_WEIGHTS` biases eviction scoring.
- `Unit` — atomic group: an assistant message with tool calls plus its matching
  tool results. Pruning operates on whole units, so a call and its result are
  never split. `build_units`, `units_to_messages`, `is_protected`.
- `PruneReport` (tokens before/after, pruned units, spilled files, released pins)
  and `CompressReport` (tokens, retained tools, trace path).
- `ContextEngine(config, log, memory=None, model_client=None)`:
  - `stable_tier(tools)` — SOUL/PERSONALITY/INSTRUCTIONS + deterministic tool
    listing (byte-identical across turns for prompt caching).
  - `context_tier()` — workspace files (`context.workspace_files`), cached by file
    signature for `context.cache_s`.
  - `volatile(...)` — date (day granularity), memory recall, `[context summary]`,
    injections, interrupted marker, nudge.
  - `prune(messages, *, session=None, threshold=None, user_message="")` — silent,
    never calls a model; spills oversized protected outputs to `data/tmp` (>32KB),
    pins units referenced by retained protected outputs (`pruning.pin_max`, 50;
    released pins are annotated `[dependency pruned: ref]`), protects the last
    `pruning.recent_turns` (3) user turns, evicts lowest-scored first, inserts a
    `[system] Pruned N older units…` note.
  - `summarize(messages, *, focus, model_client, session, save_memory=True)` —
    compression-role summary, trace in `data/tmp/summary_{stamp}.txt`, saved as a
    `context-summary` memory entry.
  - `compress(...)` — explicit compaction: keep head system + `[context summary]` +
    protected older units + recent window.
  - `nudge_for(ratio)` — `[system]` text at 80/90/95% of usable context.
  - `build(session, *, user_message="", tools=None, injections=None, summary="",
    max_history=None)` — full message list for the next model call.

Usable context is `0.9 × context_length(main)`; token counting is
`len(json.dumps(messages)) // 4`.

---

## `canary/core/governance.py`

Advisory write guardrails. **Not a sandbox**: `bash` bypasses it by design, and
the agent may edit `governance.yaml` itself. Its value is a documented default
and an audit trail.

- `Governance(config, log)` loads `{state_path}/governance.yaml`, falling back to
  `default_governance()` and recording `error`.
- `check(path) -> (allowed, reason)` — default deny, deny wins; patterns render
  against both `CANARY_ROOT`-relative and state-store-relative paths (so both
  `shared/memory/**` and `core/**` style patterns work); a bare pattern also
  covers its children.
- `allowed(path)`, `save(allow, deny)`, `validate()`, `as_dict()`.

Default `allow_write`: `shared/memory/**`, `shared/extensions/**`,
`shared/workspace/**`, `shared/data/**`, `shared/SOUL.md`,
`shared/PERSONALITY.md`, `shared/INSTRUCTIONS.md`, `shared/.env`,
`shared/governance.yaml`, `shared/models.yaml`, `shared/harness.yaml`.
Default `deny_write`: `shared/staging/**`, `shared/evals/**`, `core/**`, `api/**`,
`tests/**`, `pyproject.toml`, `releases/**`.

---

## `canary/core/tools.py`

Tool base class, the 23 built-ins, the registry, and extension hot reload.

- `Tool` — class attributes `name`, `description`, `parameters` (JSON schema),
  `requires` (env vars), `profiles`; `__init__(harness)`;
  `__call__(*args, **kwargs) -> str` (always a string; errors are returned, never
  raised); helpers `config`, `log`, `resolve_path`, `missing_requires`, `spec()`.
- Built-ins (23): `read`, `write`, `edit`, `bash`, `propose`, `revert`,
  `compress`, `memory_search`, `memory_save`, `memory_forget`,
  `memory_consolidate`, `sessions_list`, `session_read`, `session_send`, `task`,
  `evals_run`, `diagnose_run`, `job_spawn`, `job_list`, `job_status`, `job_tail`,
  `job_kill`, `job_log`.
  - `write`/`edit` consult governance and log `write_denied` events.
  - `bash` is ungoverned, uses `tools.bash_timeout_s` (120) and suggests
    `job_spawn` on timeout.
  - `propose(patch, motivation=None)` and `revert(release_id=None)` submit to the
    publish pipeline and report the result.
  - `task` wraps `Agent.spawn_worker`; `evals_run`/`diagnose_run` schedule jobs
    (never run inline).
  - Output is clipped at 200k chars (`read` at 100k).
- `ToolRegistry(harness, load_extensions=True)` — `register`, `unregister`, `get`,
  `names`, `all`, `specs(only=None)` (sorted for prompt stability),
  `call(name, args)` (never raises; unknown/disabled/TypeError become error
  strings; a failed extension reload prefixes success text with
  `[tool reload failed: …]`), `enable`/`disable`/`is_enabled`, `available()`,
  `disabled()`, `profiles()`, `restrict(names)`, `info()`, `extension_modules()`.
- Extensions live in `{state_path}/extensions/*.py`; a module defines one or more
  `Tool` subclasses; a module-level `tool = ToolClass(self)` is honored. Sources
  are `compile()`d and `exec()`d into a fresh module object (importlib bytecode
  caching served stale code for same-second edits).
- Reload: watchdog observer when available, otherwise a 1s polling thread; changes
  are debounced by `tools.reload_debounce_s`; failures keep the old tool live and
  record `reload_errors`. Old sources are archived to `extensions/archive/`
  (keep `tools.archive_keep`, 10); `rollback(name)` restores the newest archive.
- Declared self-tests in `extensions/tests/{name}_test.py` run in a scratch
  subprocess with `CANARY_ROOT`/`HARNESS_STATE_PATH` pointing at a temp dir before
  a new version is admitted.

---

## `canary/core/jobs.py`

Detached long-running commands.

`Job` fields: `id`, `agent_id`, `session_id`, `type`, `name`, `command`, `status`
(`pending|running|done|failed|killed`), `exit_code`, `pgid`, timestamps,
`log_path`, `pid_start_time`, `persistent`, `note`, `timeout_s`.

`Jobs(config, log)`:

- `spawn(command, *, shell=None, persistent=False, session_id=None, type="shell",
  name=None, cwd=None, timeout_s=None, env=None)` → `{job_id, status, log_path,
  pid}`; raises `JobLimitError` beyond `jobs.max_per_turn` (10) or
  `jobs.max_concurrent` (32).
- Child environment always includes `CANARY_ROOT`, `HARNESS_STATE_PATH`,
  `HARNESS_RELEASE_ID`, `HARNESS_COMMIT_SHA`.
- stdout/stderr are opened directly on `data/jobs/{id}.log` (no pipes),
  `start_new_session=True`; a monitor thread finalizes status, applying
  `timeout_s` via `killpg`.
- `status(job_id)` (+ alive flag and 10 log lines), `tail(job_id, n)`,
  `log(job_id, offset, lines)` (paged), `list(status, agent_id)`, `kill(job_id)`,
  `running()`, `wait(job_id, timeout_s)`, `info()`, `get(id)`.
- `begin_turn()` resets the per-turn counter; `recover()` attaches persistent
  records for this `agent.id` only: alive → `running` with a watcher thread,
  dead → `failed` with note "process gone; observed after restart".

---

## `canary/core/evals.py`

Held-out evaluation tasks, the canary gate and the diagnosis job.

- `EvalCheck` base (`type`, `check(value, response, workspace) -> (bool, str)`) —
  extensions can add check types.
- `EvalTask` — `id`, `prompt`, `tags`, `setup` (bash in scratch workspace),
  `timeout_s` (300), `model` (role override), `check` list, `path`;
  `from_dict`/`from_file`/`to_dict`/`identity`.
- Built-in check types: `contains`, `regex`, `equals`, `file_exists`,
  `file_contains`.
- `eval_set_hash(tasks)` — sha256[:12] over task identities, used in cache keys.
- `Evals(config, log, registry=None)`:
  - `load_tasks(tag=None, limit=None, path=None)` — sorted `shared/evals/*.yaml`;
    invalid files logged and skipped.
  - `run(tasks=None, *, tag, limit, model_role=None, release_id=None, run_id=None)`
    → report `{run_id, release_id, model_role, total, passes, pass_rate, tokens,
    results}`; each result appended to `data/evals.jsonl` with `agent_id`.
  - `run_task(...)` — expects `Agent` built with `ephemeral=True`,
    `self_modify=False`, scratch workspace; enforces `timeout_s` with a daemon
    thread + `agent.cancel()`.
  - `cache_path`, `cached_baseline`, `store_baseline`, `baseline`,
    `canary_gate(candidate, baseline, *, tasks, role, force)` → `{enabled, delta,
    flagged, pass_rate, baseline_pass_rate, tolerance, tasks, run_id}`. Cache key
    is `{release}.{eval_set_hash}.{role}.json`; expires after
    `evals.cache_max_age_h` (168). Gate modes: `off`, `warn` (default; deploy
    proceeds, marked `flagged` when delta < −tolerance), `block`.
  - `results(limit)`, `write_summary()` (per `(release, agent)` rollup with
    cross-copy divergence flag), `write_candidate(data, index)`,
    `diagnose(*, model_client, sessions, memory=None, limit=20)` — mines
    interrupted/aborted turns, tool errors, user corrections, failed evals and
    flagged deploys; writes ≤ `diagnosis.max_candidates` (10) candidate YAML files
    and `diagnosis`-tagged memory entries.

Disjointness rule: preflight never reads `shared/evals/`, eval tasks never import
preflight tests. Enforced by convention and the staging `.gitignore`.

---

## `canary/core/health.py`

Liveness/readiness, the publish pipeline, releases and canary handover.

- `classify(paths)` → `memory|extension|config|code`; decides which preflight tier
  a change needs.
- `Ports(data_dir, log, low, high)` — canary port leases in
  `data/canary_ports.json` (+ `.lock`): `allocate`, `update`, `free`, `in_use`,
  `gc()` (frees leases whose PID is dead, skips occupied ports).
- `Health(config, log, agent=None, *, listener_fd=None, port=None,
  drain_callback=None)`:
  - readiness: `set_ready`, `set_draining`, `check_ready()`, `ready_response()`,
    `health_response()` (both include release id and commit sha).
  - `base_gate()`, `publish(patch, *, motivation, session_id)`,
    `revert(release_id=None, ...)`, `apply_config(paths, ...)`,
    `canary_release(release_dir, *, label, eval_gate)`.
  - `probe_child(port, timeout, api_key)` — GET `/ready` then POST `/agent/smoke`,
    both with the child's API key.
  - `unlock(nonce)`, `status()`, `list_releases()`, `green_tags()` (creation
    order), `last_green_sha()`, `current_release()`, `staging_clean()`.
  - `_do_publish` steps: patch file under `data/patches/proposed/`, `git apply`,
    static gate (ruff + pytest with scratch root), commit explicit paths, annotate
    green tag, `git archive` release, canary probe + eval gate, atomic `current`
    swap, `SIGUSR1`, async drain thread, deploy log with eval delta and motivation.
  - `_do_revert` picks the newest green strictly older than the current release
    (or the explicit release id; materializes the dir from its tag when needed).
  - `publish.state.json` makes crashes resumable/rollback-able step by step.
  - `_release_sha` stamps the child env with the release tag's commit.

Drain is performed on a daemon thread (2s delay) so publishing from inside the
serving process does not deadlock the in-flight request.

---

## `canary/core/agent.py`

The harness object. Ties everything together and runs turns.

`Agent(config: Config | dict | None = None, *, state_path=None, ephemeral=False,
self_modify=None, workspace=None, is_worker=False, parent=None, model=None,
tools=None)`.

- In serve mode (`root` set, not ephemeral) it creates `Log`, `Models`,
  `Governance`, `Memory`, `SessionManager`, `Jobs` (+`recover()`), `Health`;
  embedded mode creates the same minus `Health`.
- Workers pass `parent=self` and share the parent's stores; `task` is not
  available to workers, and depth is capped at 1.
- `run(message, *, session_id=None, model=None, on_event=None, fail_fast=False)`
  → final assistant text. Without `session_id` the turn is stateless in a
  throwaway session. `fail_fast=True` raises `SessionBusy` instead of queueing.
- Turn loop: session lock → `begin_turn` → user message → inbox drain →
  `ContextEngine.build` → model calls (with per-call and whole-turn timeouts,
  cancellation checks, silent pruning and nudges) → tool dispatch (atomic
  call+result, per-tool failure tracking, malformed JSON args repaired once) →
  final text → extraction/consolidation → metrics and events.
- Context overflow triggers one forced compaction and a retry.
- `on_event` receives `turn_start`, `delta`, `assistant`, `tool`, `prune`,
  `compressed`, `turn_end` events (used for SSE).
- `spawn_worker(prompt, name=None, tools=None, model=None, timeout_s=900)` runs
  the worker in a thread, enforces the timeout with cancellation, and returns its
  final text (errors come back as `error: …` strings).
- Harness surface used by tools: `propose`, `revert`, `compress_current`,
  `schedule_eval_job`, `schedule_diagnose_job`, `session_id`, `agent_id`,
  `tool_specs()`, `last_usage`, `last_total_tokens`, `total_tokens`.
- `cancel(session_id=None)`, `close()`, `shutdown()` (drain + close), context
  manager support.

---

## `canary/api/server.py`

FastAPI app, uvicorn lifecycle, listener inheritance, promotion and drain.

- `make_listener(host, port, backlog=2048)` — binds and listens; sets
  `SO_REUSEADDR`, marks the fd inheritable; never `SO_REUSEPORT`.
- `listener_from_env()` — systemd socket activation (`LISTEN_FDS`, `LISTEN_PID`).
- `socket_from_fd(fd)`.
- `_DrainServer(uvicorn.Server)` — first SIGTERM/SIGINT triggers
  `on_exit_request()` then graceful shutdown; second forces exit.
- `_TurnStream` — bridges a background turn to an async SSE queue.
- `APIServer(agent, *, host="127.0.0.1", port=None, allow_insecure_local=False,
  listener=None, listener_fd=None, canary_mode=False, canary_port=None)`.
  - Normal mode: listener = provided > systemd > freshly bound; `/ready` set true;
    serves until SIGTERM; drain flips `/ready` to 503, stops accepting, waits
    `releases.drain_timeout_s`.
  - Canary mode: builds the production socket from the inherited fd, registers
    `SIGUSR1` for promotion, serves only the canary port, and syncs `Health`
    (`listener_fd`, `port`, `drain_callback`) so a promoted child can itself
    publish/revert.
  - Auth: `HARNESS_API_KEY` Bearer token; loopback clients may skip auth only with
    `--allow-insecure-local`; constant-time comparison.
  - Endpoints: see [http-api.md](http-api.md).

---

## `canary/api/__init__.py`

Package marker; intentionally empty so importing it has no side effects.
