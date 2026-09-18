# Development

How to work on Canary itself: environment, tests, conventions, and the parts of
the codebase that bite.

## Environment

```bash
uv sync --extra dev --extra embeddings --extra watch
uv run python -c "import canary; print(canary.__version__)"
```

`uv` keeps the venv, the Python interpreter and all caches inside the project
directory, so nothing is written outside the repo (except model downloads, which
land in `shared/data/models` for initialized roots).

Ruff config lives in `pyproject.toml` (`line-length = 100`, target py312, rules
`E,F,I,UP,B,TID251`). `TID251` bans importing `canary.api` from anywhere except
`canary/cli.py`, `canary/__main__.py` and `canary/api/**` — the core must stay
importable without the API dependencies.

```bash
uv run ruff check canary/ tests/
uv run ruff check --fix canary/ tests/
```

## Repository layout

```
canary/
  core/                harness internals (no api imports)
  api/server.py        FastAPI app, listener lifetime
  cli.py               argparse entrypoint
tests/
  conftest.py          root/cfg/log fixtures, build_config, offline guards
  test_*.py            unit tests (46)
  integration/         process-level tests (12), marked "integration"
docs/                  documentation
e2e-root/              local end-to-end scratch root (gitignored)
```

## Testing

### Unit suite

```bash
uv run pytest tests/ -q
```

- All tests run with `pytest-timeout=120`.
- `tests/conftest.py` isolates every test: `root` fixture moves cwd into a temp
  dir, sets `CANARY_ROOT`, `HARNESS_STATE_PATH` and `HARNESS_EMBEDDING_BACKEND=hash`,
  and clears all model/api-key env vars. Never point a test at the real state
  store.
- `build_config(...)` writes a mock `models.yaml` with per-role scripts, so no
  network call ever happens. `_offline` sets `NO_PROXY=*` as a belt.
- `Log` is required by most components; use the `log` fixture. Passing `None`
  logs quietly but some paths expect it.

### Integration suite

```bash
uv run pytest tests/integration -m integration -q
```

These spawn real servers and children on ephemeral ports and materialize real
roots under `tmp_path`. They are excluded from the staging preflight via
`addopts = "-m 'not integration'"`, so a published release never runs them
against a half-built staging tree.

Coverage: publish e2e (release → probe → swap → promote), codebase lock race,
cross-process memory, extension hot reload, cross-process injection, release
revert, canary port GC, inherited-fd handover and drain, eval gate.

Rules of thumb used throughout:

- Use `run_python(code, env, ...)` for subprocess assertions; assert on parsed
  JSON, not substrings, except for error messages.
- Always kill spawned process groups in a `finally` with `kill_process_group`.
- Take ports with `free_port()`; never hardcode.
- To test publish paths without booting a real child, monkeypatch
  `health.canary_release` (`fake_canary`) and `health._run_static_gate`
  (`ruff_only_gate`).

### E2E with a live model

`./e2e-root/` is a gitignored scratch root for manual end-to-end runs against a
real provider:

```bash
uv run canary init --root e2e-root
cp .env e2e-root/shared/.env        # or rely on the repo .env for provider env
CANARY_ROOT=e2e-root uv run canary serve --port 18090
curl -s localhost:18090/ready | jq
curl -s localhost:18090/v1/chat/completions -H "Authorization: Bearer $HARNESS_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"main","messages":[{"role":"user","content":"hello"}]}' | jq -r '.choices[0].message.content'
```

A live publish is a single `POST /agent/propose` with a patch and a motivation; watch
`e2e-root/shared/data/deploys.jsonl` and the release dirs appear. Remember that
releases are built from the staging snapshot taken at `init`; after changing repo
code, re-init or publish a patch that contains your change.

## Extending the harness

### A built-in tool

1. Add a `Tool` subclass in `canary/core/tools.py` with `name`, `description`,
   `parameters` (JSON schema), optionally `requires`/`profiles`, and
   `__call__(...) -> str`.
2. Append it to `BUILTIN_TOOLS` (order is deterministic; `specs()` sorts anyway).
3. Add it to `EXPECTED_TOOLS` in `tests/test_boot.py`.
4. If it requires harness support that does not exist yet, add the method to
   `Agent` (tools call `self.harness.<method>`).

Rules: never raise from `__call__` — return `error: ...` strings; keep output
sane (the registry clips at 200k); tools must be usable by workers (workers do
not get `task`).

### An extension (no core change)

See [operations.md](operations.md#extensions). Extensions are the supported way
for the agent to add capabilities without touching core code.

### A new eval check type

Subclass `EvalCheck` in a module under `shared/extensions/`; `Evals` discovers
check classes through `ToolRegistry.extension_modules()`.

### A new config key

1. Add it to `DEFAULTS` in `canary/core/config.py`.
2. Document it in [configuration.md](configuration.md).
3. If it has an environment override, add it to `_ENV_MAP`.

## Conventions and gotchas

- **Atomic writes only.** `util.atomic_write_*` for state files; single
  `append_line`/`append_jsonl` for logs.
- **Explicit pathspec commits.** `git_commit` always passes `-- <paths>` so a
  concurrent process's staged files are never swept into a commit.
- **`flock` semantics.** Locks are advisory and released by the kernel on death;
  never write sentinel-file locks with PID checks (PID reuse).
- **No pipes for children.** Jobs redirect file descriptors directly; pipes would
  EPIPE when the parent dies.
- **Exports for prompt caching.** `stable_tier` and tool `specs()` are sorted and
  byte-stable; do not inject timestamps or random order there.
- **Interrupted turns are never replayed.** Tool side effects must not run twice.
- **Cache invalidation.** Embedding indexes are per-model directories with an
  atomic `active.json` swap; never overwrite the active index in place.
- **ONNX single-text embedding.** Quantized ONNX embeddings are not
  batch-composition-invariant; `Memory._embed_each` embeds one text at a time for
  that backend. Do not reintroduce batching for that backend.
- Avoid `pkill -f canary` while developing servers — the pattern matches the
  invoking shell. Track PIDs and use `kill -TERM <pid>` (or
  `util.kill_process_group`).

## Commit conventions

Commits are phased, one logical change per commit, with self-describing messages:

- `scope: imperative summary` (for example `core: memory retrieval and conflict
  surfacing`).
- Keep `ruff check canary/ tests/` and both test suites green before each commit.
- Never rewrite published history; fix forward instead.

## Debugging checklist

| Problem | First look |
| --- | --- |
| `/ready` 503 | Body `reason`; `config.errors`; missing model `base_url`/`context_length` |
| Model calls fail | `shared/logs/harness.log` (`model_call_failed`), rate limit file, key env |
| Publish rejected | `deploys.jsonl` row `error`, patch file under `data/patches/proposed/`, staging `git status` |
| Canary child fails to boot | `shared/data/canary/{port}.log` |
| Extension not loading | `GET /agent/tools` → `reload_errors`; run the declared self-test directly |
| Memory search degraded | `harness.log` `embedding_*` events; `embedding.backend` config |
| Stale lock | `shared/` lock holder metadata; `canary unlock --nonce` |
