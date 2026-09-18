# Canary

[![CI](https://github.com/PetrouilFan/canary/actions/workflows/ci.yml/badge.svg)](https://github.com/PetrouilFan/canary/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

Canary is a minimal, modular, self-improving agent harness. It runs as a
single Python process per copy, exposes an OpenAI-compatible HTTP API plus an admin
API, keeps a durable memory store shared across copies, and can modify, test and
publish its own code through an atomic release pipeline.

The design is built around four invariants:

- **No artificial limiters.** The agent can do anything a normal process can do.
  Safety comes from auditability, atomic changes and reversible releases, not from
  hard blocks. The governance file is an advisory workflow contract, not a sandbox.
- **Everything replaceable.** Every component (memory, tools, context engine,
  governance, extensions) is a module with a narrow interface that can be swapped.
- **Shared state, isolated code.** All copies of the same `CANARY_ROOT` share one
  state store (memory, identity, config, extensions) but each release is immutable
  code; publishing swaps a `current` symlink atomically and promotes the running
  process through an inherited listener file descriptor.
- **Green releases.** Code never reaches production without passing preflight
  (ruff + pytest), booting as a canary child on a spare port, answering `/ready`
  and a mock smoke turn, and passing the eval gate.

The design specification is maintained outside this repository; this codebase is a
complete implementation of it. Intentional differences are documented in
[docs/deviations.md](docs/deviations.md).

## Documentation map

| Document | Contents |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | Stores, releases, process model, concurrency, data flows |
| [docs/modules.md](docs/modules.md) | Every module, class and function in `canary/` |
| [docs/configuration.md](docs/configuration.md) | Full config reference, env vars, YAML files, identity files |
| [docs/operations.md](docs/operations.md) | Init, running, publishing, reverts, locks, jobs, extensions, recovery |
| [docs/http-api.md](docs/http-api.md) | HTTP API reference (OpenAI + admin) |
| [docs/development.md](docs/development.md) | Repo layout, tests, dev workflow, how to extend |
| [docs/deviations.md](docs/deviations.md) | Implementation vs. spec differences and rationale |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

## Status

- Implementation: complete; version `0.1.0`.
- Tests: 46 unit tests and 12 process-level integration tests, all passing.
- Platform: Linux and other POSIX systems with `flock(2)` and `rename(2)`.

## Requirements

- Python >= 3.12
- [`uv`](https://github.com/astral-sh/uv) is used for the dev environment
  (but the package is a plain Python package and installs with pip too)
- A POSIX filesystem with working `flock(2)` and `rename(2)` (local disk; NFS is
  not supported). Atomicity and locking guarantees depend on this.

## Install

```bash
uv sync --extra dev          # or: pip install -e ".[dev]"
```

Optional extras:

| Extra | Adds |
| --- | --- |
| `embeddings` | `onnxruntime`, `tokenizers`, `huggingface_hub`, `numpy` (local ONNX embeddings) |
| `fastembed` | `fastembed` (alternative local embedding backend) |
| `watch` | `watchdog` (event-driven extension hot reload; falls back to polling) |
| `dev` | `pytest`, `pytest-timeout`, `ruff` |

## Quick start

### 1. Embedded, no server

Canary works as a library with zero layout on disk:

```python
from canary import Agent

agent = Agent()                    # state in ./.canary/ by default
print(agent.run("Remember that the preferred port is 9090"))
print(agent.run("What is the preferred port?"))
agent.shutdown()
```

Or from the CLI (no server, state in `./.canary/`):

```bash
canary run "Summarize what you know about the project"
canary run --json "hello"          # machine-readable output
canary run --ephemeral "hello"     # throwaway state in a temp dir
```

### 2. A full copy with the release pipeline

```bash
canary init --root /path/to/root   # create the layout, state repo, staging repo
cd /path/to/root
canary serve --port 8080           # serve the API
canary status                      # revision, green tags, last deploy
```

`canary init` does **not** require a running copy and never creates a green tag.
The first green tag appears when `canary serve` passes readiness, or when the first
publish pipeline succeeds. See [docs/operations.md](docs/operations.md).

### 3. Query the API

```bash
curl -s localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $HARNESS_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"main","messages":[{"role":"user","content":"hi"}]}'
```

## CLI

| Command | Purpose |
| --- | --- |
| `canary init [--root PATH] [--no-embedding]` | Create a root layout, seed state and staging repos |
| `canary run PROMPT [--state-dir PATH] [--ephemeral] [--model ROLE] [--json]` | One-shot turn without a server |
| `canary serve [--root PATH] [--port N] [--fd N] [--canary-mode --canary-port P]` | Run the API server (canary mode for release children) |
| `canary eval [--tag T] [--limit N] [--json]` | Run the held-out eval set |
| `canary diagnose [--limit N] [--json]` | Contrastive diagnosis job over recent failures |
| `canary revert [RELEASE_ID]` | Revert to a previous green release |
| `canary status [--json]` | Copy/revision/readiness status |
| `canary unlock --nonce N` | Force-unlock a stale `codebase.lock` |

All commands respect `CANARY_ROOT`; `--root` wins over the environment.

## HTTP API (summary)

Two surfaces, one server:

- **OpenAI-compatible**: `POST /v1/chat/completions` (streaming and not),
  `GET /v1/models`. With `session_id` in the payload the server-side session
  history is authoritative; without it the request is stateless. Client-supplied
  `tools` are rejected; the harness owns tool execution.
- **Admin/agent**: `/health`, `/ready`, `/agent/status`, sessions (list, read,
  cancel, inject, SSE events), tools, memory, jobs, evals, governance, propose,
  revert, releases, unlock, `/agent/smoke`, `/agent/run`.

Full reference with request/response shapes: [docs/http-api.md](docs/http-api.md).

## Tests

```bash
uv run pytest tests/ -q                          # unit suite (integration deselected)
uv run pytest tests/integration -m integration   # process-level integration suite
uv run ruff check canary/ tests/
```

The integration suite spawns real servers and children on ephemeral ports, runs
the real publish pipeline against throwaway roots in `tmp_path`, and verifies fd
handover, locking, revert, hot reload and cross-process memory. It never touches
the network. See [docs/development.md](docs/development.md).

## Repository layout (short)

```
canary/
  core/        # harness internals (never imports canary.api)
    util.py config.py observability.py models.py embeddings.py
    memory.py session.py context.py governance.py tools.py
    jobs.py evals.py health.py agent.py
  api/
    server.py  # FastAPI app, listener handling, drain/promotion
  cli.py       # argparse entrypoint
tests/         # unit + integration suites
docs/          # this documentation
```

`CANARY_ROOT` layout at runtime:

```
CANARY_ROOT/
  current -> releases/<id>/     # atomic symlink to the live release
  releases/<id>/                # immutable code snapshots
  codebase.lock                 # single-writer lock for the publish pipeline
  shared/                       # state store (shared by all copies)
    staging/                    # the single mutable code checkout
    memory/ extensions/ evals/ workspace/ data/ logs/
    SOUL.md PERSONALITY.md INSTRUCTIONS.md
    governance.yaml models.yaml harness.yaml .env
```

## License

Released under the [MIT License](LICENSE).

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
