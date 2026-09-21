# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-18

Initial release. The harness is feature-complete for the v2.3 specification.

### Added

- **Agent loop** with turn limits, cancellation, mid-turn model compaction,
  malformed tool-argument repair, and concurrent session handling (FIFO queue or
  fail-fast).
- **Context engine** with three prompt tiers (stable / context / volatile),
  silent pruning (atomic tool-call units, dependency pinning, spill of oversized
  protected outputs), LLM summarization, and 80/90/95% usage nudges.
- **Memory** as versioned Markdown entries in a separate state repository:
  hybrid retrieval (embeddings, importance, recency), conflict surfacing with
  `supersedes` / `contradicts` / `supports` relations, gated extraction,
  incremental consolidation, and per-model embedding indexes with atomic
  `active.json` swap and reindex.
- **Embeddings** with pluggable backends: local ONNX (Qwen3-Embedding-0.6B
  quantized), OpenAI-compatible endpoints, fastembed, deterministic hash for
  tests, and tag+recency degradation when no backend is available.
- **Tools** with 23 built-ins, extension hot reload (watchdog or polling),
  per-extension self-tests, archives with rollback, operator enable/disable, and
  declared profiles/requirements.
- **Governance** as an advisory allow/deny contract for tool writes, with
  default-deny evaluation and audit through the state repository.
- **Sessions**: persistent history, per-session cross-process locks, inbox
  injection with TTL/cap and delivery receipts, branching, idle archival, and
  interrupted-turn abort (never replay).
- **Jobs**: file-descriptor redirected output (no pipes), process-group kill,
  PID + start-time liveness, per-turn and global concurrency limits, and
  observed-not-resumed recovery across restarts.
- **Evals**: held-out YAML tasks with extensible checks, baseline caching,
  candidate capture, rollup summaries with cross-copy divergence detection, and
  the `diagnose` job for contrastive failure analysis.
- **Release pipeline**: single-writer `codebase.lock` with holder metadata and
  crash resume, static gate (ruff + pytest), immutable release snapshots,
  canary child on a spare port with readiness and mock smoke turn, eval gate
  (off / warn / block), atomic `current` symlink swap, and promotion by
  inherited listener file descriptor with graceful drain.
- **HTTP API**: OpenAI-compatible `/v1/chat/completions` (streaming and not)
  and `/v1/models`, plus admin endpoints for status, sessions, tools, memory,
  jobs, evals, governance, propose, revert, releases, unlock, metrics, and SSE
  event streams. Bearer authentication with an optional loopback exception.
- **CLI**: `canary init`, `run`, `serve`, `eval`, `diagnose`, `revert`,
  `status`, and `unlock`.
- **Embedded mode**: `Agent()` for one-shot and library use with no server and
  no on-disk layout.
- **Documentation**: README plus `docs/` covering architecture, every module,
  configuration, operations, HTTP API, development, and known deviations.
- **Tests**: 119 unit tests and 12 process-level integration tests covering
  publish, locking, revert, fd handover, hot reload, injection, cross-process
  memory, port leasing, and the eval gate.

[0.1.0]: https://github.com/PetrouilFan/canary/releases/tag/v0.1.0
