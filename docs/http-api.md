# HTTP API reference

Base URL is the copy's `--port` (default 8080). All endpoints require
`Authorization: Bearer $HARNESS_API_KEY` unless the client is loopback **and** the
server was started with `--allow-insecure-local`.

Missing/incorrect key → `401`. Unknown resources → `404`. Busy session with
`fail_fast` → `409`. Injection rate limit → `429`. Invalid request → `400`.

## Liveness and readiness

### `GET /health`

Liveness. Always 200 while the process is alive. Body:

```json
{
  "status": "ok",
  "release_id": "20260917T212942Z-e0eb64d",
  "commit_sha": "0552b79a...",
  "agent_id": "host-8080",
  "pid": 12345,
  "timestamp": "2026-09-18T00:00:00Z",
  "listener": "127.0.0.1:8080"
}
```

### `GET /ready`

Readiness. `200` when the copy can serve (config parsed, tools loaded, memory
index opened, main model role present); `503` while draining or on config errors.
Body includes `{ready, reason, draining, release_id, commit_sha, agent_id, since}`.

## OpenAI-compatible

### `GET /v1/models`

Lists roles: `{id, provider, model, context_length}` per role.

### `POST /v1/chat/completions`

Accepts the standard payload plus an optional `session_id`.

- With `session_id`: the server-side session history is authoritative. The
  `messages` you send are treated as the next user input.
- Without `session_id`: the request is stateless; all `messages` form the turn in
  a throwaway session that is deleted afterwards.
- `tools` supplied by the client are rejected with `400` — the harness owns tool
  execution.
- `model` may be a role name (`main`) or a provider model id mapped to a role.
- `stream: true` returns SSE chunks (`chat.completion.chunk`), ending with a
  usage-only chunk and `data: [DONE]`.

```bash
curl -s localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $HARNESS_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "main",
        "session_id": "work",
        "messages": [{"role": "user", "content": "remember: deploy port is 9090"}]
      }'
```

Response (non-streaming) is the standard OpenAI shape plus `session_id`:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1789000000,
  "model": "main",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 120, "completion_tokens": 8, "total_tokens": 128},
  "session_id": "work"
}
```

## Agent

### `POST /agent/run`

Harness-native turn endpoint.

```json
{"message": "hi", "session_id": "work", "model": "main", "stream": false, "fail_fast": false}
```

Non-streaming response: `{response, session_id, usage, release_id, agent_id}`.
Streaming (`stream: true`): SSE with `event: turn_start`, `delta`, `tool`,
`assistant`, `prune`, `compressed`, `turn_end`, plus a final `event: done`
(payload the same as the non-streaming body) or `event: error`.

### `POST /agent/smoke`

Runs one mock/health-role turn in a throwaway session. Returns
`{ok, response, release_id, commit_sha}`. Used by canary probes.

### `GET /agent/status`

`agent.info()` + health status + today's token usage + model roles + uptime and
listener info. Includes `flagged` for the last deploy.

## Sessions

| Endpoint | Purpose |
| --- | --- |
| `GET /agent/sessions` | List sessions (id, name, agent, status, last activity, context usage, worker/parent) |
| `GET /agent/sessions/{id}` | Detail: session info, transcript (`?last_n=`), recent events |
| `POST /agent/sessions/{id}/cancel` | Request cancellation (idempotent) |
| `POST /agent/sessions/{id}/inject` | Queue a message into the session inbox. Body `{message, from_agent?, persist?}`. Shares the code path with `session_send`; returns the delivery receipt (`queued`/`delivered`, `position`, `previous_read`); `429` when rate-limited |
| `GET /agent/sessions/{id}/events` | SSE stream of session events; supports `Last-Event-ID`; `: ping` heartbeats every 15s |

## Tools

| Endpoint | Purpose |
| --- | --- |
| `GET /agent/tools` | Info: built-ins, extensions, disabled, profiles, reload errors; plus full OpenAI schemas |
| `POST /agent/tools/{name}/enable` | Re-enable a manually disabled tool |
| `POST /agent/tools/{name}/disable` | Disable a tool (calls return `error: tool 'x' is disabled`) |

## Memory

| Endpoint | Purpose |
| --- | --- |
| `GET /agent/memory/search?q=...&k=10&tag=a&tag=b` | Search entries; returns hits with scores and components; `[conflict]` marks |
| `POST /agent/memory/save` | Body `{body, tags, entry_id?, importance?, relations?}`; source `api` |

## Jobs

| Endpoint | Purpose |
| --- | --- |
| `GET /agent/jobs?status=&agent_id=` | List jobs |
| `GET /agent/jobs/{id}/status` | Status + alive + last 10 log lines |
| `GET /agent/jobs/{id}/log?offset=&lines=` | Paged log read (`{offset, next_offset, total_lines, eof, lines}`) |
| `POST /agent/jobs/{id}/kill` | Kill the process group |

## Evals

| Endpoint | Purpose |
| --- | --- |
| `POST /agent/evals/run` | Body `{tag?, limit?, model?}`; runs tasks and returns the report |
| `GET /agent/evals/results?limit=100` | Recent `evals.jsonl` rows |
| `GET /agent/evals/tasks?tag=&limit=` | Loaded task definitions |

## Governance and self-modification

| Endpoint | Purpose |
| --- | --- |
| `POST /agent/governance/validate` | Body `{path}` → `{path, allowed, reason, relative candidates}` |
| `POST /agent/propose` | Body `{patch, motivation?, session_id?}`; runs the full publish pipeline; `400` with the failure reason on rejection |
| `POST /agent/revert` | Body `{release_id?}`; reverts to the newest green strictly older than current (or the given id) |
| `GET /agent/releases` | `{current, releases:[{release_id, current, mtime}], green, running_release}` |
| `POST /agent/unlock` | Body `{nonce}`; force-unlock `codebase.lock` (refused if the holder is alive or the nonce is wrong) |

## Metrics

### `GET /agent/metrics?limit=200`

Tail of `shared/data/metrics.jsonl` parsed into JSON rows.

## Error shapes

Admin endpoints return `{"detail": "..."}` on errors (FastAPI default) except
tool-style results, which prefer `{"ok": false, "error": "..."}` or the raw
result object. Publish/revert failures return `200` with
`{"ok": false, "error": ..., "holder": {...}}` when the failure is a lock
contention (the holder metadata tells you who to wait for or unlock).
