"""Observability: harness log, metrics, health, deploys, evals JSONL streams.

Every line carries the revision identity (release_id, commit_sha) and agent id
(spec 4.6, 13). Logs are append-only; rotation is external.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from canary.core import util
from canary.core.config import Config


class Log:
    def __init__(self, config: Config):
        self.config = config
        self.data: Path = config.data_path
        self.log_file: Path = config.state_path / "logs" / "harness.log"
        self.agent_id: str = config.get("agent.id", "unknown")

    # -- base ---------------------------------------------------------------

    def _base(self) -> dict:
        return {
            "timestamp": util.utc_now(),
            "release_id": self.config.release_id,
            "commit_sha": self.config.commit_sha,
            "agent_id": self.agent_id,
        }

    def _write(self, path: Path, obj: dict) -> None:
        try:
            util.append_jsonl(path, obj)
        except OSError:
            pass

    # -- harness log ---------------------------------------------------------

    def event(self, event: str, level: str = "info", **fields: Any) -> None:
        line = self._base()
        line["level"] = level
        line["event"] = event
        for key, value in fields.items():
            if isinstance(value, str):
                value = util.redact(util.truncate(value, 4000))
            line[key] = value
        self._write(self.log_file, line)
        if level in ("warning", "error"):
            try:
                payload = json.dumps(fields, default=str)[:500]
                print(f"[canary:{level}] {event} {payload}", file=sys.stderr)
            except OSError:
                pass

    def info(self, event: str, **fields: Any) -> None:
        self.event(event, "info", **fields)

    def warn(self, event: str, **fields: Any) -> None:
        self.event(event, "warning", **fields)

    def error(self, event: str, **fields: Any) -> None:
        self.event(event, "error", **fields)

    # -- JSONL streams -------------------------------------------------------

    def metric(self, **fields: Any) -> None:
        line = self._base()
        line.update(fields)
        self._write(self.data / "metrics.jsonl", line)

    def health_probe(
        self, probe: str, result: str, duration_ms: float, detail: str = ""
    ) -> None:
        line = self._base()
        line.update(
            {
                "probe": probe,
                "result": result,
                "duration_ms": round(duration_ms, 2),
                "detail": detail,
            }
        )
        self._write(self.data / "health.log", line)

    def deploy(self, **fields: Any) -> None:
        line = self._base()
        line.update(fields)
        self._write(self.data / "deploys.jsonl", line)

    def eval_result(self, **fields: Any) -> None:
        line = self._base()
        line.update(fields)
        self._write(self.data / "evals.jsonl", line)

    def eval_summary(self, **fields: Any) -> None:
        line = self._base()
        line.update(fields)
        self._write(self.data / "evals_summary.jsonl", line)

    # -- queries -------------------------------------------------------------

    def tokens_today(self) -> dict:
        day = util.utc_day()
        tokens_in = tokens_out = calls = 0
        for row in util.read_jsonl(self.data / "metrics.jsonl"):
            if str(row.get("timestamp", "")).startswith(day):
                tokens_in += int(row.get("tokens_in") or 0)
                tokens_out += int(row.get("tokens_out") or 0)
                calls += 1
        return {"tokens_in": tokens_in, "tokens_out": tokens_out, "model_calls": calls}

    def recent_deploys(self, n: int = 20) -> list[dict]:
        rows = list(util.read_jsonl(self.data / "deploys.jsonl"))
        return rows[-n:]

    def last_deploy_flagged(self) -> bool:
        rows = self.recent_deploys(1)
        return bool(rows and rows[-1].get("flagged"))
