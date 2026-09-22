"""test_evals.py - task parsing, check types, scoring, cache, gate, rollups."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import yaml

from canary.core.evals import (
    Evals,
    EvalTask,
    _run_check,
    code_identity,
    eval_set_hash,
    resolve_check_path,
)
from canary.core.util import read_jsonl, utc_now
from tests.conftest import build_config


def _task_data(task_id: str, **overrides) -> dict:
    data = {
        "id": task_id,
        "prompt": "say the word",
        "tags": ["memory"],
        "check": [{"type": "contains", "value": "OK"}],
    }
    data.update(overrides)
    return data


def _write_task(cfg, name: str, data: dict) -> Path:
    path = cfg.state_path / "evals" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.mark.timeout(60)
def test_task_parsing_and_identity(root: Path, log) -> None:
    cfg = build_config(root)
    evals = Evals(cfg, log)
    path = _write_task(cfg, "t1", _task_data("t1", setup="echo hi", timeout_s=30))
    tasks = evals.load_tasks()
    assert [t.id for t in tasks] == ["t1"]
    task = tasks[0]
    assert task.setup == "echo hi" and task.timeout_s == 30
    assert task.tags == ["memory"]
    assert task.path == path
    assert task.identity()["check"] == task.check

    tagged = evals.load_tasks(tag="nope")
    assert tagged == []
    assert len(evals.load_tasks(limit=1)) == 1

    digest = eval_set_hash(tasks)
    changed = EvalTask.from_dict(_task_data("t1", check=[{"type": "contains", "value": "X"}]))
    assert eval_set_hash([changed]) != digest

    for bad in (
        {"prompt": "x", "check": [{"type": "contains", "value": "x"}]},
        {"id": "x", "check": [{"type": "contains", "value": "x"}]},
        {"id": "x", "prompt": "x"},
    ):
        with pytest.raises(ValueError):
            EvalTask.from_dict(bad)


@pytest.mark.timeout(120)
def test_check_types_and_scoring(root: Path, log) -> None:
    cfg = build_config(
        root,
        scripts={
            "main": ["PORT=9090 content-42"],
            "compression": ["{}"],
        },
    )
    evals = Evals(cfg, log)
    _write_task(
        cfg,
        "contains",
        _task_data("contains", check=[{"type": "contains", "value": "9090"}]),
    )
    _write_task(
        cfg,
        "regex",
        _task_data("regex", check=[{"type": "regex", "value": r"PORT=\d+"}]),
    )
    _write_task(
        cfg,
        "equals",
        _task_data(
            "equals",
            prompt="say exact",
            check=[{"type": "equals", "value": "PORT=9090 content-42"}],
        ),
    )
    _write_task(
        cfg,
        "file_exists",
        _task_data(
            "file_exists",
            prompt="make a file",
            setup='echo "content-42" > artifact.txt',
            check=[{"type": "file_exists", "path": "artifact.txt"}],
        ),
    )
    _write_task(
        cfg,
        "file_contains",
        _task_data(
            "file_contains",
            prompt="make a file",
            setup='echo "content-42" > artifact.txt',
            check=[
                {
                    "type": "file_contains",
                    "path": "artifact.txt",
                    "value": "content-42",
                }
            ],
        ),
    )
    _write_task(
        cfg,
        "failing",
        _task_data("failing", check=[{"type": "contains", "value": "MISSING"}]),
    )
    _write_task(
        cfg,
        "unknown-check",
        _task_data("unknown-check", check=[{"type": "nope", "value": "x"}]),
    )

    report = evals.run()
    assert report["total"] == 7
    assert report["passes"] == 5
    assert report["pass_rate"] == pytest.approx(5 / 7, abs=1e-4)
    by_id = {r["task_id"]: r for r in report["results"]}
    assert by_id["contains"]["pass"] is True
    assert by_id["regex"]["pass"] is True
    assert by_id["equals"]["pass"] is True
    assert by_id["file_exists"]["pass"] is True
    assert by_id["file_contains"]["pass"] is True
    assert by_id["failing"]["pass"] is False
    assert by_id["unknown-check"]["pass"] is False
    assert "unknown check type" in by_id["unknown-check"]["checks"][0]["detail"]

    rows = evals.results(limit=50)
    assert len(rows) == 7
    assert all(row["agent_id"] for row in rows)
    assert all("release_id" in row for row in rows)


@pytest.mark.timeout(120)
def test_cache_and_canary_gate(root: Path, log) -> None:
    cfg = build_config(root, scripts={"main": ["OK"], "compression": ["{}"]})
    evals = Evals(cfg, log)
    _write_task(cfg, "gate", _task_data("gate"))
    tasks = evals.load_tasks()
    assert tasks

    cached = evals.cached_baseline("rel-a", "main", tasks)
    assert cached is None
    first = evals.canary_gate("rel-b", "rel-a", tasks=tasks, role="main")
    assert first["enabled"] is True
    assert first["delta"] == pytest.approx(0.0)
    assert first["flagged"] is False
    assert evals.cached_baseline("rel-a", "main", tasks) is not None
    assert evals.cached_baseline("rel-b", "main", tasks) is not None

    cached_again = evals.cached_baseline("rel-a", "main", tasks)
    assert cached_again is not None
    assert cached_again["pass_rate"] == pytest.approx(1.0)

    cfg.models["models"]["main"]["script"] = ["NOPE"]
    (cfg.state_path / "models.yaml").write_text(
        yaml.safe_dump(cfg.models), encoding="utf-8"
    )
    second = evals.canary_gate("rel-c", "rel-a", tasks=tasks, role="main")
    assert second["flagged"] is True
    assert second["delta"] == pytest.approx(-1.0)

    off = build_config(root / "off", scripts={"main": ["OK"], "compression": ["{}"]})
    off.set("evals.gate", "off")
    off.set("evals.canary_tasks", 0)
    off_evals = Evals(off, log)
    _write_task(off, "gate", _task_data("gate"))
    disabled = off_evals.canary_gate("rel-x", "rel-y")
    assert disabled["enabled"] is False


@pytest.mark.timeout(120)
def test_candidates_and_rollup_summary(root: Path, log) -> None:
    cfg = build_config(root, scripts={"main": ["OK"], "compression": ["{}"]})
    evals = Evals(cfg, log)
    _write_task(cfg, "rollup", _task_data("rollup"))
    evals.run()

    path = evals.write_candidate(
        {"id": "cand-1", "prompt": "p", "check": [{"type": "contains", "value": "x"}]},
        index=0,
    )
    assert path is not None and path.is_file()
    assert path.parent == cfg.data_path / "eval_candidates"
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert loaded["id"] == "cand-1"

    summaries = evals.write_summary()
    assert summaries
    row = summaries[0]
    assert row["runs"] >= 1 and row["pass_rate"] == pytest.approx(1.0)
    assert row["agent_id"]
    file_rows = list(read_jsonl(evals.summary_file))
    assert file_rows and file_rows[-1]["runs"] >= 1


def test_audit_prefixed_paths_read_the_agent_state_dir(
    root: Path, log
) -> None:
    """`audit:` checks see the agent's records; plain paths stay workspace-local."""
    cfg = build_config(root)
    workspace = cfg.state_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    logs = cfg.state_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "harness.log").write_text(
        json.dumps({"event": "tool_call", "tool": "write"}) + "\n", encoding="utf-8"
    )
    spec = {
        "type": "file_contains",
        "path": "audit:logs/harness.log",
        "value": "tool_call",
    }
    result = _run_check(spec, "", workspace, {}, state_path=cfg.state_path)
    assert result["pass"] is True, result
    # the same relative path without the prefix is workspace-local, so it misses
    miss = _run_check(
        {"type": "file_exists", "path": "logs/harness.log"},
        "",
        workspace,
        {},
        state_path=cfg.state_path,
    )
    assert miss["pass"] is False, miss
    assert resolve_check_path("audit:logs/harness.log", workspace, None) == (
        workspace / "logs/harness.log"
    )


@pytest.mark.timeout(60)
def test_run_task_scores_workspace_and_audit_checks(root: Path, log) -> None:
    """End to end: response form + workspace artifact + the agent's own audit log."""
    script = [
        {
            "content": "",
            "tool_calls": [
                {"name": "bash", "arguments": {"command": "cat notes/release.txt"}}
            ],
        },
        "DONE 4af9fa3",
    ]
    cfg = build_config(root, scripts={"main": script})
    task = EvalTask.from_dict(
        _task_data(
            "response-form-artifacts",
            prompt="read notes/release.txt, then answer in the required form",
            setup="mkdir -p notes && printf 'release 4af9fa3\n' > notes/release.txt",
            check=[
                {"type": "regex", "value": r"^DONE \S+\s*$"},
                {
                    "type": "file_contains",
                    "path": "notes/release.txt",
                    "value": "release 4af9fa3",
                },
                {
                    "type": "file_contains",
                    "path": "audit:logs/harness.log",
                    "value": "tool_call",
                },
            ],
        )
    )
    result = Evals(cfg, log).run_task(
        task, run_id="r1", model_role="main", release_id="test"
    )
    assert result["status"] == "ok", (result["reason"], result["checks"])
    assert result["checks_passed"] == result["checks_total"] == 3, result["checks"]

@pytest.mark.timeout(120)
def test_eval_run_keeps_its_records_out_of_the_live_state(root: Path, log) -> None:
    """The eval agent's records land under scratch/state, never in the live root."""
    sentinel = "EVAL-STATE-PROBE-5c1f"
    script = [
        {
            "content": "",
            "tool_calls": [
                {"name": "bash", "arguments": {"command": f"echo {sentinel}"}}
            ],
        },
        "DONE 5c1f",
    ]
    cfg = build_config(root, scripts={"main": script})
    live_log = Path(cfg.state_path) / "logs" / "harness.log"
    live_log.parent.mkdir(parents=True, exist_ok=True)
    live_log.write_text("", encoding="utf-8")
    task = EvalTask.from_dict(
        _task_data(
            "state-override",
            prompt="run the probe, then answer in the required form",
            setup="mkdir -p notes && printf '5c1f\n' > notes/release-5c1f.txt",
            check=[
                {"type": "regex", "value": r"^DONE 5c1f\s*$"},
                {
                    "type": "file_contains",
                    "path": "audit:logs/harness.log",
                    "value": "tool_call",
                },
            ],
        )
    )
    result = Evals(cfg, log).run_task(
        task, run_id="r1", model_role="main", release_id="test"
    )
    assert result["status"] == "ok", (result["reason"], result["checks"])
    assert result["checks_passed"] == result["checks_total"] == 2
    # The audit check reads the state dir the agent actually used, which the run
    # deletes with its scratch dir; the live root's log must not have seen any
    # of this run's tool calls.
    assert sentinel not in live_log.read_text(encoding="utf-8", errors="replace")
    assert live_log.read_bytes() == b""


def _seed_cache_entry(
    evals: Evals, release: str, role: str, tasks: list[EvalTask], pass_rate: float,
    *, identity: str | None,
) -> Path:
    """Write a baseline cache entry by hand, optionally with a code identity."""
    path = evals.cache_path(release, role, tasks)
    payload = {
        "release_id": release,
        "model_role": role,
        "eval_set_hash": eval_set_hash(tasks),
        "cached_at": utc_now(),
        "cached_at_unix": time.time(),
        "pass_rate": pass_rate,
        "results": [
            {"task_id": t.id, "pass": pass_rate >= 1.0, "checks_passed": 1,
             "checks_total": 1, "tokens": 0, "latency_ms": 0}
            for t in tasks
        ],
    }
    if identity is not None:
        payload["code_identity"] = identity
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_code_identity_tracks_the_measuring_tree(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    for base in (a, b):
        (base / "canary" / "core").mkdir(parents=True)
        for name in ("evals.py", "agent.py"):
            (base / "canary" / "core" / name).write_text(f"# {name}\n", encoding="utf-8")
    assert code_identity(a) == code_identity(b)
    assert code_identity(None) == code_identity(None)
    (b / "canary" / "core" / "agent.py").write_text("# changed\n", encoding="utf-8")
    assert code_identity(a) != code_identity(b)
    (b / "canary" / "core" / "evals.py").unlink()
    assert code_identity(b) is None


@pytest.mark.timeout(120)
def test_legacy_baseline_is_remeasured_not_reused(root: Path, log) -> None:
    """A cache entry written before identities existed must not be trusted."""
    cfg = build_config(root, scripts={"main": ["OK"], "compression": ["{}"]})
    evals = Evals(cfg, log)
    _write_task(cfg, "gate", _task_data("gate"))
    tasks = evals.load_tasks()
    path = _seed_cache_entry(evals, "rel-a", "main", tasks, 0.0, identity=None)
    assert evals.cached_baseline("rel-a", "main", tasks) is None

    gate = evals.canary_gate("rel-b", "rel-a", tasks=tasks, role="main")
    # Reusing the 0.0 would have produced a fake delta of +1.0; the baseline is
    # re-measured by this code (1.0), so the gate reports no change.
    assert gate["baseline_pass_rate"] == pytest.approx(1.0)
    assert gate["delta"] == pytest.approx(0.0)
    assert gate["flagged"] is False
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["code_identity"] == code_identity(None)
    assert stored["pass_rate"] == pytest.approx(1.0)


@pytest.mark.timeout(120)
def test_cache_is_reused_by_the_same_code_only(root: Path, tmp_path: Path, log) -> None:
    cfg = build_config(root, scripts={"main": ["OK"], "compression": ["{}"]})
    evals = Evals(cfg, log)
    _write_task(cfg, "gate", _task_data("gate"))
    tasks = evals.load_tasks()

    _seed_cache_entry(evals, "rel-a", "main", tasks, 0.25,
                      identity=code_identity(None))
    hit = evals.cached_baseline("rel-a", "main", tasks)
    assert hit is not None and hit["pass_rate"] == pytest.approx(0.25)
    # ... but a tree that measures differently is not allowed to reuse it.
    other = tmp_path / "other"
    (other / "canary" / "core").mkdir(parents=True)
    for name in ("evals.py", "agent.py"):
        (other / "canary" / "core" / name).write_text("# other\n", encoding="utf-8")
    assert evals.cached_baseline("rel-a", "main", tasks, code_dir=other) is None
    assert evals.cached_baseline("rel-a", "main", tasks, code_dir=tmp_path / "gone") is None
    fresh = evals.baseline("rel-a", "main", tasks, code_dir=other)
    assert fresh["measured"] is True
    stored = json.loads(evals.cache_path("rel-a", "main", tasks).read_text(encoding="utf-8"))
    assert stored["code_identity"] == code_identity(other)
    assert evals.cached_baseline("rel-a", "main", tasks, code_dir=other) is not None
