"""test_evals.py - task parsing, check types, scoring, cache, gate, rollups."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from canary.core.evals import Evals, EvalTask, _run_check, eval_set_hash, resolve_check_path
from canary.core.util import read_jsonl
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
