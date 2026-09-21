"""The tasks `canary init` seeds must stay loadable and must test real work."""

from canary.cli import _SEEDED_EVAL_TASKS
from canary.core.evals import EvalTask


def test_seeded_tasks_parse_and_have_unique_ids():
    ids = [task["id"] for task in _SEEDED_EVAL_TASKS]
    assert len(ids) == len(set(ids))
    for task in _SEEDED_EVAL_TASKS:
        parsed = EvalTask.from_dict(task)
        assert parsed.id == task["id"]
        assert parsed.check, task["id"]


def test_write_artifact_task_checks_the_agents_own_output():
    task = next(t for t in _SEEDED_EVAL_TASKS if t["id"] == "write-artifact")
    paths = {(c["type"], c.get("path")) for c in task["check"]}
    assert ("file_exists", "notes/result.txt") in paths
    assert ("file_contains", "notes/result.txt") in paths
    assert ("file_contains", "audit:logs/harness.log") in paths
    # setup may create the token source but never the checked artifact
    assert "notes/source.txt" in task["setup"]
    assert "notes/result.txt" not in task["setup"]
