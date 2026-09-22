"""The publish gate must score the candidate's code, not the supervisor's."""
from __future__ import annotations

import uuid
from pathlib import Path

from canary.core.evals import CHILD_EVAL_MARKER, Evals, EvalTask
from tests.conftest import build_config

STUB = '''
import json, sys
payload = json.load(sys.stdin)
cand = "cand" in payload["release_id"]
task = payload["tasks"][0]
print('@M' + json.dumps({
    "pass_rate": 0.0 if cand else 1.0,
    "passes": 0 if cand else 1, "total": 1, "tokens": 1,
    "run_id": "r-" + payload["release_id"], "release_id": payload["release_id"],
    "model_role": payload.get("model_role"),
    "results": [{ "task_id": task["id"], "pass": not cand, "checks_passed": 0 if cand else 1,
                  "checks_total": 1, "tokens": 1, "latency_ms": 1, "status": "ok",
                  "reason": "", "response": "", "model_role": "main",
                  "agent_id": "stub", "release_id": payload["release_id"] }],
}))
'''


def _task() -> EvalTask:
    return EvalTask.from_dict({
        "id": "probe", "prompt": "say hi", "tags": ["probe"],
        "check": [{"type": "contains", "value": "hi"}],
    })


def test_run_under_returns_the_child_report(tmp_path):
    cfg = build_config(tmp_path)
    from canary.core.observability import Log
    evals = Evals(cfg, Log(cfg))
    code = tmp_path / "cand"
    code.mkdir()
    report = evals.run_under([_task()], code_dir=code, model_role="main",
                             release_id="cand-1", snippet=STUB.replace('@M', CHILD_EVAL_MARKER))
    assert report["release_id"] == "cand-1"
    assert report["results"][0]["task_id"] == "probe"


def test_gate_flags_regression_only_in_candidate_code(tmp_path, monkeypatch):
    import canary.core.evals as evals_mod
    from canary.core.observability import Log
    monkeypatch.setattr(evals_mod, "CHILD_EVAL_SNIPPET", STUB.replace("@M", CHILD_EVAL_MARKER))
    cfg = build_config(tmp_path)
    evals = Evals(cfg, Log(cfg))
    code = tmp_path / "cand"
    code.mkdir()
    rid = uuid.uuid4().hex[:8]
    result = evals.canary_gate(f"cand-{rid}", f"cur-{rid}", tasks=[_task()],
                               candidate_code_dir=code, baseline_code_dir=code,
                               force=True)
    assert result["baseline_pass_rate"] == 1.0, result
    assert result["flagged"] is True
    assert result["delta"] == -1.0
    assert result["candidate_code_dir"] == str(code)


def test_real_snippet_imports_and_reports(tmp_path):
    """The real child entry point must run, not just the test stub.

    An empty task list exercises ``CHILD_EVAL_SNIPPET`` end to end - the child
    imports Config/Evals/EvalTask/Log from ``code_dir``, reads the stdin payload
    and prints the marker - with no model calls and no network.
    """
    import canary
    from canary.core.observability import Log

    cfg = build_config(tmp_path)
    evals = Evals(cfg, Log(cfg))
    code = Path(canary.__file__).resolve().parents[1]
    report = evals.run_under([], code_dir=code, model_role="main",
                             release_id="real-snippet")
    assert report["total"] == 0
    assert report["pass_rate"] == 0.0
    assert report["release_id"] == "real-snippet"
