"""Canary command-line interface.

Subcommands:

* ``canary init``      - create a CANARY_ROOT (state repo, staging repo, seeds)
* ``canary run``       - one-shot / embedded turn (no server required)
* ``canary serve``     - run the API server (listener owner, canary mode)
* ``canary eval``      - run the held-out eval set
* ``canary diagnose``  - run the contrastive diagnosis job
* ``canary revert``    - revert the code repo to an earlier green release
* ``canary status``    - show copy / revision / deploy status
* ``canary unlock``    - force-unlock a stale codebase.lock (nonce required)
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from canary import __version__
from canary.core import util
from canary.core.agent import Agent
from canary.core.config import (
    BUILTIN_INSTRUCTIONS,
    BUILTIN_PERSONALITY,
    BUILTIN_SOUL,
    DEFAULTS,
    Config,
    default_governance,
    default_models,
)
from canary.core.evals import Evals
from canary.core.health import Health
from canary.core.observability import Log

_SKIP_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "node_modules",
    "e2e-root",
    "shared",
    "releases",
    "dist",
    "build",
    ".env",
    "current",
    "codebase.lock",
}

_SEEDED_EVAL_TASKS: list[dict[str, Any]] = [
    {
        "id": "memory-knowledge-update",
        "tags": ["memory", "seeded"],
        "prompt": (
            "The atlas deployment port is 8080. Save that fact to memory. "
            "Then update it: a teammate says the port changed to 9090. "
            "Confirm the current value you have stored."
        ),
        "timeout_s": 300,
        "check": [{"type": "contains", "value": "9090"}],
    },
    {
        "id": "memory-conflict-resolution",
        "tags": ["memory", "seeded"],
        "prompt": (
            "Save to memory that service atlas runs on port 8080. "
            "Then a teammate reports atlas now runs on port 9090. "
            "Record both values and say which one you trust and why."
        ),
        "timeout_s": 300,
        "check": [
            {"type": "contains", "value": "8080"},
            {"type": "contains", "value": "9090"},
        ],
    },
]


# -- small helpers -----------------------------------------------------------


def _root(args: argparse.Namespace) -> Path:
    raw = getattr(args, "root", None) or os.environ.get("CANARY_ROOT")
    return Path(raw).expanduser().resolve() if raw else Path.cwd().resolve()


def _write_text(path: Path, text: str, mode: int | None = None) -> None:
    util.ensure_dir(path.parent)
    util.atomic_write_text(path, text)
    if mode is not None:
        os.chmod(path, mode)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *util.GIT_IDENTITY, *args],
        capture_output=True,
        text=True,
    )


def _project_root() -> Path | None:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def _ignore(_dir: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if name in _SKIP_NAMES or name.endswith((".egg-info", ".pyc")):
            ignored.add(name)
    return ignored


def _auth_headers() -> dict[str, str]:
    key = os.environ.get("HARNESS_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}


def _probe_ready(port: int) -> bool:
    try:
        import httpx

        resp = httpx.get(
            f"http://127.0.0.1:{port}/ready", headers=_auth_headers(), timeout=2.0
        )
        return resp.status_code in (200, 503)
    except Exception:
        return False


def _http_json(
    port: int, path: str, *, method: str = "GET", payload: Any = None, timeout: float = 600.0
) -> dict[str, Any]:
    import httpx

    resp = httpx.request(
        method,
        f"http://127.0.0.1:{port}{path}",
        headers=_auth_headers(),
        json=payload,
        timeout=timeout,
    )
    try:
        data = resp.json()
    except Exception:
        data = {"status_code": resp.status_code, "text": resp.text[:1000]}
    if resp.status_code >= 400 and isinstance(data, dict):
        data.setdefault("status_code", resp.status_code)
    return data


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


# -- init --------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve()
    shared = root / "shared"
    if shared.exists() and any(shared.iterdir()) and not args.force:
        print(
            f"error: {shared} already exists and is not empty (use --force to re-init)",
            file=sys.stderr,
        )
        return 1

    util.ensure_dir(root)
    util.ensure_dir(root / "releases")
    (root / "codebase.lock").touch()
    link = root / "current"
    if not link.is_symlink() and not link.exists():
        util.atomic_symlink(link, "releases")
    for sub in (
        "staging",
        "memory/entries",
        "extensions/archive",
        "extensions/tests",
        "evals",
        "workspace",
        "logs",
        "data/jobs",
        "data/sessions",
        "data/patches/proposed",
        "data/ratelimit",
        "data/tmp",
        "data/memory_index",
        "data/eval_cache",
        "data/eval_candidates",
    ):
        util.ensure_dir(shared / sub)

    source = _project_root()
    if source is None:
        print(
            "warning: project root not found; staging was left empty",
            file=sys.stderr,
        )
    else:
        shutil.copytree(source, shared / "staging", ignore=_ignore, dirs_exist_ok=True)

    _write_text(shared / ".gitignore", "data/\nlogs/\nworkspace/\nstaging/\n.env\n")
    _write_text(
        shared / "memory" / "index.md", "# Memory Index\n\n_No entries yet._\n"
    )
    for name, template in (
        ("SOUL.md", BUILTIN_SOUL),
        ("PERSONALITY.md", BUILTIN_PERSONALITY),
        ("INSTRUCTIONS.md", BUILTIN_INSTRUCTIONS),
    ):
        target = shared / name
        if not target.exists():
            _write_text(target, template)
    _write_text(
        shared / "governance.yaml",
        yaml.safe_dump(default_governance(), sort_keys=False),
    )
    _write_text(
        shared / "models.yaml", yaml.safe_dump(default_models(), sort_keys=False)
    )
    _write_text(shared / "harness.yaml", yaml.safe_dump(DEFAULTS, sort_keys=False))

    for task in _SEEDED_EVAL_TASKS:
        path = shared / "evals" / f"{task['id']}.yaml"
        if not path.exists():
            _write_text(path, yaml.safe_dump(task, sort_keys=False))

    cfg = Config(root=root)
    log = Log(cfg)
    env_lines = [f"HARNESS_API_KEY={os.environ.get('HARNESS_API_KEY') or secrets.token_hex(16)}"]
    main = cfg.role("main") or {}
    if main.get("provider") not in (None, "mock"):
        if main.get("base_url"):
            env_lines.append(f"HARNESS_MODEL_BASE_URL={main['base_url']}")
        if main.get("model"):
            env_lines.append(f"HARNESS_MODEL_NAME={main['model']}")
        if main.get("context_length"):
            env_lines.append(f"HARNESS_MODEL_CONTEXT_LENGTH={main['context_length']}")
    key_env = main.get("api_key_env") or "HARNESS_MODEL_API_KEY"
    key_value = os.environ.get(key_env)
    if key_value:
        env_lines.append(f"{key_env}={key_value}")
    _write_text(shared / ".env", "\n".join(env_lines) + "\n", mode=0o600)

    if not (shared / "staging" / ".git").exists():
        _git(shared / "staging", "init", "-q")
        util.git_commit(
            shared / "staging", "bootstrap: staging code repo", ["."], log=log
        )
    if not (shared / ".git").exists():
        _git(shared, "init", "-q")
        util.git_commit(shared, "bootstrap: state store", ["."], log=log)

    if not args.no_embedding:
        try:
            from canary.core import embeddings

            if not embeddings.can_import_onnx():
                print("note: onnxruntime/tokenizers missing; skipping embedding download")
            elif not embeddings.onnx_ready(cfg):
                embeddings.download_onnx_model(cfg, log)
        except Exception as exc:  # noqa: BLE001 - pre-download is best effort
            print(f"warning: embedding pre-download failed: {exc}", file=sys.stderr)

    print(f"canary initialized at {root}")
    print(f"  state store: {shared}")
    print(f"  code repo:   {shared / 'staging'}")
    print(f"  api key:     {env_lines[0].split('=', 1)[1]}")
    print(f"next: canary serve --root {root}")
    return 0


# -- run ---------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    cfg = Config(
        root=args.root, state_path=args.state_dir, ephemeral=args.ephemeral
    )
    agent = Agent(cfg)
    streamed = False

    def emit(event: dict[str, Any]) -> None:
        nonlocal streamed
        if event.get("type") == "delta":
            sys.stdout.write(event.get("content", ""))
            sys.stdout.flush()
            streamed = True

    try:
        text = agent.run(
            args.prompt, model=args.model, on_event=None if args.json else emit
        )
    finally:
        agent.close()

    if args.json:
        _print_json(
            {
                "response": text,
                "usage": agent.last_usage,
                "agent_id": agent.agent_id,
                "release_id": cfg.release_id,
            }
        )
    elif streamed:
        print()
    else:
        print(text)
    return 0


# -- serve -------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace) -> int:
    root = _root(args)
    overrides = {"port": args.port} if args.port else None
    cfg = Config(root=root, overrides=overrides)
    for error in cfg.errors:
        print(f"warning: config error: {error}", file=sys.stderr)
    agent = Agent(cfg)

    from canary.api.server import APIServer, socket_from_fd

    listener = socket_from_fd(args.fd) if args.fd else None
    server = APIServer(
        agent,
        host=args.host,
        port=args.port,
        allow_insecure_local=args.allow_insecure_local,
        listener=listener,
        listener_fd=args.fd,
        canary_mode=args.canary_mode,
        canary_port=args.canary_port,
    )
    try:
        server.serve_forever()
    finally:
        agent.close()
    return 0


# -- eval / diagnose ---------------------------------------------------------


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = Config(root=_root(args))
    log = Log(cfg)
    report = Evals(cfg, log).run(tag=args.tag, limit=args.limit, model_role=args.model)
    if args.json:
        _print_json(report)
        return 0
    print(
        f"run {report['run_id']} release {report['release_id']} "
        f"role {report['model_role']}"
    )
    for row in report.get("results", []):
        status = "PASS" if row.get("pass") else "FAIL"
        detail = "" if row.get("pass") else f" - {row.get('reason') or ''}"
        print(f"{status} {row.get('task_id')} ({row.get('latency_ms', 0)}ms){detail}")
    print(
        f"pass rate: {report['passes']}/{report['total']} "
        f"({report['pass_rate']:.0%}), {report['tokens']} tokens"
    )
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    cfg = Config(root=_root(args))
    agent = Agent(cfg)
    try:
        evals = Evals(cfg, agent.log, registry=agent.tools)
        result = evals.diagnose(
            model_client=agent.models.client_for("compression"),
            sessions=agent.sessions,
            memory=agent.memory,
            limit=args.limit,
        )
    finally:
        agent.close()
    if args.json:
        _print_json(result)
        return 0
    print(
        f"diagnosis: {result.get('candidates', 0)} candidate task(s) "
        f"from {result.get('signals', 0)} signal(s)"
    )
    for path in result.get("paths", []):
        print(f"  {path}")
    if result.get("reason"):
        print(f"  reason: {result['reason']}")
    if result.get("error"):
        print(f"  error: {result['error']}", file=sys.stderr)
    return 0


# -- revert / status / unlock ------------------------------------------------


def cmd_revert(args: argparse.Namespace) -> int:
    root = _root(args)
    cfg = Config(root=root)
    port = args.port or int(cfg.get("port", 8080) or 8080)
    if _probe_ready(port):
        data = _http_json(
            port,
            "/agent/revert",
            method="POST",
            payload={"release_id": args.release_id, "motivation": "cli revert"},
        )
    else:
        log = Log(cfg)
        health = Health(cfg, log)
        health.base_gate_override = (True, "no running copy; local revert")
        data = health.revert(args.release_id, motivation="cli revert")
    if args.json:
        _print_json(data)
    elif data.get("ok"):
        print(f"reverted: {data.get('release_id') or args.release_id or 'latest green'}")
        if data.get("eval") is not None:
            print(f"  eval gate: {data['eval']}")
    else:
        print(f"error: {data.get('error', 'revert failed')}", file=sys.stderr)
    return 0 if data.get("ok") else 1


def cmd_status(args: argparse.Namespace) -> int:
    root = _root(args)
    cfg = Config(root=root)
    port = args.port or int(cfg.get("port", 8080) or 8080)
    if _probe_ready(port):
        data = _http_json(port, "/agent/status")
        data["running"] = True
    else:
        log = Log(cfg)
        data = {"running": False, **Health(cfg, log).status()}
        data["releases"] = Health(cfg, log).list_releases()
        deploys = log.recent_deploys(1)
        data["last_deploy"] = deploys[0] if deploys else None
    if args.json:
        _print_json(data)
        return 0
    print(f"running:    {data.get('running', False)}")
    print(f"release:    {data.get('release_id', cfg.release_id)}")
    print(f"commit:     {data.get('commit_sha', cfg.commit_sha)}")
    print(f"agent id:   {data.get('agent_id', cfg.get('agent.id'))}")
    print(f"ready:      {data.get('ready', '-')} ({data.get('reason', '')})")
    print(f"draining:   {data.get('draining', False)}")
    flagged = data.get("last_deploy_flagged")
    if flagged is not None:
        print(f"flagged:    {flagged}")
    releases = data.get("releases") or {}
    if releases.get("green"):
        print(f"green tags: {', '.join(releases['green'][-3:])}")
    return 0


def cmd_unlock(args: argparse.Namespace) -> int:
    root = _root(args)
    lock = root / "codebase.lock"
    meta = util.read_lock_meta(lock)
    ok, message = util.force_unlock(lock, args.nonce)
    data = {"ok": ok, "message": message, "meta": meta}
    if args.json:
        _print_json(data)
    else:
        print(f"{'unlocked' if ok else 'refused'}: {message}")
        if meta:
            print(f"  holder: {meta}")
    return 0 if ok else 1


def cmd_jobs(args: argparse.Namespace) -> int:
    """``canary jobs progress`` - self-report progress for a live job (spec 4.9).

    Writes the same file a job's own process writes
    (``{state}/data/jobs/{job_id}.progress.json``, atomically). No HTTP, no
    server needed: the harness merges the file while the job is live.
    """
    root = _root(args)
    cfg = Config(root=root)
    record: dict[str, Any] = {
        "job_id": args.job_id,
        "updated_at": util.utc_now(),
        "reporter_pid": os.getpid(),
    }
    if args.percent is not None:
        record["percent"] = max(0.0, min(100.0, float(args.percent)))
    if args.message:
        record["message"] = util.truncate(args.message, 200)
    path = Path(cfg.data_path) / "jobs" / f"{args.job_id}.progress.json"
    util.atomic_write_json(path, record)
    if args.json:
        _print_json({"ok": True, "path": str(path), "progress": record})
    else:
        shown = record.get("percent")
        print(
            f"progress: {args.job_id} "
            f"{'-' if shown is None else f'{shown:g}%'} {record.get('message') or ''}".rstrip()
        )
    return 0


# -- argparse ----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="canary", description="Canary agent harness"
    )
    parser.add_argument("--version", action="version", version=f"canary {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="create a CANARY_ROOT layout")
    p.add_argument("--root", default=None, help="target root (default: cwd)")
    p.add_argument("--no-embedding", action="store_true", help="skip model download")
    p.add_argument("--force", action="store_true", help="re-init a non-empty state store")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("run", help="run one turn without a server")
    p.add_argument("prompt")
    p.add_argument("--root", default=None)
    p.add_argument("--state-dir", default=None)
    p.add_argument("--ephemeral", action="store_true")
    p.add_argument("--model", default=None, help="model role (default: main)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("serve", help="run the API server")
    p.add_argument("--root", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--fd", type=int, default=None, help="inherited listener fd")
    p.add_argument("--allow-insecure-local", action="store_true")
    p.add_argument("--canary-mode", action="store_true")
    p.add_argument("--canary-port", type=int, default=None)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("eval", help="run the held-out eval set")
    p.add_argument("--root", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--model", default=None, help="model role override")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("diagnose", help="run the contrastive diagnosis job")
    p.add_argument("--root", default=None)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("revert", help="revert to an earlier green release")
    p.add_argument("release_id", nargs="?", default=None)
    p.add_argument("--root", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_revert)

    p = sub.add_parser("status", help="show copy and revision status")
    p.add_argument("--root", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("jobs", help="job helpers")
    p.set_defaults(func=None)
    jsub = p.add_subparsers(dest="jobs_command")
    pj = jsub.add_parser("progress", help="self-report progress for a job")
    pj.add_argument("--job-id", required=True, help="job id ($CANARY_JOB_ID inside a job)")
    pj.add_argument("--percent", type=float, default=None, help="0-100 (clamped)")
    pj.add_argument("--message", default=None, help="free-text note (truncated to 200)")
    pj.add_argument("--root", default=None)
    pj.add_argument("--json", action="store_true")
    pj.set_defaults(func=cmd_jobs)

    p = sub.add_parser("unlock", help="force-unlock a stale codebase.lock")
    p.add_argument("--nonce", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_unlock)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
