"""Fixtures and helpers for integration tests (spec §15).

Integration tests spawn processes and servers. They are excluded from the
staging preflight gate via the ``integration`` marker (see pyproject.toml)
and must be run explicitly: ``pytest tests/integration -m integration``.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from canary.cli import main as cli_main

REPO_ROOT = Path(__file__).resolve().parents[2]

_ENV_CLEAR = (
    "HARNESS_MODEL_NAME",
    "HARNESS_MODEL_BASE_URL",
    "HARNESS_MODEL_CONTEXT_LENGTH",
    "HARNESS_MODEL_API_KEY",
    "HARNESS_API_KEY",
    "HARNESS_EMBEDDING_BACKEND",
    "HARNESS_EMBEDDING_MODEL",
    "HARNESS_SELF_MODIFY",
    "HARNESS_MEMORY_GIT",
    "HARNESS_EVAL_GATE",
    "HARNESS_RELEASE_ID",
    "HARNESS_COMMIT_SHA",
    "HARNESS_CANARY_PORT",
    "CANARY_RELEASE_DIR",
    "PYTHONPATH",
    "LISTEN_FDS",
    "LISTEN_PID",
)


def init_root(root: Path, *, api_key: str = "integration-key") -> None:
    """Bootstrap a copy root with mock models via the real CLI."""
    rc = cli_main(["init", "--root", str(root), "--no-embedding"])
    assert rc == 0
    assert (root / "shared" / "harness.yaml").is_file()


def clean_env(root: Path, state: Path | None = None) -> dict[str, str]:
    env = dict(os.environ)
    for name in _ENV_CLEAR:
        env.pop(name, None)
    env["CANARY_ROOT"] = str(root)
    env["HARNESS_STATE_PATH"] = str(state or (root / "shared"))
    env["HARNESS_EMBEDDING_BACKEND"] = "hash"
    env["HARNESS_API_KEY"] = "integration-key"
    env["PYTHONPATH"] = str(REPO_ROOT)
    return env


def run_python(
    code: str,
    env: dict[str, str],
    *args: str,
    timeout: int = 120,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
    )


def http_get(url: str, *, timeout: float = 2.0, headers: dict | None = None):
    import httpx

    return httpx.get(url, timeout=timeout, headers=headers or {})


def wait_ready(port: int, timeout: float = 30.0, api_key: str = "integration-key") -> bool:
    deadline = time.monotonic() + timeout
    headers = {"Authorization": f"Bearer {api_key}"}
    while time.monotonic() < deadline:
        try:
            response = http_get(f"http://127.0.0.1:{port}/ready", headers=headers, timeout=1.0)
            if response.status_code == 200:
                return True
        except Exception:  # noqa: BLE001 - poll loop
            pass
        time.sleep(0.2)
    return False


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def listener(port: int) -> socket.socket:
    from canary.api.server import make_listener

    return make_listener("127.0.0.1", port)


def spawn_serve(
    root: Path,
    *extra: str,
    env: dict[str, str],
    pass_fds: tuple[int, ...] = (),
) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "canary", "serve", "--root", str(root), *extra]
    return subprocess.Popen(
        cmd,
        env=env,
        cwd=str(REPO_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        pass_fds=pass_fds,
    )


@pytest.fixture()
def initialized(root: Path) -> Path:
    init_root(root)
    return root


def ruff_only_gate(health, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the staging gate with ruff only (avoids nested pytest runs)."""

    def ruff_only() -> str:
        proc = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "."],
            cwd=health.staging,
            capture_output=True,
            text=True,
        )
        return "" if proc.returncode == 0 else proc.stdout + proc.stderr

    monkeypatch.setattr(health, "_run_static_gate", ruff_only)


def fake_canary(health, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Deterministic canary_release: swap the symlink, no child process."""
    from canary.core.util import atomic_symlink

    calls: list[str] = []

    def fake(release_dir, *, label=None, eval_gate=True):
        calls.append(str(label))
        atomic_symlink(health.current_link, release_dir)
        return {
            "ok": True,
            "release_id": label,
            "port": None,
            "child_pid": None,
            "promoted": True,
            "eval": {"enabled": False, "delta": None, "flagged": False},
        }

    monkeypatch.setattr(health, "canary_release", fake)
    return calls

