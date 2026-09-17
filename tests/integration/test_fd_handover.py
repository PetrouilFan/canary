"""test_fd_handover.py - inherited listener survives promotion (spec §5.3)."""

from __future__ import annotations

import os
import signal
import socket
import time
from pathlib import Path

import pytest

from canary.core.util import kill_process_group
from tests.integration.conftest import (
    clean_env,
    free_port,
    http_get,
    listener,
    spawn_serve,
    wait_ready,
)

pytestmark = pytest.mark.integration


def _connect_ok(port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.mark.timeout(120)
def test_fd_handover_and_drain(initialized: Path) -> None:
    root = initialized
    env = clean_env(root)
    prod_port = free_port()
    canary_port = free_port()
    prod_listener = listener(prod_port)
    try:
        proc = spawn_serve(
            root,
            "--canary-mode",
            "--canary-port",
            str(canary_port),
            "--port",
            str(prod_port),
            "--fd",
            str(prod_listener.fileno()),
            env=env,
            pass_fds=(prod_listener.fileno(),),
        )
    finally:
        prod_listener.close()

    try:
        assert wait_ready(canary_port), "canary child never became ready"
        headers = {"Authorization": "Bearer integration-key"}

        ready = http_get(f"http://127.0.0.1:{canary_port}/ready", headers=headers)
        assert ready.status_code == 200

        assert _connect_ok(prod_port), "inherited listener should accept TCP"
        try:
            http_get(f"http://127.0.0.1:{prod_port}/ready", headers=headers, timeout=1.0)
            speaking_before = True
        except Exception:  # noqa: BLE001 - nobody accepts yet
            speaking_before = False
        assert speaking_before is False, "production listener must not accept pre-promotion"

        os.kill(proc.pid, signal.SIGUSR1)
        deadline = time.monotonic() + 15
        promoted = False
        while time.monotonic() < deadline:
            try:
                response = http_get(
                    f"http://127.0.0.1:{prod_port}/ready", headers=headers, timeout=1.0
                )
                if response.status_code == 200:
                    promoted = True
                    break
            except Exception:  # noqa: BLE001 - poll
                pass
            time.sleep(0.2)
        assert promoted, "production listener did not start answering after SIGUSR1"

        canary_after = http_get(f"http://127.0.0.1:{canary_port}/health", headers=headers)
        assert canary_after.status_code == 200

        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=45) == 0
    finally:
        if proc.poll() is None:
            kill_process_group(proc.pid)
        proc.wait(timeout=10)
