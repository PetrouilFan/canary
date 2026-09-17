"""test_memory_cross_process.py - concurrent writers share one memory store."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from canary.core.config import Config
from canary.core.memory import Memory
from canary.core.observability import Log
from tests.integration.conftest import clean_env, init_root, run_python

pytestmark = pytest.mark.integration

_WRITER = """
import os, sys
from pathlib import Path
from canary.core.config import Config
from canary.core.memory import Memory

body = sys.argv[1]
cfg = Config(root=Path(os.environ["CANARY_ROOT"]))
cfg.set("memory.git", False)
memory = Memory(cfg)
entry = memory.save(body, tags=["cross-process"], source="integration")
print(entry.id)
"""


@pytest.mark.timeout(120)
def test_parallel_writers_and_reader(root: Path) -> None:
    init_root(root)
    cfg = Config(root=root)
    cfg.set("memory.git", False)
    log = Log(cfg)
    memory = Memory(cfg, log)
    memory.save("parent entry alpha", tags=["cross-process"], source="parent")

    env = clean_env(root)
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _WRITER, body],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for body in ("child entry beta", "child entry gamma")
    ]
    ids = []
    for proc in procs:
        out, err = proc.communicate(timeout=90)
        assert proc.returncode == 0, err
        ids.append(out.strip())
    assert len(set(ids)) == 2

    fresh = Memory(Config(root=root), log)
    entries = {e.id: e for e in fresh.all_entries()}
    for entry_id in ids:
        assert entry_id in entries, f"missing {entry_id} after cross-process save"
    hits = fresh.search("entry alpha", tags=["cross-process"])
    assert hits
    assert any("alpha" in h.entry.body for h in hits)


@pytest.mark.timeout(120)
def test_second_process_sees_memory_written_by_first(root: Path) -> None:
    init_root(root)
    env = clean_env(root)
    proc = run_python(_WRITER, env, "single child entry", timeout=90)
    assert proc.returncode == 0, proc.stderr
    entry_id = proc.stdout.strip()

    cfg = Config(root=root)
    cfg.set("memory.git", False)
    memory = Memory(cfg, Log(cfg))
    assert memory.get(entry_id) is not None
