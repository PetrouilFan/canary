"""test_port_gc.py - canary port allocation, reuse, stale GC."""

from __future__ import annotations

from pathlib import Path

import pytest

from canary.core.config import Config
from canary.core.health import Ports
from canary.core.observability import Log
from canary.core.util import atomic_write_json
from tests.integration.conftest import init_root

pytestmark = pytest.mark.integration


@pytest.mark.timeout(60)
def test_port_allocation_reuse_and_gc(root: Path) -> None:
    init_root(root)
    cfg = Config(root=root)
    log = Log(cfg)
    ports = Ports(cfg.data_path, log, 9000, 9010)

    ports.gc()
    first = ports.allocate(release_id="r1", cmd="test")
    second = ports.allocate(release_id="r2", cmd="test")
    assert first and second and first != second
    assert 9000 <= first <= 9010 and 9000 <= second <= 9010

    ports.free(first)
    third = ports.allocate(release_id="r3", cmd="test")
    assert third == first

    data = cfg.data_path / "canary_ports.json"
    atomic_write_json(
        data,
        {
            str(9020): {  # out of range -> dropped
                "pid": 999999,
                "cmd": "x",
                "release_id": "gone",
                "started": "old",
            },
            str(first): {
                "pid": 999999,
                "cmd": "x",
                "release_id": "dead",
                "started": "old",
            },
        },
    )
    freed = ports.gc()
    assert set(freed) == {9020, first}
    loaded = Ports(cfg.data_path, log, 9000, 9010)
    released = loaded.allocate(release_id="r4", cmd="test")
    assert released == first, "GC should release entries whose pid is gone"
    loaded.free(released)
