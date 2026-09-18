"""test_git.py - git must work without a user-configured identity (CI)."""

from __future__ import annotations

import os
import subprocess

from canary.core import util


def _run(repo, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_git_commit_without_global_identity(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "init", "-q")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    (repo / "file.txt").write_text("content\n", encoding="utf-8")

    assert util.git_commit(repo, "commit without identity", ["."]) is True

    log = _run(repo, "log", "--oneline")
    assert "commit without identity" in log.stdout
