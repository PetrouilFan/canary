"""Health probes, preflight tiers, and the release pipeline (spec §4.6, §5.3).

Everything that can mutate code goes through :meth:`Health.publish` or
:meth:`Health.revert`: both run under ``CANARY_ROOT/codebase.lock`` and follow
the step order recorded in ``shared/data/publish.state.json`` so a crash is
recoverable.  The listener socket is never closed, rebound, or duplicated: the
canary child inherits the production fd, holds it, and starts accepting only
when promoted with ``SIGUSR1``.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from .config import Config
from .evals import Evals
from .observability import Log
from .util import (
    FileLock,
    LockTimeout,
    atomic_symlink,
    atomic_write_json,
    atomic_write_text,
    ensure_dir,
    glob_match,
    kill_process_group,
    mono,
    pid_alive,
    pid_start_time,
    port_available,
    read_json,
    short_sha,
    truncate,
    utc_now,
    utc_stamp,
)

CONFIG_PATHS = (
    "governance.yaml", "models.yaml", "harness.yaml",
    "SOUL.md", "INSTRUCTIONS.md", "PERSONALITY.md",
)
CODE_PATTERNS = ("core/**", "api/**", "tests/**", "pyproject.toml", "canary/**")
EXTENSION_PATTERN = "extensions/**"
MEMORY_PATTERNS = ("memory/**", "*.md", "workspace/**")

TIER_ORDER = {"memory": 0, "extension": 1, "config": 2, "code": 3}


def classify(paths: list[str]) -> str:
    """Worst tier implicated by a set of repo-relative paths."""
    tier = "memory"
    for raw in paths:
        path = str(raw).replace("\\", "/").lstrip("./")
        if path.startswith("shared/"):
            path = path[len("shared/"):]
        if any(glob_match(p, path) for p in CODE_PATTERNS):
            return "code"
        if path in CONFIG_PATHS or path.startswith("evals/"):
            tier = max(tier, "config", key=lambda t: TIER_ORDER[t])
        elif glob_match(EXTENSION_PATTERN, path):
            tier = max(tier, "extension", key=lambda t: TIER_ORDER[t])
        else:
            tier = max(tier, "memory", key=lambda t: TIER_ORDER[t])
    return tier


class Ports:
    """Canary port leases in ``shared/data/canary_ports.json`` (§4.6)."""

    def __init__(self, data_dir: Path, log: Log, low: int = 9000, high: int = 9100):
        self.path = ensure_dir(data_dir) / "canary_ports.json"
        self.lock_path = data_dir / "canary_ports.lock"
        self.log = log
        self.low, self.high = int(low), int(high)

    def _read(self) -> dict[str, dict]:
        data = read_json(self.path, default={})
        return data if isinstance(data, dict) else {}

    def gc(self) -> list[int]:
        with FileLock(self.lock_path, op="ports-gc", timeout=5.0):
            leases = self._read()
            freed = []
            for port, meta in list(leases.items()):
                pid = int(meta.get("pid") or 0)
                if pid and not pid_alive(pid, meta.get("pid_start_time")):
                    leases.pop(port, None)
                    freed.append(int(port))
            atomic_write_json(self.path, leases)
            for port in freed:
                self.log.info("canary_port_freed", port=port, reason="pid gone")
            return freed

    def allocate(self, *, release_id: str, pid: int | None = None,
                 cmd: str = "") -> int | None:
        with FileLock(self.lock_path, op="ports-allocate", timeout=5.0):
            leases = self._read()
            now = utc_now()
            for port in range(self.low, self.high + 1):
                key = str(port)
                meta = leases.get(key)
                if meta and pid_alive(int(meta.get("pid") or 0),
                                     meta.get("pid_start_time")):
                    continue
                if not port_available(port):
                    self.log.warn("canary_port_occupied", port=port)
                    continue
                leases[key] = {
                    "pid": pid or os.getpid(),
                    "pid_start_time": pid_start_time(pid or os.getpid()),
                    "release_id": release_id,
                    "cmd": truncate(cmd, 200),
                    "ts": now,
                }
                atomic_write_json(self.path, leases)
                self.log.info("canary_port_allocated", port=port, release_id=release_id)
                return port
            self.log.warn("canary_port_exhausted", low=self.low, high=self.high)
            return None

    def update(self, port: int, **fields: Any) -> None:
        with FileLock(self.lock_path, op="ports-update", timeout=5.0):
            leases = self._read()
            meta = leases.get(str(port))
            if meta is None:
                return
            meta.update(fields)
            if "pid" in fields:
                meta["pid_start_time"] = pid_start_time(int(fields["pid"]))
            atomic_write_json(self.path, leases)

    def free(self, port: int | None) -> bool:
        if not port:
            return False
        with FileLock(self.lock_path, op="ports-free", timeout=5.0):
            leases = self._read()
            removed = leases.pop(str(port), None)
            atomic_write_json(self.path, leases)
        if removed is not None:
            self.log.info("canary_port_freed", port=port, reason="released")
            return True
        return False

    def in_use(self) -> dict[str, dict]:
        leases = self._read()
        alive = {}
        for port, meta in leases.items():
            pid = int(meta.get("pid") or 0)
            if pid and pid_alive(pid, meta.get("pid_start_time")):
                alive[port] = meta
        return alive


class Health:
    def __init__(
        self,
        config: Config,
        log: Log,
        agent: Any = None,
        *,
        listener_fd: int | None = None,
        port: int | None = None,
        drain_callback: Callable[[float], bool] | None = None,
    ) -> None:
        self.config = config
        self.log = log
        self.agent = agent
        self.listener_fd = listener_fd
        self.port = port
        self.drain_callback = drain_callback
        self.base_gate_override: tuple[bool, str] | None = None
        rng = config.get("canary.port_range") or [9000, 9100]
        self.ports = Ports(config.data_path, log, low=rng[0], high=rng[1])
        self.evals: Evals | None = None
        self._canary: dict[str, Any] = {}
        self.state = {
            "ready": False,
            "draining": False,
            "reason": "starting",
            "since": utc_now(),
        }

    # -- readiness ---------------------------------------------------------

    def set_ready(self, ready: bool, reason: str = "") -> None:
        self.state.update({"ready": bool(ready), "reason": reason,
                           "since": utc_now()})

    def set_draining(self, draining: bool = True) -> None:
        self.state["draining"] = bool(draining)
        if draining:
            self.state["ready"] = False
            self.state["reason"] = "draining"

    def check_ready(self) -> tuple[bool, str]:
        if self.state.get("draining"):
            return False, "draining"
        errors = list(getattr(self.config, "errors", []) or [])
        if errors:
            return False, f"config errors: {errors[0]}"
        if self.agent is not None:
            tools = getattr(self.agent, "tools", None)
            if tools is not None:
                available = tools.available()
                if not available:
                    return False, "no tools available"
            try:
                self.agent.memory.search("ready-probe", k=1)
            except Exception as exc:  # noqa: BLE001
                return False, f"memory not usable: {exc}"
            try:
                self.config.role("main")
            except Exception as exc:  # noqa: BLE001
                return False, f"model not configured: {exc}"
        return True, "ready"

    def ready_response(self) -> tuple[dict[str, Any], int]:
        ok, reason = self.check_ready()
        self.state["ready"] = ok
        self.state["reason"] = reason
        body = {
            "ready": ok,
            "reason": reason,
            "draining": bool(self.state.get("draining")),
            "release_id": self.config.release_id,
            "commit_sha": self.config.commit_sha,
            "agent_id": self.config.get("agent.id"),
            "since": self.state.get("since"),
        }
        return body, 200 if ok else 503

    def health_response(self) -> dict[str, Any]:
        base = {
            "status": "ok",
            "release_id": self.config.release_id,
            "commit_sha": self.config.commit_sha,
            "agent_id": self.config.get("agent.id"),
            "pid": os.getpid(),
            "timestamp": utc_now(),
        }
        if self.listener_fd is not None:
            try:
                sock = socket.socket(fileno=self.listener_fd)
                base["listener"] = sock.getsockname()
                sock.detach()
            except OSError:
                base["listener"] = None
        return base

    # -- base gate ---------------------------------------------------------

    def base_gate(self) -> tuple[bool, str]:
        if self.base_gate_override is not None:
            return self.base_gate_override
        if self.config.root is None:
            return False, "no CANARY_ROOT: publishing requires a root"
        ok, reason = self.check_ready()
        if not ok:
            return False, f"base gate red: {reason}"
        return True, "ok"

    # -- staging paths -----------------------------------------------------

    @property
    def root(self) -> Path:
        assert self.config.root is not None
        return Path(self.config.root)

    @property
    def staging(self) -> Path:
        return self.config.state_path / "staging"

    @property
    def releases_dir(self) -> Path:
        return ensure_dir(self.root / "releases")

    @property
    def current_link(self) -> Path:
        return self.root / "current"

    @property
    def publish_state_path(self) -> Path:
        return self.config.data_path / "publish.state.json"

    def current_release(self) -> Path | None:
        try:
            if self.current_link.is_symlink():
                target = Path(os.readlink(self.current_link))
                if not target.is_absolute():
                    target = self.current_link.parent / target
                return target
        except OSError:
            pass
        return None

    # -- git helpers -------------------------------------------------------

    def _git(self, *args: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(self.staging), *args],
            capture_output=True, text=True, timeout=timeout, check=False,
        )

    def last_green_sha(self) -> str | None:
        result = self._git("tag", "-l", "green/*")
        tags = sorted(result.stdout.split())
        if not tags:
            return None
        sha = self._git("rev-list", "-n", "1", tags[-1])
        return sha.stdout.strip() or None

    def green_tags(self) -> list[str]:
        """Green tags oldest -> newest.

        Ordered by tag creation time: zero-padded stamps alone cannot
        disambiguate two releases published in the same UTC second.
        """
        result = self._git(
            "for-each-ref",
            "--sort=creatordate",
            "--format=%(refname:short)",
            "refs/tags/green/*",
        )
        tags = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if tags:
            return tags
        return sorted(self._git("tag", "-l", "green/*").stdout.split())

    def staging_clean(self) -> bool:
        result = self._git("status", "--porcelain")
        return result.stdout.strip() == ""

    # -- preflight tiers ---------------------------------------------------

    def preflight_extension(self, path: Path | str) -> dict[str, Any]:
        path = Path(path)
        ruff = subprocess.run(
            [sys.executable, "-m", "ruff", "check", str(path)],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if ruff.returncode != 0:
            return {"ok": False, "tier": "extension",
                    "error": truncate(ruff.stdout + ruff.stderr, 4000)}
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except (SyntaxError, OSError) as exc:
            return {"ok": False, "tier": "extension", "error": str(exc)}
        return {"ok": True, "tier": "extension"}

    def preflight_config(self) -> list[str]:
        try:
            fresh = Config(root=self.config.root, state_path=self.config.state_path)
            errors = list(fresh.errors)
        except Exception as exc:  # noqa: BLE001 - invalid files are findings
            errors = [f"config reload failed: {exc}"]
        if not (self.config.state_path / "governance.yaml").exists():
            errors.append("governance.yaml missing")
        return errors

    def apply_config(self, paths: list[str], *, motivation: str | None = None,
                     session_id: str | None = None) -> dict[str, Any]:
        """Config tier: validate → smoke-boot → promote → commit state."""
        if self.config.root is None:
            return {"ok": True, "tier": "config", "note": "embedded: applied directly"}
        errors = self.preflight_config()
        if errors:
            return {"ok": False, "tier": "config", "error": "; ".join(errors)}
        result = self.canary_release(None, label="config", eval_gate=False)
        if not result.get("ok"):
            return {"ok": False, "tier": "config",
                    "error": result.get("error", "smoke boot failed"),
                    "log_tail": result.get("log_tail", "")}
        commit_paths = [p for p in paths if p in CONFIG_PATHS]
        if commit_paths and (self.config.state_path / ".git").exists():
            from .util import git_commit

            git_commit(self.config.state_path,
                       f"state: config {', '.join(sorted(commit_paths))}"
                       + (f" ({motivation})" if motivation else ""),
                       commit_paths, log=self.log)
        return {"ok": True, "tier": "config", "promoted": result.get("promoted", False)}

    # -- full code pipeline -------------------------------------------------

    def publish(self, patch: str | None = None, *, motivation: str | None = None,
                session_id: str | None = None) -> dict[str, Any]:
        return self._pipeline("publish", patch=patch, motivation=motivation,
                              session_id=session_id)

    def revert(self, release_id: str | None = None, *, motivation: str | None = None,
               session_id: str | None = None) -> dict[str, Any]:
        return self._pipeline("revert", release_id=release_id,
                              motivation=motivation, session_id=session_id)

    def _pipeline(self, op: str, *, patch: str | None = None,
                  release_id: str | None = None, motivation: str | None = None,
                  session_id: str | None = None) -> dict[str, Any]:
        started = mono()
        gate_ok, gate_reason = self.base_gate()
        if not gate_ok:
            return {"ok": False, "op": op, "error": gate_reason}
        timeout = float(self.config.get("releases.lock_timeout_s", 300) or 300)
        lock = FileLock(self.root / "codebase.lock", op=op, timeout=timeout)
        if not lock.acquire():
            return {"ok": False, "op": op,
                    "error": "another copy is publishing (codebase.lock held)",
                    "holder": lock.meta}
        self.log.info("pipeline_start", op=op, nonce=lock.nonce,
                      motivation=motivation, session_id=session_id)
        try:
            if not self.staging_clean() or self.publish_state_path.exists():
                recovery = self._recover(lock)
                if recovery.get("action") == "inconsistent":
                    return {"ok": False, "op": op, "error": recovery["error"]}
            if not self.staging_clean():
                return {"ok": False, "op": op,
                        "error": "staging is dirty and could not be recovered"}
            if op == "publish":
                if not patch:
                    return {"ok": False, "op": op, "error": "no patch provided"}
                result = self._do_publish(lock, patch, motivation, session_id, started)
            else:
                result = self._do_revert(lock, release_id, motivation, session_id,
                                         started)
        except LockTimeout as exc:
            result = {"ok": False, "op": op, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - pipeline failures are results
            self.log.error("pipeline_error", op=op, error=f"{type(exc).__name__}: {exc}")
            self._reset_staging()
            self._clear_state()
            result = {"ok": False, "op": op,
                      "error": f"{type(exc).__name__}: {exc}"}
        finally:
            if lock.fd is not None:
                lock.release()
        result["duration_ms"] = int((mono() - started) * 1000)
        self.log.deploy(op=result.get("op", op), ok=bool(result.get("ok")),
                        release_id=result.get("release_id"),
                        motivation=motivation, source_session=session_id,
                        eval_delta=(result.get("eval") or {}).get("delta"),
                        flagged=(result.get("eval") or {}).get("flagged"),
                        error=result.get("error"),
                        duration_ms=result["duration_ms"])
        return result

    def _do_publish(self, lock: FileLock, patch: str, motivation: str | None,
                    session_id: str | None, started: float) -> dict[str, Any]:
        patch_dir = ensure_dir(self.config.data_path / "patches" / "proposed")
        stamp = utc_stamp()
        patch_path = patch_dir / f"{stamp}.diff"
        atomic_write_text(patch_path, patch)
        paths = self._patch_paths(patch)
        state = {
            "op": "publish", "step": "apply", "nonce": lock.nonce,
            "patch": str(patch_path), "paths": paths, "started": utc_now(),
        }
        self._write_state(state)

        check = self._git("apply", "--check", str(patch_path))
        if check.returncode != 0:
            self._clear_state()
            return {"ok": False, "op": "publish", "error": "patch does not apply",
                    "log_tail": truncate(check.stderr, 2000)}
        applied = self._git("apply", str(patch_path))
        if applied.returncode != 0:
            self._reset_staging()
            self._clear_state()
            return {"ok": False, "op": "publish", "error": "git apply failed",
                    "log_tail": truncate(applied.stderr, 2000)}

        state["step"] = "static"
        self._write_state(state)
        red = self._run_static_gate()
        if red:
            self._reset_staging()
            self._clear_state()
            self.log.health_probe("preflight", "red", 0, red[:500])
            return {"ok": False, "op": "publish", "error": "static/unit gate failed",
                    "log_tail": truncate(red, 4000)}

        state["step"] = "commit"
        self._write_state(state)
        message = "agent: " + (motivation or "self-modification")
        if session_id:
            message += f" [session {session_id}]"
        add = self._git("add", "-A", "--", *paths)
        if add.returncode != 0:
            self._reset_staging()
            self._clear_state()
            return {"ok": False, "op": "publish", "error": "git add failed",
                    "log_tail": truncate(add.stderr, 2000)}
        commit = self._git("commit", "-q", "-m", message, "--", *paths)
        if commit.returncode != 0:
            self._reset_staging()
            self._clear_state()
            return {"ok": False, "op": "publish", "error": "git commit failed",
                    "log_tail": truncate(commit.stderr, 2000)}
        sha = self._git("rev-parse", "HEAD").stdout.strip()
        sha7 = short_sha(sha)
        release_id = f"{stamp}-{sha7}"
        tag = f"green/{release_id}"
        self._git("tag", "-a", tag, "-m", f"release {release_id}", sha)
        state.update({"step": "canary", "sha": sha, "release_id": release_id,
                      "tag": tag})
        self._write_state(state)
        release_dir = self._build_release(sha, release_id)

        gate = self.canary_release(release_dir, label=release_id, eval_gate=True)
        if not gate.get("ok"):
            kill = gate.get("child_pid")
            if kill:
                try:
                    kill_process_group(kill)
                except Exception:  # noqa: BLE001
                    pass
            self.ports.free(gate.get("port"))
            shutil.rmtree(release_dir, ignore_errors=True)
            self._git("tag", "-d", tag)
            previous = self.last_green_sha()
            if previous:
                self._git("reset", "--hard", previous)
            self._reset_staging()
            self._clear_state()
            self.log.health_probe("canary_gate", "red", 0,
                                  truncate(str(gate.get("error", "")), 500))
            return {"ok": False, "op": "publish", "release_id": release_id,
                    "error": gate.get("error", "canary gate failed"),
                    "log_tail": gate.get("log_tail", ""),
                    "eval": gate.get("eval")}

        self._clear_state()
        self.gc_releases()
        return {"ok": True, "op": "publish", "release_id": release_id,
                "commit_sha": sha, "tag": tag,
                "eval": gate.get("eval", {}),
                "promoted": gate.get("promoted", False),
                "port": gate.get("port"), "child_pid": gate.get("child_pid")}

    def _do_revert(self, lock: FileLock, release_id: str | None,
                   motivation: str | None, session_id: str | None,
                   started: float) -> dict[str, Any]:
        target = None
        if release_id:
            candidate = self.releases_dir / release_id
            if candidate.is_dir():
                target = candidate
        if target is None:
            tags = self.green_tags()
            if not tags:
                return {"ok": False, "op": "revert", "error": "no green tag to revert to"}
            names = [tag[len("green/"):] for tag in tags]
            current = self.current_release()
            if current is not None and current.name in names:
                older = names[: names.index(current.name)]
                if not older:
                    return {"ok": True, "op": "revert", "release_id": current.name,
                            "note": "already serving this release", "promoted": False}
                release_name = older[-1]
            else:
                release_name = names[-1]
            target = self.releases_dir / release_name
            if not target.is_dir():
                commit = self._git("rev-list", "-n", "1", f"green/{release_name}").stdout.strip()
                if not commit:
                    return {"ok": False, "op": "revert",
                            "error": f"green tag green/{release_name} has no commit"}
                target = self._build_release(commit, release_name)
        if self.current_release() == target:
            return {"ok": True, "op": "revert", "release_id": target.name,
                    "note": "already serving this release", "promoted": False}
        self._write_state({"op": "revert", "step": "canary", "nonce": lock.nonce,
                           "release_id": target.name, "started": utc_now()})
        result = self.canary_release(target, label=target.name, eval_gate=False)
        if not result.get("ok"):
            self._clear_state()
            return {"ok": False, "op": "revert", "release_id": target.name,
                    "error": result.get("error", "canary failed"),
                    "log_tail": result.get("log_tail", "")}
        self._clear_state()
        return {"ok": True, "op": "revert", "release_id": target.name,
                "promoted": result.get("promoted", False), "eval": {}}

    # -- static gate -------------------------------------------------------

    def _run_static_gate(self) -> str:
        env = os.environ.copy()
        with tempfile.TemporaryDirectory(prefix="canary-preflight-") as tmp:
            env["CANARY_ROOT"] = tmp
            env["HARNESS_STATE_PATH"] = str(Path(tmp) / "shared")
            for key in ("HARNESS_RELEASE_ID", "HARNESS_COMMIT_SHA"):
                env.pop(key, None)
            ruff = subprocess.run(
                [sys.executable, "-m", "ruff", "check", "."],
                cwd=self.staging, capture_output=True, text=True,
                timeout=30, env=env, check=False,
            )
            if ruff.returncode != 0:
                return "ruff check failed:\n" + ruff.stdout + ruff.stderr
            tests_dir = self.staging / "tests"
            if tests_dir.is_dir():
                pytest = subprocess.run(
                    [sys.executable, "-m", "pytest", "tests/", "-x", "-q",
                     "--timeout=60"],
                    cwd=self.staging, capture_output=True, text=True,
                    timeout=120, env=env, check=False,
                )
                if pytest.returncode not in (0, 5):
                    return ("pytest failed:\n" + pytest.stdout[-6000:]
                            + pytest.stderr[-2000:])
        return ""

    # -- releases ----------------------------------------------------------

    def _build_release(self, sha: str, release_id: str) -> Path:
        release_dir = self.releases_dir / release_id
        if release_dir.exists():
            return release_dir
        tar_path = self.config.data_path / f"release-{release_id}.tar"
        archive = self._git("archive", "--format=tar", "-o", str(tar_path), sha)
        if archive.returncode != 0:
            raise RuntimeError(f"git archive failed: {archive.stderr}")
        ensure_dir(release_dir)
        with tarfile.open(tar_path) as tar:
            tar.extractall(release_dir, filter="fully_trusted")  # noqa: S202 - own archive
        tar_path.unlink(missing_ok=True)
        self.log.info("release_built", release_id=release_id, sha=sha)
        return release_dir

    def gc_releases(self) -> list[str]:
        keep = int(self.config.get("releases.keep", 10) or 10)
        current = self.current_release()
        dirs = sorted((d for d in self.releases_dir.iterdir() if d.is_dir()),
                      key=lambda d: d.name)
        removed = []
        while len(dirs) > keep:
            oldest = dirs.pop(0)
            if current is not None and oldest == current:
                continue
            if self._canary.get("release_dir") == str(oldest):
                continue
            shutil.rmtree(oldest, ignore_errors=True)
            removed.append(oldest.name)
        if removed:
            self.log.info("releases_pruned", removed=removed)
        return removed

    def list_releases(self) -> dict[str, Any]:
        current = self.current_release()
        releases = []
        for path in sorted(self.releases_dir.iterdir()):
            if not path.is_dir():
                continue
            releases.append({
                "release_id": path.name,
                "current": current == path,
                "mtime": path.stat().st_mtime,
            })
        return {
            "current": current.name if current else None,
            "releases": releases,
            "green": self.green_tags(),
            "running_release": self.config.release_id,
        }

    # -- canary child ------------------------------------------------------

    def canary_release(self, release_dir: Path | None, *, label: str = "current",
                       eval_gate: bool = True) -> dict[str, Any]:
        probe_timeout = float(self.config.get("releases.canary_probe_timeout_s", 30)
                              or 30)
        release = release_dir or self.current_release() or Path.cwd()
        port = self.ports.allocate(release_id=label, cmd="canary serve --canary-mode")
        if port is None:
            return {"ok": False, "error": "no canary port available (9000-9100)"}
        log_path = ensure_dir(self.config.data_path / "canary") / f"{port}.log"
        env = os.environ.copy()
        env["CANARY_RELEASE_DIR"] = str(release)
        env["PYTHONPATH"] = str(release) + os.pathsep + env.get("PYTHONPATH", "")
        env["HARNESS_RELEASE_ID"] = release.name if release_dir else \
            self.config.release_id
        env["HARNESS_STATE_PATH"] = str(self.config.state_path)
        env["HARNESS_CANARY_PORT"] = str(port)
        if self.config.root is not None:
            env["CANARY_ROOT"] = str(self.config.root)
        cmd = [
            sys.executable, "-m", "canary", "serve",
            "--canary-mode",
            "--canary-port", str(port),
        ]
        if self.port:
            cmd += ["--port", str(self.port)]
        if self.listener_fd is not None:
            cmd += ["--fd", str(self.listener_fd)]
        pass_fds = (self.listener_fd,) if self.listener_fd is not None else ()
        with open(log_path, "ab") as fh:
            child = subprocess.Popen(
                cmd, env=env, cwd=str(self.root), stdin=subprocess.DEVNULL,
                stdout=fh, stderr=subprocess.STDOUT, close_fds=True,
                start_new_session=True, pass_fds=pass_fds,
            )
        self.ports.update(port, pid=child.pid)
        self._canary = {"pid": child.pid, "port": port, "release_dir":
                        str(release_dir) if release_dir else None}
        self.log.event("canary_spawned", pid=child.pid, port=port, label=label,
                       release=str(release))
        self.log.health_probe("canary_spawn", "ok", 0,
                              f"pid={child.pid} port={port}")

        probe = self.probe_child(port, probe_timeout, api_key=env.get("HARNESS_API_KEY"))
        if not probe.get("ok"):
            kill_process_group(child.pid)
            self.ports.free(port)
            self._canary = {}
            return {"ok": False, "error": probe.get("error", "probe failed"),
                    "port": port, "child_pid": child.pid,
                    "exit_code": child.poll(),
                    "log_tail": self._log_tail(log_path)}

        eval_result: dict[str, Any] = {"enabled": False}
        if eval_gate and int(self.config.get("evals.canary_tasks", 3) or 0) > 0:
            self.evals = self.evals or Evals(self.config, self.log)
            current_rel = self.current_release()
            current_id = current_rel.name if current_rel else self.config.release_id
            candidate_id = release_dir.name if release_dir else \
                f"config-{utc_stamp()}"
            mode = str(self.config.get("evals.gate", "warn") or "warn")
            if mode == "off":
                eval_result = {"enabled": False, "reason": "gate off"}
            else:
                eval_result = self.evals.canary_gate(candidate_id, current_id)
                if mode == "block" and eval_result.get("flagged"):
                    kill_process_group(child.pid)
                    self.ports.free(port)
                    self._canary = {}
                    return {"ok": False, "error": "eval gate blocked (delta below "
                                                  "-tolerance)",
                            "port": port, "child_pid": child.pid,
                            "eval": eval_result,
                            "log_tail": self._log_tail(log_path)}

        if release_dir is not None:
            atomic_symlink(self.current_link, release_dir)
            self.log.info("current_swapped", release_id=release_dir.name)
        promoted = False
        if self.listener_fd is not None or self.port:
            try:
                os.kill(child.pid, signal.SIGUSR1)
                promoted = True
                self.log.event("canary_promoted", pid=child.pid, port=port)
            except ProcessLookupError:
                return {"ok": False, "error": "canary child died before promotion",
                        "port": port, "child_pid": child.pid,
                        "eval": eval_result}
        self.ports.free(port)
        drain_timeout = float(self.config.get("releases.drain_timeout_s", 60) or 60)
        self.set_draining(True)
        if self.drain_callback is not None:
            try:
                self.drain_callback(drain_timeout)
            except Exception as exc:  # noqa: BLE001
                self.log.warn("drain_callback_failed", error=str(exc))
        self.log.health_probe("canary_probe", "ok", 0,
                              f"promoted={promoted} port={port}")
        return {"ok": True, "release_id": release_dir.name if release_dir else label,
                "port": port, "child_pid": child.pid, "promoted": promoted,
                "eval": eval_result}

    def probe_child(self, port: int, timeout: float = 30.0,
                    api_key: str | None = None) -> dict[str, Any]:
        key = api_key if api_key is not None else os.environ.get("HARNESS_API_KEY", "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        base = f"http://127.0.0.1:{port}"
        deadline = mono() + timeout
        last = "not probed"
        started = mono()
        while mono() < deadline:
            try:
                response = httpx.get(f"{base}/ready", headers=headers, timeout=2.0)
                if response.status_code == 200:
                    break
                last = f"ready returned {response.status_code}: {response.text[:300]}"
            except httpx.HTTPError as exc:
                last = f"ready unreachable: {exc}"
            time.sleep(2.0)
        else:
            self.log.health_probe("canary_ready", "red", (mono() - started) * 1000,
                                  last)
            return {"ok": False, "error": last}
        self.log.health_probe("canary_ready", "ok", (mono() - started) * 1000,
                              f"port={port}")
        try:
            smoke = httpx.post(f"{base}/agent/smoke", headers=headers, timeout=30.0)
        except httpx.HTTPError as exc:
            return {"ok": False, "error": f"smoke turn unreachable: {exc}"}
        if smoke.status_code != 200 or not smoke.json().get("ok"):
            return {"ok": False,
                    "error": f"smoke turn failed: {smoke.status_code} "
                             f"{smoke.text[:500]}"}
        self.log.health_probe("canary_smoke", "ok", 0, "mock turn passed")
        return {"ok": True}

    def unlock(self, nonce: str) -> dict[str, Any]:
        from .util import force_unlock, read_lock_meta

        lock_path = self.root / "codebase.lock"
        meta = read_lock_meta(lock_path)
        ok, message = force_unlock(lock_path, nonce)
        return {"ok": ok, "message": message, "meta": meta}

    # -- publish state -----------------------------------------------------

    def _write_state(self, state: dict[str, Any]) -> None:
        atomic_write_json(self.publish_state_path, state)

    def _clear_state(self) -> None:
        try:
            self.publish_state_path.unlink()
        except FileNotFoundError:
            pass

    def _recover(self, lock: FileLock) -> dict[str, Any]:
        state = read_json(self.publish_state_path, default=None)
        if not isinstance(state, dict):
            self._clear_state()
            return {"action": "clean"}
        step = state.get("step")
        self.log.warn("publish_state_recovered", step=step, op=state.get("op"))
        release_id = state.get("release_id")
        if step in ("apply", "static", "commit"):
            self._reset_staging()
            if release_id:
                shutil.rmtree(self.releases_dir / release_id, ignore_errors=True)
            self._clear_state()
            return {"action": "rolled_back", "step": step}
        if step in ("canary", "swap"):
            if release_id and state.get("tag"):
                self._git("tag", "-d", state["tag"])
            if release_id:
                shutil.rmtree(self.releases_dir / release_id, ignore_errors=True)
            self._reset_staging()
            self._clear_state()
            return {"action": "rolled_back", "step": step}
        self._reset_staging()
        self._clear_state()
        return {"action": "rolled_back", "step": step}

    def _reset_staging(self) -> None:
        previous = self.last_green_sha()
        if previous:
            self._git("reset", "--hard", previous)
        else:
            self._git("reset", "--hard", "HEAD")
        self._git("clean", "-fd")

    def _patch_paths(self, patch: str) -> list[str]:
        paths = set()
        for line in patch.splitlines():
            match = re.match(r"^\+\+\+ b/(.+)$", line)
            if match:
                paths.add(match.group(1))
        return sorted(paths) or ["."]

    def _log_tail(self, path: Path, n: int = 30) -> str:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-n:])

    # -- status ------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        ready, reason = self.check_ready()
        return {
            "ready": ready,
            "reason": reason,
            "draining": bool(self.state.get("draining")),
            "release_id": self.config.release_id,
            "commit_sha": self.config.commit_sha,
            "agent_id": self.config.get("agent.id"),
            "listener_fd": self.listener_fd,
            "port": self.port,
            "publishing": FileLock(self.root / "codebase.lock").locked()
            if self.config.root is not None else False,
            "canary_ports": self.ports.in_use(),
            "last_deploy_flagged": self.log.last_deploy_flagged(),
            "canary": dict(self._canary),
        }
