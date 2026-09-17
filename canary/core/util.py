"""Shared primitives: atomic writes, flock locks, PID liveness, dotenv, globs.

Everything here assumes a local POSIX filesystem (spec Principle 9): flock and
rename(2) are the coordination primitives. No NFS, no multi-host.
"""

from __future__ import annotations

import fcntl
import fnmatch
import json
import os
import re
import signal
import socket
import subprocess
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

UTC = UTC


# ---------------------------------------------------------------------------
# time
# ---------------------------------------------------------------------------

def utc_now() -> str:
    """ISO-8601 UTC, second precision, Z suffix."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_day() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def utc_stamp() -> str:
    """Zero-padded UTC timestamp used in release ids and green tags."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def parse_stamp(stamp: str) -> float:
    """Parse utc_now()/utc_stamp() output into a unix timestamp."""
    text = (stamp or "").strip()
    if not text:
        raise ValueError("empty timestamp")
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y%m%dT%H%M%SZ"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=UTC).timestamp()
        except ValueError:
            continue
    raise ValueError(f"unrecognized timestamp: {stamp!r}")


def mono() -> float:
    return time.monotonic()


def new_nonce() -> str:
    return uuid.uuid4().hex[:16]


def hostname() -> str:
    try:
        return socket.gethostname().split(".")[0]
    except Exception:
        return "localhost"


# ---------------------------------------------------------------------------
# atomic filesystem writes
# ---------------------------------------------------------------------------

def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    """Write-temp + fsync + os.replace. Crash-atomic (spec 6.3)."""
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.parent / f".{p.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, p)
    _fsync_dir(p.parent)


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | os.PathLike[str], obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def append_line(path: str | os.PathLike[str], line: str, fsync: bool = False) -> None:
    """Append one complete line with a single O_APPEND write (spec 6.3)."""
    p = Path(path)
    ensure_dir(p.parent)
    data = line if line.endswith("\n") else line + "\n"
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, data.encode("utf-8"))
        if fsync:
            os.fsync(fd)
    finally:
        os.close(fd)


def append_jsonl(path: str | os.PathLike[str], obj: Any, fsync: bool = False) -> None:
    append_line(path, json.dumps(obj, ensure_ascii=False, separators=(",", ":")), fsync=fsync)


def read_jsonl(path: str | os.PathLike[str]) -> Iterator[dict]:
    """Yield parsed objects, tolerating a torn final line from a crash."""
    try:
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return


def tail_lines(path: str | os.PathLike[str], n: int) -> list[str]:
    """Last n lines; reads the tail of the file without loading it all."""
    p = Path(path)
    try:
        size = p.stat().st_size
    except FileNotFoundError:
        return []
    block = 8192
    data = b""
    with open(p, "rb") as fh:
        pos = size
        while pos > 0 and data.count(b"\n") <= n:
            step = min(block, pos)
            pos -= step
            fh.seek(pos)
            data = fh.read(step) + data
    lines = data.decode("utf-8", errors="replace").splitlines()
    return lines[-n:]


# ---------------------------------------------------------------------------
# symlinks
# ---------------------------------------------------------------------------

def atomic_symlink(link_path: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
    """Create temp symlink beside target then os.replace (never ln -sfn)."""
    link = Path(link_path)
    ensure_dir(link.parent)
    tmp = link.parent / f".{link.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(os.fspath(target), tmp)
    os.replace(tmp, link)
    _fsync_dir(link.parent)


# ---------------------------------------------------------------------------
# file locks (flock)
# ---------------------------------------------------------------------------

class LockTimeout(TimeoutError):
    pass


class FileLock:
    """Advisory flock with holder metadata and bounded wait.

    Metadata lives in the lock file itself: {pid, host, op, started, nonce}.
    Kernel releases flock automatically on process death (including SIGKILL),
    so there are no stale locks to break.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        op: str = "lock",
        timeout: float = 0.0,
        *,
        wait: bool = True,
    ):
        self.path = Path(path)
        self.op = op
        self.timeout = max(0.0, timeout)
        self.wait = wait
        self.fd: int | None = None
        self.nonce: str | None = None

    def acquire(self) -> bool:
        ensure_dir(self.path.parent)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        deadline = mono() + self.timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (11, 13):  # EAGAIN, EACCES
                    os.close(fd)
                    raise
                if not self.wait:
                    os.close(fd)
                    return False
                if mono() >= deadline:
                    os.close(fd)
                    return False
                time.sleep(0.05)
        self.fd = fd
        self.nonce = new_nonce()
        meta = {
            "pid": os.getpid(),
            "host": hostname(),
            "op": self.op,
            "started": utc_now(),
            "nonce": self.nonce,
        }
        try:
            payload = json.dumps(meta).encode("utf-8")
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, payload)
            os.fsync(fd)
        except OSError:
            pass
        return True

    def locked(self) -> bool:
        """True if some process (possibly us) currently holds the lock."""
        ensure_dir(self.path.parent)
        try:
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError:
            return False
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        finally:
            os.close(fd)

    def release(self) -> None:
        if self.fd is None:
            return
        try:
            os.ftruncate(self.fd, 0)
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            os.close(self.fd)
            self.fd = None

    def __enter__(self) -> FileLock:
        if not self.acquire():
            raise LockTimeout(f"could not acquire {self.path} within {self.timeout}s")
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()

    @property
    def meta(self) -> dict:
        return read_lock_meta(self.path)


def read_lock_meta(path: str | os.PathLike[str]) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            raw = fh.read().strip()
        return json.loads(raw) if raw else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def force_unlock(path: str | os.PathLike[str], nonce: str) -> tuple[bool, str]:
    """Operator break-glass for a wedged lock. Requires the holder nonce.

    With kernel flock a dead holder needs no break-glass; this exists for a
    live-but-stuck holder. We only clear the file when no live holder exists.
    """
    meta = read_lock_meta(path)
    if not meta:
        return True, "no holder metadata; lock is free"
    if meta.get("nonce") != nonce:
        return False, "nonce mismatch with current holder metadata"
    pid = meta.get("pid")
    if isinstance(pid, int) and pid_alive(pid):
        return False, f"holder pid {pid} is alive; stop that process first"
    p = Path(path)
    try:
        fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as exc:
        return False, f"cannot open lock file: {exc}"
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False, "lock is still held by a live process"
        os.ftruncate(fd, 0)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True, "stale lock metadata cleared"
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# processes / pid liveness
# ---------------------------------------------------------------------------

def pid_start_time(pid: int) -> str | None:
    """Kernel start time of a pid (jiffies since boot), from /proc. None if gone."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            content = fh.read()
    except (FileNotFoundError, ProcessLookupError):
        return None
    # comm may contain spaces/parens: split after the final ')'
    try:
        rest = content.rsplit(")", 1)[1].split()
        return rest[19]  # field 22 overall, state is field 3 -> index 19
    except (IndexError, ValueError):
        return None


def pid_alive(pid: int, start_time: str | None = None) -> bool:
    if pid <= 0:
        return False
    cur = pid_start_time(pid)
    if cur is None:
        return False
    if start_time is not None and str(start_time) != cur:
        return False
    return True


def kill_pid(pid: int, grace: float = 5.0) -> bool:
    """Terminate a single pid gracefully, then forcibly."""
    if not pid_alive(pid):
        return True
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = mono() + grace
    while mono() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return not pid_alive(pid)


def kill_process_group(pgid: int, grace: float = 5.0) -> bool:
    """SIGTERM then SIGKILL to an entire process group (spec 4.9)."""
    if pgid <= 1:
        return False

    def alive() -> bool:
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    if not alive():
        return True
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = mono() + grace
    while mono() < deadline:
        if not alive():
            return True
        time.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    time.sleep(0.2)
    return not alive()


# ---------------------------------------------------------------------------
# dotenv / env helpers
# ---------------------------------------------------------------------------

_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def parse_dotenv(path: str | os.PathLike[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                m = _ENV_LINE.match(line)
                if not m:
                    continue
                key, value = m.group(1), m.group(2)
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                out[key] = value
    except FileNotFoundError:
        pass
    return out


def load_dotenv(path: str | os.PathLike[str], override: bool = False) -> dict[str, str]:
    values = parse_dotenv(path)
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return values


_SECRET_HINT = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD)", re.IGNORECASE)


def secret_values() -> list[str]:
    out = []
    for key, value in os.environ.items():
        if _SECRET_HINT.search(key) and len(value) >= 8:
            out.append(value)
    return out


def redact(text: str) -> str:
    for value in secret_values():
        if value in text:
            text = text.replace(value, "***")
    return text


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# dict / path helpers
# ---------------------------------------------------------------------------

def dot_get(values: dict, path: str, default: Any = None) -> Any:
    node: Any = values
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def glob_match(pattern: str, path: str) -> bool:
    """Path-glob match with ** = zero or more path components."""
    pat = str(pattern).strip("/").split("/")
    parts = str(path).strip("/").split("/")

    def rec(i: int, j: int) -> bool:
        if i == len(pat):
            return j == len(parts)
        if pat[i] == "**":
            if i + 1 == len(pat):
                return True
            for k in range(j, len(parts) + 1):
                if rec(i + 1, k):
                    return True
            return False
        if j == len(parts):
            return False
        if not fnmatch.fnmatchcase(parts[j], pat[i]):
            return False
        return rec(i + 1, j + 1)

    return rec(0, 0)


def slugify(text: str, max_len: int = 48) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    text = text.strip("_")
    return (text[:max_len] or "entry").strip("_")


def short_sha(text: str, n: int = 7) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1")


def parse_size(text: str | int) -> int:
    if isinstance(text, int):
        return text
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([kKmMgG]?[bB]?)\s*$", str(text))
    if not m:
        return int(text)
    value = float(m.group(1))
    unit = m.group(2).lower().rstrip("b")
    factor = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3}.get(unit, 1)
    return int(value * factor)


def truncate(text: str, limit: int, marker: str = "\n[... truncated ...]") -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - len(marker))] + marker


# ---------------------------------------------------------------------------
# git helpers
# ---------------------------------------------------------------------------

def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def git_commit(
    repo: str | os.PathLike[str],
    message: str,
    paths: list[str],
    *,
    log: Any = None,
    lock_path: str | os.PathLike[str] | None = None,
    lock_timeout: float = 30.0,
) -> bool:
    """Stage+commit exactly ``paths`` inside ``repo``.

    Staging interleave is the classic race with shared checkouts: a bare
    ``git commit`` sweeps up files another process staged. We always pass an
    explicit pathspec to both ``add`` and ``commit``, and optionally serialize
    with an flock when the caller supplies one. ``git index.lock`` is handled
    by a short bounded retry rather than blind waiting.
    """
    repo_path = Path(repo)
    if not (repo_path / ".git").exists():
        return False
    lock: FileLock | None = None
    if lock_path is not None:
        lock = FileLock(lock_path, op="git", timeout=lock_timeout, wait=True)
        if not lock.acquire():
            if log:
                log.warn("git_commit_lock_timeout", repo=str(repo_path))
            return False
    try:
        spec = list(paths) or ["."]
        last = ""
        for attempt in range(3):
            added = _git(repo_path, "add", "-A", "--", *spec)
            if added.returncode != 0:
                last = added.stderr.strip()
                if "index.lock" in last and attempt < 2:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                if log:
                    log.warn("git_add_failed", repo=str(repo_path), error=last)
                return False
            committed = _git(repo_path, "commit", "-q", "-m", message, "--", *spec)
            if committed.returncode != 0:
                last = (committed.stderr or committed.stdout).strip()
                if "nothing to commit" in last:
                    return False
                if "index.lock" in last and attempt < 2:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                if log:
                    log.warn("git_commit_failed", repo=str(repo_path), error=last)
                return False
            return True
        return False
    finally:
        if lock is not None:
            lock.release()
