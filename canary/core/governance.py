"""Governance for tool-mediated writes (spec 4.5).

Honest contract: governance is a guardrail for the ``write`` and ``edit``
tools, not a sandbox.  The agent can edit ``governance.yaml`` itself and
``bash`` bypasses governance entirely.  Default is deny; deny wins on overlap.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

from .config import Config, default_governance
from .util import glob_match

try:  # pragma: no cover - exercised through Config which also imports yaml
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


def _matches(pattern: str, candidates: list[str]) -> bool:
    for candidate in candidates:
        if glob_match(pattern, candidate) or glob_match(pattern.rstrip("/") + "/**", candidate):
            return True
        if fnmatch.fnmatch(candidate, pattern):
            return True
    return False


class Governance:
    """Resolves a target path against allow_write/deny_write patterns."""

    def __init__(self, config: Config, log: Any = None) -> None:
        self.config = config
        self.log = log
        self.allow: list[str] = []
        self.deny: list[str] = []
        self.impact: dict[str, str] = {}
        self.error: str | None = None
        self.load()

    # -- loading ---------------------------------------------------------

    @property
    def path(self) -> Path:
        return self.config.governance_path

    def load(self) -> None:
        data: dict[str, Any] | None = None
        self.error = None
        try:
            if self.path.is_file() and yaml is not None:
                loaded = yaml.safe_load(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = loaded
                elif loaded is not None:
                    self.error = "governance.yaml is not a mapping"
        except Exception as exc:  # noqa: BLE001 - a broken file must not kill tools
            self.error = f"governance.yaml load failed: {exc}"
            if self.log is not None:
                self.log.warn("governance_load_failed", path=str(self.path), error=str(exc))
        if data is None:
            data = default_governance()
            if self.error is None and not self.path.is_file():
                self.error = None
        allow = data.get("allow_write") or []
        deny = data.get("deny_write") or []
        self.allow = [str(p) for p in allow if isinstance(p, (str, Path))]
        self.deny = [str(p) for p in deny if isinstance(p, (str, Path))]
        raw_impact = data.get("impact") or {}
        self.impact = (
            {str(k): str(v) for k, v in raw_impact.items()}
            if isinstance(raw_impact, dict)
            else {}
        )

    def save(self, allow: list[str], deny: list[str], impact: dict[str, str] | None = None) -> None:
        if yaml is None:  # pragma: no cover
            raise RuntimeError("PyYAML is required to write governance.yaml")
        from .util import atomic_write_text

        text = yaml.safe_dump(
            {
                "allow_write": list(allow),
                "deny_write": list(deny),
                "impact": dict(impact if impact is not None else self.impact),
            },
            sort_keys=False,
            default_flow_style=False,
        )
        atomic_write_text(self.path, text)
        self.allow, self.deny = list(allow), list(deny)
        self.impact = dict(impact if impact is not None else self.impact)
        if self.log is not None:
            self.log.event("governance_saved", allow=len(allow), deny=len(deny))

    # -- evaluation ------------------------------------------------------

    def candidates(self, path: str | Path) -> list[str]:
        """Return path rendered relative to root and to the state store."""
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        else:
            p = p.resolve()
        out: list[str] = []
        root = self.config.root
        if root is not None:
            try:
                out.append(p.relative_to(root).as_posix())
            except ValueError:
                pass
        try:
            out.append(p.relative_to(self.config.state_path).as_posix())
            out.append("shared/" + p.relative_to(self.config.state_path).as_posix())
        except ValueError:
            pass
        staging = self.config.state_path / "staging"
        try:
            out.append(p.relative_to(staging).as_posix())
        except ValueError:
            pass
        base = self.config.base_dir.resolve()
        try:
            out.append(p.relative_to(base).as_posix())
        except ValueError:
            pass
        out.append(p.as_posix().lstrip("/"))
        seen: list[str] = []
        for item in out:
            if item not in seen:
                seen.append(item)
        return seen

    def check(self, path: str | Path) -> tuple[bool, str]:
        """Return (allowed, reason).  Default deny, deny wins."""
        candidates = self.candidates(path)
        for pattern in self.deny:
            if _matches(pattern, candidates):
                return False, f"path {candidates[0]!r} matches deny_write {pattern!r}"
        for pattern in self.allow:
            if _matches(pattern, candidates):
                return True, f"allowed by {pattern!r}"
        return False, f"path {candidates[0]!r} matches no allow_write pattern (default deny)"

    def allowed(self, path: str | Path) -> bool:
        return self.check(path)[0]

    # -- validation ------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "allow_write": list(self.allow),
            "deny_write": list(self.deny),
            "impact": dict(self.impact),
        }

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.path.is_file() and yaml is not None:
            try:
                yaml.safe_load(self.path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"governance.yaml: invalid YAML: {exc}")
        if self.error:
            errors.append(f"governance.yaml: {self.error}")
        if not self.allow and not self.path.is_file():
            # Defaults are in force; that is fine, not an error.
            pass
        return errors
