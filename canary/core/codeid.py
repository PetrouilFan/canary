"""The process/package code identity.

Two digests live in this tree and they answer different questions:

* :func:`canary.core.evals.code_identity` (3c8e280) is the *eval-cache* key: it
  digests the two files that decide how a task is scored, so two trees that
  measure the same way share a baseline entry.  It is unchanged by this module.
* :func:`package_identity` here is the *process* identity: it digests every
  ``.py`` file of a package, so a running process can say which code it is
  executing and a deploy can compare that with the release it claims to run.

The identity is what a process *is*, not what its tree was called: it is
computed from file contents, so a release id, a tag or the ``current`` symlink
can disagree with it and the disagreement is the finding.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: Hex characters of the digest that are exposed and logged.
IDENTITY_CHARS = 12


def package_identity(pkg_dir: Path | str | None = None) -> str | None:
    """Digest of every ``.py`` file under *pkg_dir*, ordered by relative path.

    ``pkg_dir`` is a package directory (the ``canary`` package), or ``None``
    for the package this process imported - which is what a serve process must
    report about itself.  ``None`` is returned when the tree cannot be read, so
    "unknown" is never mistaken for an identity.
    """
    if pkg_dir is None:
        import canary

        pkg_dir = canary.__path__[0]
    base = Path(pkg_dir).resolve()
    digest = hashlib.sha256()
    try:
        files = sorted(path for path in base.rglob("*.py") if path.is_file())
        if not files:
            return None
        for path in files:
            digest.update(path.relative_to(base).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    except OSError:
        return None
    return digest.hexdigest()[:IDENTITY_CHARS]
