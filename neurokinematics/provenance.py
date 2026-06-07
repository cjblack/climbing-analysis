"""Lightweight provenance helpers for recording how an output was produced.

Used by :meth:`ExperimentSession._record_session_output` to stamp each output
with the code revision, a hash of the config/parameters, and a cheap fingerprint
of its inputs — so results are reproducible and traceable without external
data-versioning tooling.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

# session_outputs.yaml layout version — bump when the schema changes so readers
# can migrate gracefully (the nested {session_id, outputs} layout is v2; the old
# flat layout is treated as v1).
SCHEMA_VERSION = 2

_GIT_CACHE = "__unset__"   # cache the revision once per process


def git_revision() -> str | None:
    """Short commit SHA of the neurokinematics repo, with a '-dirty' suffix when
    the working tree has uncommitted changes. None if not a git checkout.

    Cached per process — the repo state is not expected to change mid-run.
    """
    global _GIT_CACHE
    if _GIT_CACHE != "__unset__":
        return _GIT_CACHE
    _GIT_CACHE = None
    try:
        root = Path(__file__).resolve().parent
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        if sha:
            dirty = subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
            _GIT_CACHE = f"{sha}-dirty" if dirty else sha
    except Exception:
        _GIT_CACHE = None
    return _GIT_CACHE


def hash_config(cfg) -> str | None:
    """Stable short hash of a config dict (order-independent)."""
    if not cfg:
        return None
    try:
        blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]
    except Exception:
        return None


def fingerprint_input(path) -> dict | None:
    """Cheap, metadata-only fingerprint of an input file or folder.

    Never reads file contents (raw ephys can be many GB) — for a folder it
    records the immediate child count and the most recent mtime, enough to
    detect "the input changed but the output didn't".
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return {"path": str(path), "exists": False}
    try:
        if p.is_file():
            st = p.stat()
            return {"path": str(p), "size": st.st_size, "mtime": int(st.st_mtime)}
        children = list(p.iterdir())
        latest = max((c.stat().st_mtime for c in children),
                     default=p.stat().st_mtime)
        return {"path": str(p), "n_children": len(children),
                "latest_mtime": int(latest)}
    except Exception:
        return {"path": str(p)}
