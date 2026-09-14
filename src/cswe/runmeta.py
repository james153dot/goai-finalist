"""Run metadata required on every newly generated numerical artifact."""

from __future__ import annotations

import datetime
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def git_commit_sha(repo: Path = ROOT) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def working_tree_dirty(repo: Path = ROOT) -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return True


def utc_timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def run_metadata(*, openfoam_executed: bool, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp_utc": utc_timestamp(),
        "git_commit_sha": git_commit_sha(),
        "working_tree_dirty": working_tree_dirty(),
        "openfoam_executed": bool(openfoam_executed),
    }
    if extra:
        payload.update(extra)
    return payload
