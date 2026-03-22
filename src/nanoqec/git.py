"""Git metadata helpers for runtime artifacts."""

from __future__ import annotations

import subprocess


def _run_git(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def current_branch_name() -> str:
    """Return the current git branch name if available."""

    return _run_git(["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"


def current_git_sha() -> str:
    """Return the current git sha if available."""

    return _run_git(["rev-parse", "HEAD"]) or "unknown"
