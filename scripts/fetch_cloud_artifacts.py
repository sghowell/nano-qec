"""Fetch NanoQEC cloud artifacts back to a local workspace."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for artifact fetch."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-repo-root", type=Path, required=True)
    parser.add_argument("--local-workspace", type=Path, default=Path("."))
    parser.add_argument("--include-data", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args(argv)


def build_rsync_commands(
    remote_host: str,
    remote_repo_root: Path,
    local_workspace: Path,
    include_data: bool,
) -> list[str]:
    """Return the rsync commands required to fetch cloud artifacts."""

    remote_root = str(remote_repo_root)
    local_root = str(local_workspace.resolve())
    paths = ["checkpoints", "results"]
    if include_data:
        paths.append("data")
    return [
        f"rsync -avz {remote_host}:{remote_root}/{path}/ {local_root}/{path}/"
        for path in paths
    ]


def run_command(command: str) -> dict[str, Any]:
    """Run one rsync command and capture output."""

    completed = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    return {
        "command": command,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "returncode": completed.returncode,
    }


def main(argv: list[str] | None = None) -> int:
    """Run or print the rsync fetch commands."""

    args = parse_args(argv)
    commands = build_rsync_commands(
        remote_host=args.remote_host,
        remote_repo_root=args.remote_repo_root,
        local_workspace=args.local_workspace,
        include_data=args.include_data,
    )
    if args.print_only:
        print(
            json.dumps(
                {
                    "print_only": True,
                    "commands": commands,
                    "remote_host": args.remote_host,
                    "remote_repo_root": str(args.remote_repo_root),
                    "local_workspace": str(args.local_workspace.resolve()),
                },
                sort_keys=True,
            )
        )
        return 0
    results = [run_command(command) for command in commands]
    print(
        json.dumps(
            {
                "print_only": False,
                "commands": commands,
                "results": results,
                "remote_host": args.remote_host,
                "remote_repo_root": str(args.remote_repo_root),
                "local_workspace": str(args.local_workspace.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
