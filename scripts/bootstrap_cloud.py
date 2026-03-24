"""Bootstrap or plan a single-host NanoQEC cloud environment."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for cloud bootstrap."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--python-version", default="3.11")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args(argv)


def build_bootstrap_commands(
    repo_root: Path,
    python_version: str,
    skip_validation: bool,
) -> list[str]:
    """Return the ordered shell commands for a repo-local cloud bootstrap."""

    repo = str(repo_root.resolve())
    torch_probe = " ".join(
        [
            "import json, torch;",
            "print(json.dumps({",
            "'torch_version': torch.__version__,",
            "'cuda_available': torch.cuda.is_available(),",
            "'cuda_device_count': torch.cuda.device_count(),",
            "'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,",
            "}))",
        ]
    )
    commands = [
        "nvidia-smi",
        (
            "bash -lc "
            "'command -v uv >/dev/null 2>&1 "
            "|| curl -LsSf https://astral.sh/uv/install.sh | sh'"
        ),
        (
            "bash -lc "
            f"'source \"$HOME/.local/bin/env\" && cd {repo} "
            f"&& uv python install {python_version}'"
        ),
        (
            "bash -lc "
            f"'source \"$HOME/.local/bin/env\" && cd {repo} "
            "&& uv sync --all-extras'"
        ),
        (
            "bash -lc "
            f"'source \"$HOME/.local/bin/env\" && cd {repo} "
            f"&& uv run python -c \"{torch_probe}\"'"
        ),
    ]
    if not skip_validation:
        commands.extend(
            [
                f"bash -lc 'source \"$HOME/.local/bin/env\" && cd {repo} && uv run ruff check .'",
                f"bash -lc 'source \"$HOME/.local/bin/env\" && cd {repo} && uv run pytest'",
            ]
        )
    return commands


def run_command(command: str) -> dict[str, Any]:
    """Run one shell command and capture structured output."""

    completed = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    return {
        "command": command,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "returncode": completed.returncode,
    }


def main(argv: list[str] | None = None) -> int:
    """Run or print the cloud bootstrap plan."""

    args = parse_args(argv)
    commands = build_bootstrap_commands(
        repo_root=args.repo_root,
        python_version=args.python_version,
        skip_validation=args.skip_validation,
    )
    if args.plan_only:
        print(
            json.dumps(
                {
                    "repo_root": str(args.repo_root.resolve()),
                    "python_version": args.python_version,
                    "plan_only": True,
                    "skip_validation": args.skip_validation,
                    "commands": commands,
                },
                sort_keys=True,
            )
        )
        return 0
    results = [run_command(command) for command in commands]
    print(
        json.dumps(
            {
                "repo_root": str(args.repo_root.resolve()),
                "python_version": args.python_version,
                "plan_only": False,
                "skip_validation": args.skip_validation,
                "commands": commands,
                "results": results,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
