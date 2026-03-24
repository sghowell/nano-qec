"""Run prepare/train/eval as one single-host cloud experiment helper."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the cloud run helper."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument(
        "--profile",
        choices=["local-d3-v1", "local-d5-v1"],
        default="local-d5-v1",
    )
    parser.add_argument("--dataset-manifest", type=Path, default=None)
    parser.add_argument("--train-shots", type=int, default=None)
    parser.add_argument("--val-shots", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="cuda",
    )
    parser.add_argument("--duration-seconds", type=float, default=1800.0)
    parser.add_argument("--eval-interval-seconds", type=float, default=120.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hypothesis", default="single-host cloud run")
    parser.add_argument("--skip-experiment-log", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args(argv)


def parse_json_output(output: str) -> dict[str, Any]:
    """Decode a script JSON payload from stdout."""

    return json.loads(output.strip())


def parse_result_line(output: str) -> dict[str, Any]:
    """Extract the RESULT payload from train.py stdout."""

    prefix = "RESULT "
    for line in output.splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix) :])
    raise ValueError("train.py output did not include a RESULT line")


def run_subprocess(
    command: list[str],
    repo_root: Path,
) -> subprocess.CompletedProcess[str]:
    """Run one repo subprocess and capture output."""

    return subprocess.run(
        command,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def prepare_dataset(args: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    """Prepare a dataset unless one was provided explicitly."""

    if args.dataset_manifest is not None:
        manifest_path = args.dataset_manifest.resolve()
        return {
            "dataset_id": manifest_path.parent.name,
            "manifest_path": str(manifest_path),
            "profile": args.profile,
            "status": "provided",
        }
    command = [
        sys.executable,
        str(repo_root / "prepare.py"),
        "--workspace",
        str(args.workspace.resolve()),
        "--profile",
        args.profile,
    ]
    if args.train_shots is not None:
        command.extend(["--train-shots", str(args.train_shots)])
    if args.val_shots is not None:
        command.extend(["--val-shots", str(args.val_shots)])
    completed = run_subprocess(command, repo_root)
    return parse_json_output(completed.stdout)


def train_model(
    args: argparse.Namespace,
    repo_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Run train.py and return structured outputs."""

    command = [
        sys.executable,
        str(repo_root / "train.py"),
        "--workspace",
        str(args.workspace.resolve()),
        "--dataset-manifest",
        str(manifest_path),
        "--device",
        args.device,
        "--duration-seconds",
        str(args.duration_seconds),
        "--eval-interval-seconds",
        str(args.eval_interval_seconds),
        "--batch-size",
        str(args.batch_size),
        "--hypothesis",
        args.hypothesis,
    ]
    if args.skip_experiment_log:
        command.append("--skip-experiment-log")
    completed = run_subprocess(command, repo_root)
    result_payload = parse_result_line(completed.stdout)
    metrics_path = (
        args.workspace.resolve()
        / "results"
        / "train"
        / f"{result_payload['run_id']}.json"
    ).resolve()
    best_checkpoint_path = (
        args.workspace.resolve() / "checkpoints" / "best.pt"
    ).resolve()
    return {
        "command": command,
        "result": result_payload,
        "metrics_path": str(metrics_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def eval_model(
    args: argparse.Namespace,
    repo_root: Path,
    manifest_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Run eval.py on the best checkpoint."""

    command = [
        sys.executable,
        str(repo_root / "eval.py"),
        "--workspace",
        str(args.workspace.resolve()),
        "--dataset-manifest",
        str(manifest_path),
        "--checkpoint",
        str(checkpoint_path),
        "--device",
        args.device,
    ]
    if args.skip_plot:
        command.append("--skip-plot")
    completed = run_subprocess(command, repo_root)
    summary = parse_json_output(completed.stdout)
    result_path = (
        args.workspace.resolve() / "results" / "eval" / "best-eval.json"
    ).resolve()
    return {
        "command": command,
        "result_path": str(result_path),
        "summary": summary,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the cloud experiment helper."""

    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    args.workspace.mkdir(parents=True, exist_ok=True)
    prepare_summary = prepare_dataset(args, repo_root)
    manifest_path = Path(str(prepare_summary["manifest_path"])).resolve()
    train_summary = train_model(args, repo_root, manifest_path)
    eval_summary = eval_model(
        args,
        repo_root,
        manifest_path,
        Path(str(train_summary["best_checkpoint_path"])).resolve(),
    )
    payload = {
        "workspace": str(args.workspace.resolve()),
        "manifest_path": str(manifest_path),
        "prepare": prepare_summary,
        "train": train_summary,
        "eval": eval_summary,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
