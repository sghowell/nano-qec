"""Run repeated fixed-budget training sweeps against the public train.py entrypoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from nanoqec.contracts import DatasetManifest


@dataclass(frozen=True, slots=True)
class ConfigPreset:
    """One named training-config preset for a tuning sweep."""

    name: str
    description: str
    train_args: tuple[str, ...]


CONFIG_PRESETS: dict[str, ConfigPreset] = {
    "baseline": ConfigPreset(
        name="baseline",
        description="Current default minimal AQ2 baseline.",
        train_args=(),
    ),
    "lr1e3": ConfigPreset(
        name="lr1e3",
        description="Higher AdamW learning rate for faster fixed-budget convergence.",
        train_args=("--learning-rate", "1e-3"),
    ),
    "lion": ConfigPreset(
        name="lion",
        description="Lion optimizer with the baseline learning rate.",
        train_args=("--optimizer", "lion"),
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the tuning harness."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument(
        "--config",
        dest="configs",
        action="append",
        choices=sorted(CONFIG_PRESETS),
        help="Named preset to evaluate. Repeat to test multiple configs.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--eval-interval-seconds", type=float, default=5.0)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--results-root", type=Path, default=Path("results/tuning"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/tuning"))
    parser.add_argument("--train-script", type=Path, default=Path("train.py"))
    return parser.parse_args(argv)


def parse_result_line(output: str) -> dict[str, Any]:
    """Extract the machine-readable RESULT payload from train.py stdout."""

    prefix = "RESULT "
    for line in output.splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix) :])
    raise ValueError("train.py output did not include a RESULT line")


def resolve_configs(config_names: list[str] | None) -> list[ConfigPreset]:
    """Return the requested config presets or the default sweep set."""

    names = config_names or ["baseline", "lr1e3", "lion"]
    return [CONFIG_PRESETS[name] for name in names]


def load_primary_metrics(metrics_payload: dict[str, Any]) -> tuple[float, float]:
    """Return the primary profile metrics from one metrics artifact."""

    return (
        float(metrics_payload["aggregate_val_ler"]),
        float(metrics_payload["aggregate_mwpm_ratio"]),
    )


def build_run_command(
    train_script: Path,
    workspace: Path,
    dataset_manifest: Path,
    device: str,
    duration_seconds: float,
    eval_interval_seconds: float,
    checkpoint_dir: Path,
    results_dir: Path,
    hypothesis: str,
    preset: ConfigPreset,
) -> list[str]:
    """Construct one train.py subprocess command."""

    return [
        sys.executable,
        str(train_script),
        "--workspace",
        str(workspace),
        "--dataset-manifest",
        str(dataset_manifest),
        "--device",
        device,
        "--duration-seconds",
        str(duration_seconds),
        "--eval-interval-seconds",
        str(eval_interval_seconds),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--results-dir",
        str(results_dir),
        "--hypothesis",
        hypothesis,
        "--skip-experiment-log",
        *preset.train_args,
    ]


def run_single_training(
    repo_root: Path,
    train_script: Path,
    workspace: Path,
    dataset_manifest: Path,
    preset: ConfigPreset,
    repeat_index: int,
    sweep_id: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run one fixed-budget training job and return structured results."""

    repeat_dir = Path(sweep_id) / preset.name / f"repeat-{repeat_index:02d}"
    checkpoint_dir = (workspace / args.checkpoint_root / repeat_dir).resolve()
    results_dir = (workspace / args.results_root / repeat_dir).resolve()
    command = build_run_command(
        train_script=train_script,
        workspace=workspace,
        dataset_manifest=dataset_manifest,
        device=args.device,
        duration_seconds=args.duration_seconds,
        eval_interval_seconds=args.eval_interval_seconds,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
        hypothesis=f"tuning sweep {sweep_id} preset {preset.name}",
        preset=preset,
    )
    completed = subprocess.run(
        command,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    result_payload = parse_result_line(completed.stdout)
    metrics_path = results_dir / f"{result_payload['run_id']}.json"
    metrics_payload = json.loads(metrics_path.read_text())
    aggregate_val_ler, aggregate_mwpm_ratio = load_primary_metrics(metrics_payload)
    return {
        "repeat_index": repeat_index,
        "run_id": result_payload["run_id"],
        "aggregate_val_ler": aggregate_val_ler,
        "aggregate_mwpm_ratio": aggregate_mwpm_ratio,
        "decision_threshold": float(metrics_payload["decision_threshold"]),
        "steps": int(metrics_payload["steps"]),
        "examples_seen": int(metrics_payload["examples_seen"]),
        "metrics_path": str(metrics_path),
        "best_checkpoint_path": str(checkpoint_dir / "best.pt"),
    }


def summarize_config_runs(preset: ConfigPreset, runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate repeated runs for one config preset."""

    val_lers = [float(run["aggregate_val_ler"]) for run in runs]
    mwpm_ratios = [float(run["aggregate_mwpm_ratio"]) for run in runs]
    best_run = min(runs, key=lambda run: float(run["aggregate_val_ler"]))
    return {
        "config_name": preset.name,
        "description": preset.description,
        "train_args": list(preset.train_args),
        "repeat_count": len(runs),
        "mean_val_ler": mean(val_lers),
        "stdev_val_ler": pstdev(val_lers) if len(val_lers) > 1 else 0.0,
        "mean_mwpm_ratio": mean(mwpm_ratios),
        "stdev_mwpm_ratio": pstdev(mwpm_ratios) if len(mwpm_ratios) > 1 else 0.0,
        "best_val_ler": float(best_run["aggregate_val_ler"]),
        "best_run_id": str(best_run["run_id"]),
        "runs": runs,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the repeated tuning sweep and emit a JSON summary."""

    args = parse_args(argv)
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    repo_root = Path(__file__).resolve().parents[1]
    workspace = args.workspace.resolve()
    dataset_manifest = args.dataset_manifest.resolve()
    manifest = DatasetManifest.load(dataset_manifest)
    train_script = (repo_root / args.train_script).resolve()
    presets = resolve_configs(args.configs)
    sweep_id = f"{manifest.dataset_id}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"

    config_summaries: list[dict[str, Any]] = []
    for preset in presets:
        runs = [
            run_single_training(
                repo_root=repo_root,
                train_script=train_script,
                workspace=workspace,
                dataset_manifest=dataset_manifest,
                preset=preset,
                repeat_index=repeat_index,
                sweep_id=sweep_id,
                args=args,
            )
            for repeat_index in range(1, args.repeats + 1)
        ]
        config_summaries.append(summarize_config_runs(preset, runs))

    ranking = [
        {
            "config_name": summary["config_name"],
            "mean_val_ler": summary["mean_val_ler"],
            "best_val_ler": summary["best_val_ler"],
            "mean_mwpm_ratio": summary["mean_mwpm_ratio"],
        }
        for summary in sorted(config_summaries, key=lambda item: float(item["mean_val_ler"]))
    ]
    summary_payload = {
        "schema_version": "nanoqec.tuning.v1",
        "sweep_id": sweep_id,
        "dataset_id": manifest.dataset_id,
        "dataset_manifest": str(dataset_manifest),
        "device": args.device,
        "duration_seconds": args.duration_seconds,
        "eval_interval_seconds": args.eval_interval_seconds,
        "repeats": args.repeats,
        "configs": config_summaries,
        "ranking": ranking,
    }

    summary_path = (workspace / args.results_root / sweep_id / "summary.json").resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "sweep_id": sweep_id,
                "ranking": ranking,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
