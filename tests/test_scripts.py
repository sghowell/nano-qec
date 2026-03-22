"""Tests for repo-local experiment utilities."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from nanoqec.prepare_cli import parse_args as parse_prepare_args
from nanoqec.prepare_cli import run_prepare


def run_script(script_path: Path, *args: str) -> dict[str, object]:
    """Execute a repo script with the current interpreter and parse JSON stdout."""

    completed = subprocess.run(
        [sys.executable, str(script_path), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip())


def build_dataset(tmp_path: Path) -> Path:
    """Prepare one tiny deterministic profile for script tests."""

    prepare_args = parse_prepare_args(
        [
            "--workspace",
            str(tmp_path),
            "--profile",
            "local-d3-v1",
            "--train-shots",
            "32",
            "--val-shots",
            "16",
        ]
    )
    summary = run_prepare(prepare_args)
    return Path(summary["manifest_path"])


def test_scripts_operate_on_runtime_artifacts(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "dataset_id": "demo",
                "aggregate_val_ler": 0.2,
                "schema_version": "nanoqec.metrics.v2",
            }
        )
    )
    experiment_log = tmp_path / "experiments.jsonl"
    experiment_log.write_text(
        json.dumps(
            {
                "dataset_id": "demo",
                "metrics": {"aggregate_val_ler": 0.3, "aggregate_mwpm_ratio": 1.2},
                "run_id": "old",
            }
        )
        + "\n"
        + json.dumps(
            {
                "dataset_id": "demo",
                "metrics": {"aggregate_val_ler": 0.25, "aggregate_mwpm_ratio": 1.1},
                "run_id": "newer",
            }
        )
        + "\n"
    )

    repo_root = Path(__file__).resolve().parents[1]
    improvement_output = run_script(
        repo_root / "scripts" / "check_improvement.py",
        "--metrics-json",
        str(metrics_path),
        "--experiment-log",
        str(experiment_log),
    )
    assert improvement_output["improved"] is True

    output_path = tmp_path / "progress.png"
    plot_output = run_script(
        repo_root / "scripts" / "plot_progress.py",
        "--experiment-log",
        str(experiment_log),
        "--output",
        str(output_path),
    )
    assert plot_output["points"] == 2
    assert output_path.exists()


def test_tuning_script_runs_one_repeat(tmp_path: Path) -> None:
    manifest_path = build_dataset(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    tuning_output = run_script(
        repo_root / "scripts" / "tune_profile.py",
        "--workspace",
        str(tmp_path),
        "--dataset-manifest",
        str(manifest_path),
        "--config",
        "warmup_cosine",
        "--repeats",
        "1",
        "--duration-seconds",
        "0.1",
        "--duration-seconds",
        "0.2",
        "--eval-interval-seconds",
        "0.1",
        "--device",
        "cpu",
    )
    summary_path = Path(str(tuning_output["summary_path"]))
    assert summary_path.exists()
    summary_payload = json.loads(summary_path.read_text())
    assert summary_payload["schema_version"] == "nanoqec.tuning.v1"
    assert summary_payload["duration_seconds_sweep"] == [0.1, 0.2]
    assert [config["config_name"] for config in summary_payload["configs"]] == [
        "warmup_cosine",
        "warmup_cosine",
    ]
    assert [config["duration_seconds"] for config in summary_payload["configs"]] == [0.1, 0.2]
    assert all(config["repeat_count"] == 1 for config in summary_payload["configs"])
    assert all(len(config["runs"]) == 1 for config in summary_payload["configs"])
    assert [config["runs"][0]["train_seed"] for config in summary_payload["configs"]] == [
        20260323,
        20260323,
    ]
