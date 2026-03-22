"""Tests for repo-local experiment utilities."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_script(script_path: Path, *args: str) -> dict[str, object]:
    """Execute a repo script with the current interpreter and parse JSON stdout."""

    completed = subprocess.run(
        [sys.executable, str(script_path), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip())


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
