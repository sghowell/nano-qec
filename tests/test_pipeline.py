"""End-to-end smoke tests for the local NanoQEC harness."""

from __future__ import annotations

import json
from pathlib import Path

from nanoqec.eval_cli import parse_args as parse_eval_args
from nanoqec.eval_cli import run_eval
from nanoqec.prepare_cli import parse_args as parse_prepare_args
from nanoqec.prepare_cli import run_prepare
from nanoqec.train_cli import parse_args as parse_train_args
from nanoqec.train_cli import run_train


def build_dataset(tmp_path: Path) -> Path:
    prepare_args = parse_prepare_args(
        [
            "--workspace",
            str(tmp_path),
            "--profile",
            "local-d3-v1",
            "--train-shots",
            "48",
            "--val-shots",
            "24",
        ]
    )
    summary = run_prepare(prepare_args)
    return Path(summary["manifest_path"])


def parse_result_line(output: str) -> dict[str, object]:
    prefix = "RESULT "
    for line in output.splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix) :])
    raise AssertionError("missing RESULT line")


def test_train_and_eval_smoke_for_two_models(tmp_path: Path, capsys) -> None:
    manifest_path = build_dataset(tmp_path)
    common_args = [
        "--workspace",
        str(tmp_path),
        "--dataset-manifest",
        str(manifest_path),
        "--duration-seconds",
        "0.4",
        "--eval-interval-seconds",
        "0.2",
        "--batch-size",
        "16",
        "--device",
        "cpu",
        "--skip-experiment-log",
    ]

    baseline_metrics = run_train(parse_train_args(common_args))
    baseline_output = capsys.readouterr().out
    baseline_result = parse_result_line(baseline_output)
    baseline_metrics_payload = json.loads(Path(baseline_metrics["metrics_path"]).read_text())
    assert Path(baseline_metrics["checkpoint_path"]).exists()
    assert Path(baseline_metrics["best_checkpoint_path"]).exists()
    assert Path(baseline_metrics["metrics_path"]).exists()
    assert {"run_id", "val_ler", "mwpm_ratio", "kept"} <= baseline_result.keys()
    assert "decision_threshold" in baseline_metrics_payload

    eval_summary = run_eval(
        parse_eval_args(
            [
                "--workspace",
                str(tmp_path),
                "--dataset-manifest",
                str(manifest_path),
                "--checkpoint",
                baseline_metrics["best_checkpoint_path"],
                "--device",
                "cpu",
            ]
        )
    )
    eval_output = capsys.readouterr().out
    assert Path(eval_summary["result_path"]).exists()
    eval_payload = json.loads(eval_output.strip())
    assert eval_payload["model_name"] == "minimal_aq2"
    assert eval_payload["decision_threshold"] == baseline_metrics_payload["decision_threshold"]
    assert len(eval_payload["per_slice"]) == 5

    alt_metrics = run_train(
        parse_train_args([*common_args, "--model-name", "trivial_linear"])
    )
    alt_output = capsys.readouterr().out
    alt_result = parse_result_line(alt_output)
    assert Path(alt_metrics["checkpoint_path"]).exists()
    assert alt_result["run_id"] != baseline_result["run_id"]

    alt_eval_summary = run_eval(
        parse_eval_args(
            [
                "--workspace",
                str(tmp_path),
                "--dataset-manifest",
                str(manifest_path),
                "--checkpoint",
                alt_metrics["best_checkpoint_path"],
                "--device",
                "cpu",
                "--skip-plot",
            ]
        )
    )
    alt_eval_output = capsys.readouterr().out
    assert Path(alt_eval_summary["result_path"]).exists()
    alt_eval_payload = json.loads(alt_eval_output.strip())
    assert alt_eval_payload["model_name"] == "trivial_linear"
