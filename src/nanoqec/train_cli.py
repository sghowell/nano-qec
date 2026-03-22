"""CLI for the local NanoQEC training harness."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from nanoqec.contracts import (
    CHECKPOINT_SCHEMA_VERSION,
    EXPERIMENT_SCHEMA_VERSION,
    METRICS_SCHEMA_VERSION,
    DatasetManifest,
    append_jsonl,
    load_jsonl,
    write_metrics,
)
from nanoqec.git import current_branch_name, current_git_sha
from nanoqec.models import LayoutSpec, build_model, default_model_spec

LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for training."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument(
        "--model-name",
        choices=["minimal_aq2", "trivial_linear"],
        default="minimal_aq2",
    )
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-seed", type=int, default=20260323)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/train"))
    parser.add_argument("--experiment-log", type=Path, default=Path("results/experiments.jsonl"))
    parser.add_argument("--hypothesis", default="baseline smoke run")
    parser.add_argument("--branch-name", default=None)
    parser.add_argument("--git-sha", default=None)
    parser.add_argument("--skip-experiment-log", action="store_true")
    return parser.parse_args(argv)


def resolve_device(choice: str) -> torch.device:
    """Resolve a runtime device from CLI input."""

    if choice == "cpu":
        return torch.device("cpu")
    if choice == "mps":
        return torch.device("mps")
    if choice == "cuda":
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_split(
    manifest_path: Path,
    manifest: DatasetManifest,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a dataset split from the manifest."""

    split_path = manifest.split_path(manifest_path, split_name)
    loaded = np.load(split_path)
    return loaded["detection_events"], loaded["observable_flips"]


def evaluate_model(model: nn.Module, features: Tensor, labels: Tensor) -> tuple[float, float]:
    """Compute validation logical error rate and accuracy."""

    model.eval()
    with torch.no_grad():
        logits = model(features)
        probs = torch.sigmoid(logits)
        predictions = (probs >= 0.5).to(labels.dtype)
        mismatch = (predictions != labels).to(torch.float32)
        val_ler = float(mismatch.mean().item())
        val_accuracy = float(1.0 - val_ler)
    return val_ler, val_accuracy


def compare_against_history(
    experiment_log_path: Path,
    dataset_id: str,
    candidate_val_ler: float,
) -> bool:
    """Decide whether a run is the best seen so far for this dataset."""

    previous = [
        row
        for row in load_jsonl(experiment_log_path)
        if row.get("dataset_id") == dataset_id and "metrics" in row
    ]
    if not previous:
        return True
    previous_best = min(float(row["metrics"]["val_ler"]) for row in previous)
    return candidate_val_ler < previous_best


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    """Train a model under the fixed v0 contract."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    workspace = args.workspace.resolve()
    manifest_path = args.dataset_manifest.resolve()
    manifest = DatasetManifest.load(manifest_path)
    layout = LayoutSpec.from_manifest(manifest)
    model_spec = default_model_spec(args.model_name, layout)

    train_events, train_labels = load_split(manifest_path, manifest, "train")
    val_events, val_labels = load_split(manifest_path, manifest, "val")

    device = resolve_device(args.device)
    LOGGER.info("training on %s", device)
    torch.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)

    model = build_model(args.model_name, layout, model_spec=model_spec).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    train_x = torch.tensor(train_events, dtype=torch.float32, device=device)
    train_y = torch.tensor(train_labels.reshape(-1), dtype=torch.float32, device=device)
    val_x = torch.tensor(val_events, dtype=torch.float32, device=device)
    val_y = torch.tensor(val_labels.reshape(-1), dtype=torch.float32, device=device)

    rng = np.random.default_rng(args.train_seed)
    steps = 0
    examples_seen = 0
    last_loss = 0.0
    start_time = time.perf_counter()

    while True:
        if steps > 0 and (time.perf_counter() - start_time) >= args.duration_seconds:
            break
        batch_size = min(args.batch_size, len(train_x))
        indices = rng.integers(0, len(train_x), size=batch_size)
        batch_index = torch.tensor(indices, dtype=torch.long, device=device)
        batch_x = train_x.index_select(0, batch_index)
        batch_y = train_y.index_select(0, batch_index)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        loss.backward()
        optimizer.step()

        last_loss = float(loss.item())
        steps += 1
        examples_seen += batch_size

    wall_time = max(time.perf_counter() - start_time, 1e-9)
    val_ler, val_accuracy = evaluate_model(model, val_x, val_y)
    mwpm_val_ler = float(manifest.baselines["mwpm_val_ler"])
    mwpm_ratio = float(val_ler / mwpm_val_ler) if mwpm_val_ler > 0 else float("inf")

    run_id = f"{args.model_name}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    checkpoint_dir = (workspace / args.checkpoint_dir).resolve()
    results_dir = (workspace / args.results_dir).resolve()
    experiment_log_path = (workspace / args.experiment_log).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{run_id}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    metrics_path = results_dir / f"{run_id}.json"

    checkpoint_metadata = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_name": args.model_name,
        "model_spec": model_spec,
        "train_config": {
            "duration_seconds": args.duration_seconds,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "device": str(device),
        },
        "dataset_id": manifest.dataset_id,
        "train_seed": args.train_seed,
        "git_sha": args.git_sha or current_git_sha(),
    }
    torch.save({"metadata": checkpoint_metadata, "state_dict": model.state_dict()}, checkpoint_path)
    shutil.copy2(checkpoint_path, latest_path)

    kept = compare_against_history(experiment_log_path, manifest.dataset_id, val_ler)
    metrics_payload = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "run_id": run_id,
        "dataset_id": manifest.dataset_id,
        "model_name": args.model_name,
        "model_spec": model_spec,
        "device": str(device),
        "train_seed": args.train_seed,
        "duration_seconds_requested": args.duration_seconds,
        "duration_seconds_actual": wall_time,
        "steps": steps,
        "examples_seen": examples_seen,
        "throughput_samples_per_second": examples_seen / wall_time,
        "train_loss_last": last_loss,
        "val_ler": val_ler,
        "val_accuracy": val_accuracy,
        "mwpm_val_ler": mwpm_val_ler,
        "mwpm_ratio": mwpm_ratio,
        "kept": kept,
        "artifacts": {
            "checkpoint_path": str(checkpoint_path),
            "latest_checkpoint_path": str(latest_path),
        },
    }
    write_metrics(metrics_path, metrics_payload)

    if not args.skip_experiment_log:
        experiment_payload = {
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "run_id": run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "dataset_id": manifest.dataset_id,
            "hypothesis": args.hypothesis,
            "branch_name": args.branch_name or current_branch_name(),
            "git_sha": checkpoint_metadata["git_sha"],
            "model_name": args.model_name,
            "model_spec": model_spec,
            "config_snapshot": checkpoint_metadata["train_config"],
            "metrics": {
                "val_ler": val_ler,
                "val_accuracy": val_accuracy,
                "mwpm_val_ler": mwpm_val_ler,
                "mwpm_ratio": mwpm_ratio,
            },
            "artifacts": {
                "checkpoint_path": str(checkpoint_path),
                "metrics_path": str(metrics_path),
            },
            "kept": kept,
            "wall_time_sec": wall_time,
        }
        append_jsonl(experiment_log_path, experiment_payload)

    result_payload = {
        "run_id": run_id,
        "val_ler": val_ler,
        "mwpm_ratio": mwpm_ratio,
        "kept": kept,
    }
    print(f"RESULT {json.dumps(result_payload, sort_keys=True)}")
    return {
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "result": result_payload,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the training CLI."""

    args = parse_args(argv)
    run_train(args)
    return 0
