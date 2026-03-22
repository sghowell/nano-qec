"""CLI for the local NanoQEC training harness."""

from __future__ import annotations

import argparse
import copy
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
from nanoqec.datasets import DatasetSliceArrays, load_profile_slices
from nanoqec.git import current_branch_name, current_git_sha
from nanoqec.models import LayoutSpec, build_model, default_model_spec, parameter_count
from nanoqec.optimizers import Lion

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
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--eval-interval-seconds", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-seed", type=int, default=20260323)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--optimizer", choices=["adamw", "lion"], default="adamw")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/train"))
    parser.add_argument("--experiment-log", type=Path, default=Path("results/experiments.jsonl"))
    parser.add_argument("--hypothesis", default="baseline local research run")
    parser.add_argument("--branch-name", default=None)
    parser.add_argument("--git-sha", default=None)
    parser.add_argument("--skip-experiment-log", action="store_true")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-blocks", type=int, default=None)
    parser.add_argument("--n-transformer-per-block", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--feedforward-mult", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
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


def build_model_spec_from_args(args: argparse.Namespace, layout: LayoutSpec) -> dict[str, Any]:
    """Build a serializable model spec from CLI overrides."""

    model_spec = default_model_spec(args.model_name, layout)
    overrides = {
        "d_model": args.d_model,
        "n_blocks": args.n_blocks,
        "n_transformer_per_block": args.n_transformer_per_block,
        "nhead": args.nhead,
        "dropout": args.dropout,
        "feedforward_mult": args.feedforward_mult,
        "group_size": args.group_size,
    }
    for key, value in overrides.items():
        if value is not None and key in model_spec:
            model_spec[key] = value
    return model_spec


def build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Build the requested optimizer."""

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if optimizer_name == "lion":
        return Lion(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError(f"unsupported optimizer: {optimizer_name}")


def evaluate_binary_logits(logits: Tensor, labels: Tensor) -> tuple[float, float]:
    """Compute logical error rate and accuracy from logits."""

    probs = torch.sigmoid(logits)
    predictions = (probs >= 0.5).to(labels.dtype)
    mismatch = (predictions != labels).to(torch.float32)
    val_ler = float(mismatch.mean().item())
    val_accuracy = float(1.0 - val_ler)
    return val_ler, val_accuracy


def evaluate_profile(
    model: nn.Module,
    slice_arrays: list[DatasetSliceArrays],
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate a model across all prepared slices for a dataset profile."""

    model.eval()
    per_slice: list[dict[str, Any]] = []
    with torch.no_grad():
        for dataset_slice in slice_arrays:
            features = torch.tensor(dataset_slice.val_events, dtype=torch.float32, device=device)
            labels = torch.tensor(
                dataset_slice.val_labels.reshape(-1),
                dtype=torch.float32,
                device=device,
            )
            logits = model(features)
            val_ler, val_accuracy = evaluate_binary_logits(logits, labels)
            mwpm_val_ler = float(dataset_slice.mwpm_val_ler)
            mwpm_ratio = float(val_ler / mwpm_val_ler) if mwpm_val_ler > 0 else float("inf")
            per_slice.append(
                {
                    "slice_id": dataset_slice.slice_id,
                    "p_error": dataset_slice.p_error,
                    "val_ler": val_ler,
                    "val_accuracy": val_accuracy,
                    "mwpm_val_ler": mwpm_val_ler,
                    "mwpm_ratio": mwpm_ratio,
                    "val_shots": int(dataset_slice.val_events.shape[0]),
                }
            )
    aggregate = {
        "aggregate_val_ler": float(np.mean([item["val_ler"] for item in per_slice])),
        "aggregate_val_accuracy": float(np.mean([item["val_accuracy"] for item in per_slice])),
        "aggregate_mwpm_val_ler": float(np.mean([item["mwpm_val_ler"] for item in per_slice])),
    }
    aggregate["aggregate_mwpm_ratio"] = (
        float(aggregate["aggregate_val_ler"] / aggregate["aggregate_mwpm_val_ler"])
        if aggregate["aggregate_mwpm_val_ler"] > 0
        else float("inf")
    )
    return aggregate, per_slice


def sample_train_batch(
    dataset_slice: DatasetSliceArrays,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Sample one batch from a specific slice."""

    effective_batch_size = min(batch_size, len(dataset_slice.train_events))
    indices = rng.integers(0, len(dataset_slice.train_events), size=effective_batch_size)
    batch_x = torch.tensor(dataset_slice.train_events[indices], dtype=torch.float32, device=device)
    batch_y = torch.tensor(
        dataset_slice.train_labels[indices].reshape(-1),
        dtype=torch.float32,
        device=device,
    )
    return batch_x, batch_y


def extract_primary_val_ler(history_row: dict[str, Any]) -> float | None:
    """Extract the primary evaluation metric from an experiment log row."""

    metrics = history_row.get("metrics", {})
    if "aggregate_val_ler" in metrics:
        return float(metrics["aggregate_val_ler"])
    if "val_ler" in metrics:
        return float(metrics["val_ler"])
    return None


def compare_against_history(
    experiment_log_path: Path,
    dataset_id: str,
    candidate_val_ler: float,
) -> bool:
    """Decide whether a run is the best seen so far for this dataset."""

    previous = []
    for row in load_jsonl(experiment_log_path):
        if row.get("dataset_id") != dataset_id:
            continue
        primary_val_ler = extract_primary_val_ler(row)
        if primary_val_ler is not None:
            previous.append(primary_val_ler)
    if not previous:
        return True
    return candidate_val_ler < min(previous)


def build_checkpoint_payload(
    args: argparse.Namespace,
    manifest: DatasetManifest,
    model_name: str,
    model_spec: dict[str, Any],
    device: torch.device,
    best_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build checkpoint metadata."""

    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_name": model_name,
        "model_spec": model_spec,
        "train_config": {
            "duration_seconds": args.duration_seconds,
            "eval_interval_seconds": args.eval_interval_seconds,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "optimizer": args.optimizer,
            "device": str(device),
        },
        "dataset_id": manifest.dataset_id,
        "train_seed": args.train_seed,
        "git_sha": args.git_sha or current_git_sha(),
        "profile": manifest.profile,
        "distance": manifest.distance,
        "rounds": manifest.rounds,
        "best_metrics": best_metrics,
    }


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    """Train a model under the fixed local contract."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    workspace = args.workspace.resolve()
    manifest_path = args.dataset_manifest.resolve()
    manifest = DatasetManifest.load(manifest_path)
    layout = LayoutSpec.from_manifest(manifest)
    model_spec = build_model_spec_from_args(args, layout)
    slice_arrays = load_profile_slices(manifest_path, manifest)

    device = resolve_device(args.device)
    LOGGER.info("training on %s", device)
    torch.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)

    model = build_model(args.model_name, layout, model_spec=model_spec).to(device)
    optimizer = build_optimizer(model, args.optimizer, args.learning_rate, args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.train_seed)

    steps = 0
    examples_seen = 0
    last_loss = 0.0
    last_eval_time = 0.0
    start_time = time.perf_counter()
    best_state_dict = copy.deepcopy(model.state_dict())
    best_aggregate: dict[str, Any] | None = None
    best_per_slice: list[dict[str, Any]] = []

    while True:
        elapsed = time.perf_counter() - start_time
        if steps > 0 and elapsed >= args.duration_seconds:
            break
        selected_slice = slice_arrays[int(rng.integers(0, len(slice_arrays)))]
        batch_x, batch_y = sample_train_batch(selected_slice, args.batch_size, rng, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        loss.backward()
        optimizer.step()

        last_loss = float(loss.item())
        steps += 1
        examples_seen += len(batch_x)

        elapsed = time.perf_counter() - start_time
        if elapsed - last_eval_time >= args.eval_interval_seconds:
            aggregate, per_slice = evaluate_profile(model, slice_arrays, device)
            if best_aggregate is None or (
                aggregate["aggregate_val_ler"] < best_aggregate["aggregate_val_ler"]
            ):
                best_aggregate = aggregate
                best_per_slice = per_slice
                best_state_dict = copy.deepcopy(model.state_dict())
            last_eval_time = elapsed

    final_aggregate, final_per_slice = evaluate_profile(model, slice_arrays, device)
    if best_aggregate is None or (
        final_aggregate["aggregate_val_ler"] < best_aggregate["aggregate_val_ler"]
    ):
        best_aggregate = final_aggregate
        best_per_slice = final_per_slice
        best_state_dict = copy.deepcopy(model.state_dict())

    if best_aggregate is None:
        raise RuntimeError("training produced no evaluation results")

    wall_time = max(time.perf_counter() - start_time, 1e-9)
    run_id = f"{args.model_name}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    checkpoint_dir = (workspace / args.checkpoint_dir).resolve()
    results_dir = (workspace / args.results_dir).resolve()
    experiment_log_path = (workspace / args.experiment_log).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{run_id}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"
    metrics_path = results_dir / f"{run_id}.json"

    checkpoint_metadata = build_checkpoint_payload(
        args=args,
        manifest=manifest,
        model_name=args.model_name,
        model_spec=model_spec,
        device=device,
        best_metrics=best_aggregate,
    )
    torch.save(
        {"metadata": checkpoint_metadata, "state_dict": model.state_dict()},
        checkpoint_path,
    )
    torch.save(
        {"metadata": checkpoint_metadata, "state_dict": best_state_dict},
        best_path,
    )
    shutil.copy2(checkpoint_path, latest_path)

    kept = compare_against_history(
        experiment_log_path,
        manifest.dataset_id,
        best_aggregate["aggregate_val_ler"],
    )
    metrics_payload = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "run_id": run_id,
        "dataset_id": manifest.dataset_id,
        "profile": manifest.profile,
        "distance": manifest.distance,
        "rounds": manifest.rounds,
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
        "optimizer": args.optimizer,
        "parameter_count": parameter_count(model),
        **best_aggregate,
        "per_slice": best_per_slice,
        "kept": kept,
        "artifacts": {
            "checkpoint_path": str(checkpoint_path),
            "latest_checkpoint_path": str(latest_path),
            "best_checkpoint_path": str(best_path),
        },
    }
    write_metrics(metrics_path, metrics_payload)

    if not args.skip_experiment_log:
        experiment_payload = {
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "run_id": run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "dataset_id": manifest.dataset_id,
            "profile": manifest.profile,
            "hypothesis": args.hypothesis,
            "branch_name": args.branch_name or current_branch_name(),
            "git_sha": checkpoint_metadata["git_sha"],
            "model_name": args.model_name,
            "model_spec": model_spec,
            "config_snapshot": checkpoint_metadata["train_config"],
            "metrics": {
                **best_aggregate,
                "per_slice": best_per_slice,
            },
            "artifacts": {
                "checkpoint_path": str(checkpoint_path),
                "best_checkpoint_path": str(best_path),
                "metrics_path": str(metrics_path),
            },
            "kept": kept,
            "wall_time_sec": wall_time,
        }
        append_jsonl(experiment_log_path, experiment_payload)

    result_payload = {
        "run_id": run_id,
        "val_ler": best_aggregate["aggregate_val_ler"],
        "mwpm_ratio": best_aggregate["aggregate_mwpm_ratio"],
        "kept": kept,
    }
    print(f"RESULT {json.dumps(result_payload, sort_keys=True)}")
    return {
        "checkpoint_path": str(checkpoint_path),
        "best_checkpoint_path": str(best_path),
        "metrics_path": str(metrics_path),
        "result": result_payload,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the training CLI."""

    args = parse_args(argv)
    run_train(args)
    return 0
