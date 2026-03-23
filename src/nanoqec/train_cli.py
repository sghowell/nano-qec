"""CLI for the local NanoQEC training harness."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

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
    parser.add_argument("--duration-seconds", type=float, default=120.0)
    parser.add_argument("--eval-interval-seconds", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-seed", type=int, default=20260323)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--optimizer", choices=["adamw", "lion"], default="lion")
    parser.add_argument(
        "--scheduler",
        choices=["constant", "warmup_cosine"],
        default="warmup_cosine",
    )
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--min-learning-rate-scale", type=float, default=0.1)
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


def validate_scheduler_args(args: argparse.Namespace) -> None:
    """Validate CLI scheduler controls."""

    if args.warmup_fraction < 0.0 or args.warmup_fraction >= 1.0:
        raise ValueError("--warmup-fraction must be in [0.0, 1.0)")
    if args.min_learning_rate_scale < 0.0 or args.min_learning_rate_scale > 1.0:
        raise ValueError("--min-learning-rate-scale must be in [0.0, 1.0]")


def compute_learning_rate_scale(
    elapsed_seconds: float,
    duration_seconds: float,
    scheduler_name: str,
    warmup_fraction: float,
    min_learning_rate_scale: float,
) -> float:
    """Compute the current learning-rate scale from the wall-clock schedule."""

    if scheduler_name == "constant":
        return 1.0
    if scheduler_name != "warmup_cosine":
        raise ValueError(f"unsupported scheduler: {scheduler_name}")
    if duration_seconds <= 0.0:
        return 1.0
    progress = min(max(elapsed_seconds / duration_seconds, 0.0), 1.0)
    if warmup_fraction > 0.0 and progress < warmup_fraction:
        warmup_progress = progress / warmup_fraction
        return max(1e-3, float(warmup_progress))
    decay_progress = (
        1.0
        if warmup_fraction >= 1.0
        else (progress - warmup_fraction) / max(1.0 - warmup_fraction, 1e-9)
    )
    cosine_scale = 0.5 * (1.0 + math.cos(math.pi * min(max(decay_progress, 0.0), 1.0)))
    return float(min_learning_rate_scale + (1.0 - min_learning_rate_scale) * cosine_scale)


def set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    base_learning_rate: float,
    scale: float,
) -> float:
    """Apply a scheduled learning rate to every optimizer parameter group."""

    current_learning_rate = float(base_learning_rate * scale)
    for group in optimizer.param_groups:
        group["lr"] = current_learning_rate
    return current_learning_rate


def evaluate_binary_logits(logits: Tensor, labels: Tensor) -> tuple[float, float]:
    """Compute logical error rate and accuracy from logits."""

    return evaluate_binary_logits_with_threshold(logits, labels, decision_threshold=0.5)


def evaluate_binary_logits_with_threshold(
    logits: Tensor,
    labels: Tensor,
    decision_threshold: float,
) -> tuple[float, float]:
    """Compute logical error rate and accuracy from logits at a fixed threshold."""

    probs = torch.sigmoid(logits)
    predictions = (probs >= decision_threshold).to(labels.dtype)
    mismatch = (predictions != labels).to(torch.float32)
    val_ler = float(mismatch.mean().item())
    val_accuracy = float(1.0 - val_ler)
    return val_ler, val_accuracy


def build_p_error_batch(
    p_error: float,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """Build one physical-error-rate conditioning tensor for a batch."""

    return torch.full((batch_size,), float(p_error), dtype=torch.float32, device=device)


def forward_model(
    model: nn.Module,
    detector_events: Tensor,
    p_error: Tensor,
) -> Tensor:
    """Run a decoder with the internal conditioning metadata it expects."""

    return model(detector_events, p_error=p_error)


def balanced_bce_loss(
    logits: Tensor,
    labels: Tensor,
    pos_weight: float,
) -> Tensor:
    """Blend standard BCE with a slice-balanced positive-class weighting."""

    unweighted = F.binary_cross_entropy_with_logits(logits, labels)
    weighted = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=torch.tensor(pos_weight, dtype=logits.dtype, device=logits.device),
    )
    return 0.5 * (unweighted + weighted)


def compute_slice_sampling_weights(
    per_slice: list[dict[str, Any]],
    num_slices: int,
) -> np.ndarray:
    """Compute slice sampling weights from per-slice MWPM ratios.

    Slices with higher MWPM ratio (worse relative to MWPM) get more training
    time. Uses softmax over clipped log-ratios for smooth, bounded weighting.
    Falls back to uniform when no per-slice data is available.
    """
    if not per_slice or len(per_slice) != num_slices:
        return np.ones(num_slices, dtype=np.float64) / num_slices
    ratios = np.array(
        [max(s.get("mwpm_ratio", 1.0), 1.0) for s in per_slice],
        dtype=np.float64,
    )
    finite_mask = np.isfinite(ratios)
    if finite_mask.any():
        max_finite = ratios[finite_mask].max()
        ratios[~finite_mask] = max_finite
    log_ratios = np.log(np.clip(ratios, 1.0, 50.0))
    temperature = 1.0
    shifted = log_ratios / temperature
    shifted -= shifted.max()
    weights = np.exp(shifted)
    weights /= weights.sum()
    uniform = np.ones(num_slices, dtype=np.float64) / num_slices
    blend_alpha = 0.5
    blended = blend_alpha * weights + (1.0 - blend_alpha) * uniform
    blended /= blended.sum()
    return blended


def focal_bce_loss(
    logits: Tensor,
    labels: Tensor,
    pos_weight: float,
    gamma: float = 2.0,
) -> Tensor:
    """Focal loss variant of balanced BCE that down-weights easy examples.

    Focuses training on hard-to-classify samples, which helps mid-range
    error rate slices where confident wrong predictions waste gradient signal.
    """
    probs = torch.sigmoid(logits)
    p_t = probs * labels + (1.0 - probs) * (1.0 - labels)
    focal_weight = (1.0 - p_t) ** gamma
    unweighted = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    weighted = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=torch.tensor(pos_weight, dtype=logits.dtype, device=logits.device),
        reduction="none",
    )
    base_loss = 0.5 * (unweighted + weighted)
    return (focal_weight.detach() * base_loss).mean()


def select_decision_threshold(
    model: nn.Module,
    slice_arrays: list[DatasetSliceArrays],
    device: torch.device,
) -> float:
    """Choose a global threshold that minimizes train-split aggregate LER."""

    candidate_thresholds = np.linspace(0.05, 0.95, 19)
    best_threshold = 0.5
    best_ler = float("inf")
    model.eval()
    with torch.no_grad():
        for threshold in candidate_thresholds:
            slice_lers: list[float] = []
            for dataset_slice in slice_arrays:
                features = torch.tensor(
                    dataset_slice.train_events,
                    dtype=torch.float32,
                    device=device,
                )
                labels = torch.tensor(
                    dataset_slice.train_labels.reshape(-1),
                    dtype=torch.float32,
                    device=device,
                )
                p_error = build_p_error_batch(dataset_slice.p_error, features.shape[0], device)
                logits = forward_model(model, features, p_error)
                train_ler, _ = evaluate_binary_logits_with_threshold(
                    logits,
                    labels,
                    decision_threshold=float(threshold),
                )
                slice_lers.append(train_ler)
            aggregate_ler = float(np.mean(slice_lers))
            if aggregate_ler < best_ler:
                best_ler = aggregate_ler
                best_threshold = float(threshold)
    return best_threshold


def evaluate_profile(
    model: nn.Module,
    slice_arrays: list[DatasetSliceArrays],
    device: torch.device,
    decision_threshold: float = 0.5,
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
            p_error = build_p_error_batch(dataset_slice.p_error, features.shape[0], device)
            logits = forward_model(model, features, p_error)
            val_ler, val_accuracy = evaluate_binary_logits_with_threshold(
                logits,
                labels,
                decision_threshold=decision_threshold,
            )
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
    decision_threshold: float,
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
            "scheduler": args.scheduler,
            "warmup_fraction": args.warmup_fraction,
            "min_learning_rate_scale": args.min_learning_rate_scale,
            "device": str(device),
        },
        "dataset_id": manifest.dataset_id,
        "train_seed": args.train_seed,
        "git_sha": args.git_sha or current_git_sha(),
        "decision_threshold": decision_threshold,
        "profile": manifest.profile,
        "distance": manifest.distance,
        "rounds": manifest.rounds,
        "best_metrics": best_metrics,
    }


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    """Train a model under the fixed local contract."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    workspace = args.workspace.resolve()
    validate_scheduler_args(args)
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
    rng = np.random.default_rng(args.train_seed)

    steps = 0
    examples_seen = 0
    last_loss = 0.0
    last_eval_time = 0.0
    current_learning_rate = args.learning_rate
    start_time = time.perf_counter()
    best_state_dict = copy.deepcopy(model.state_dict())
    best_selection_aggregate: dict[str, Any] | None = None
    eval_history: list[dict[str, Any]] = []
    num_slices = len(slice_arrays)
    p_values = [dataset_slice.p_error for dataset_slice in slice_arrays]
    p_weights = np.array(p_values, dtype=np.float64)
    p_weights = p_weights / p_weights.sum()
    uniform = np.ones(num_slices, dtype=np.float64) / num_slices
    slice_weights = 0.6 * p_weights + 0.4 * uniform
    slice_weights = slice_weights / slice_weights.sum()

    while True:
        elapsed = time.perf_counter() - start_time
        if steps > 0 and elapsed >= args.duration_seconds:
            break
        current_learning_rate = set_optimizer_learning_rate(
            optimizer,
            args.learning_rate,
            compute_learning_rate_scale(
                elapsed_seconds=elapsed,
                duration_seconds=args.duration_seconds,
                scheduler_name=args.scheduler,
                warmup_fraction=args.warmup_fraction,
                min_learning_rate_scale=args.min_learning_rate_scale,
            ),
        )
        slice_idx = int(rng.choice(num_slices, p=slice_weights))
        selected_slice = slice_arrays[slice_idx]
        batch_x, batch_y = sample_train_batch(selected_slice, args.batch_size, rng, device)
        batch_p_error = build_p_error_batch(selected_slice.p_error, len(batch_x), device)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_model(model, batch_x, batch_p_error)
        loss = focal_bce_loss(logits, batch_y, selected_slice.train_pos_weight)
        loss.backward()
        optimizer.step()

        last_loss = float(loss.item())
        steps += 1
        examples_seen += len(batch_x)

        elapsed = time.perf_counter() - start_time
        if elapsed - last_eval_time >= args.eval_interval_seconds:
            aggregate, per_slice = evaluate_profile(
                model,
                slice_arrays,
                device,
            )
            eval_history.append(
                {
                    "elapsed_seconds": elapsed,
                    "steps": steps,
                    "examples_seen": examples_seen,
                    "learning_rate": current_learning_rate,
                    "aggregate_val_ler": aggregate["aggregate_val_ler"],
                    "aggregate_mwpm_ratio": aggregate["aggregate_mwpm_ratio"],
                    "aggregate_val_accuracy": aggregate["aggregate_val_accuracy"],
                }
            )
            if best_selection_aggregate is None or (
                aggregate["aggregate_val_ler"] < best_selection_aggregate["aggregate_val_ler"]
            ):
                best_selection_aggregate = aggregate
                best_state_dict = copy.deepcopy(model.state_dict())
            last_eval_time = elapsed

    final_aggregate, final_per_slice = evaluate_profile(
        model,
        slice_arrays,
        device,
    )
    final_elapsed = time.perf_counter() - start_time
    eval_history.append(
        {
            "elapsed_seconds": final_elapsed,
            "steps": steps,
            "examples_seen": examples_seen,
            "learning_rate": current_learning_rate,
            "aggregate_val_ler": final_aggregate["aggregate_val_ler"],
            "aggregate_mwpm_ratio": final_aggregate["aggregate_mwpm_ratio"],
            "aggregate_val_accuracy": final_aggregate["aggregate_val_accuracy"],
        }
    )
    if best_selection_aggregate is None or (
        final_aggregate["aggregate_val_ler"] < best_selection_aggregate["aggregate_val_ler"]
    ):
        best_selection_aggregate = final_aggregate
        best_state_dict = copy.deepcopy(model.state_dict())

    if best_selection_aggregate is None:
        raise RuntimeError("training produced no evaluation results")

    model.load_state_dict(best_state_dict)
    best_decision_threshold = select_decision_threshold(model, slice_arrays, device)
    best_aggregate, best_per_slice = evaluate_profile(
        model,
        slice_arrays,
        device,
        decision_threshold=best_decision_threshold,
    )

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
        decision_threshold=best_decision_threshold,
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
        "scheduler": args.scheduler,
        "warmup_fraction": args.warmup_fraction,
        "min_learning_rate_scale": args.min_learning_rate_scale,
        "decision_threshold": best_decision_threshold,
        "parameter_count": parameter_count(model),
        "eval_history": eval_history,
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
