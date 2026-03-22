"""CLI for evaluation against prepared NanoQEC dataset profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional import guard
    plt = None

import torch

from nanoqec.contracts import (
    METRICS_SCHEMA_VERSION,
    DatasetManifest,
    load_checkpoint_metadata,
    write_metrics,
)
from nanoqec.datasets import load_profile_slices
from nanoqec.models import LayoutSpec, build_model
from nanoqec.train_cli import evaluate_profile, resolve_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse evaluation CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--results-dir", type=Path, default=Path("results/eval"))
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args(argv)


def maybe_write_plot(
    per_slice: list[dict[str, Any]],
    output_path: Path,
    skip_plot: bool,
) -> str | None:
    """Optionally render a simple LER-vs-error-rate plot."""

    if skip_plot or plt is None:
        return None
    figure, axis = plt.subplots(figsize=(6, 4))
    p_errors = [item["p_error"] for item in per_slice]
    val_lers = [item["val_ler"] for item in per_slice]
    mwpm_lers = [item["mwpm_val_ler"] for item in per_slice]
    axis.plot(p_errors, val_lers, marker="o", label="Model")
    axis.plot(p_errors, mwpm_lers, marker="s", label="MWPM")
    axis.set_xscale("log")
    axis.set_xlabel("Physical error rate")
    axis.set_ylabel("Validation logical error rate")
    axis.set_title("NanoQEC evaluation profile")
    axis.legend()
    axis.grid(alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return str(output_path)


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate a checkpoint against the validation slices in a prepared profile."""

    manifest_path = args.dataset_manifest.resolve()
    manifest = DatasetManifest.load(manifest_path)
    layout = LayoutSpec.from_manifest(manifest)
    slice_arrays = load_profile_slices(manifest_path, manifest)

    checkpoint_path = args.checkpoint.resolve()
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    metadata = load_checkpoint_metadata(checkpoint_payload["metadata"])
    model = build_model(metadata["model_name"], layout, model_spec=metadata["model_spec"])
    model.load_state_dict(checkpoint_payload["state_dict"])
    device = resolve_device(args.device)
    model.to(device)

    decision_threshold = float(metadata.get("decision_threshold", 0.5))
    aggregate, per_slice = evaluate_profile(
        model,
        slice_arrays,
        device,
        decision_threshold=decision_threshold,
    )
    results_dir = (args.workspace.resolve() / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    eval_id = f"{checkpoint_path.stem}-eval"
    plot_path = maybe_write_plot(
        per_slice=per_slice,
        output_path=results_dir / f"{eval_id}-ler-vs-p.png",
        skip_plot=args.skip_plot,
    )
    result_path = results_dir / f"{eval_id}.json"
    payload = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "run_id": eval_id,
        "dataset_id": manifest.dataset_id,
        "profile": manifest.profile,
        "checkpoint_path": str(checkpoint_path),
        "model_name": metadata["model_name"],
        "model_spec": metadata["model_spec"],
        "decision_threshold": decision_threshold,
        **aggregate,
        "per_slice": per_slice,
        "plot_path": plot_path,
    }
    write_metrics(result_path, payload)
    print(json.dumps(payload, sort_keys=True))
    return {"result_path": str(result_path), "summary": payload}


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation CLI."""

    args = parse_args(argv)
    run_eval(args)
    return 0
