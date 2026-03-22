"""CLI for evaluation against the fixed local dataset contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from nanoqec.contracts import (
    METRICS_SCHEMA_VERSION,
    DatasetManifest,
    load_checkpoint_metadata,
    write_metrics,
)
from nanoqec.models import LayoutSpec, build_model
from nanoqec.train_cli import evaluate_model, load_split, resolve_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse evaluation CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--results-dir", type=Path, default=Path("results/eval"))
    return parser.parse_args(argv)


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate a checkpoint against the validation split."""

    manifest_path = args.dataset_manifest.resolve()
    manifest = DatasetManifest.load(manifest_path)
    layout = LayoutSpec.from_manifest(manifest)
    val_events, val_labels = load_split(manifest_path, manifest, "val")

    checkpoint_path = args.checkpoint.resolve()
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    metadata = load_checkpoint_metadata(checkpoint_payload["metadata"])
    model = build_model(metadata["model_name"], layout, model_spec=metadata["model_spec"])
    model.load_state_dict(checkpoint_payload["state_dict"])
    device = resolve_device(args.device)
    model.to(device)

    val_x = torch.tensor(val_events, dtype=torch.float32, device=device)
    val_y = torch.tensor(val_labels.reshape(-1), dtype=torch.float32, device=device)
    val_ler, val_accuracy = evaluate_model(model, val_x, val_y)
    mwpm_val_ler = float(manifest.baselines["mwpm_val_ler"])
    mwpm_ratio = float(val_ler / mwpm_val_ler) if mwpm_val_ler > 0 else float("inf")

    results_dir = (args.workspace.resolve() / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    eval_id = f"{checkpoint_path.stem}-eval"
    result_path = results_dir / f"{eval_id}.json"
    payload = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "run_id": eval_id,
        "dataset_id": manifest.dataset_id,
        "checkpoint_path": str(checkpoint_path),
        "model_name": metadata["model_name"],
        "model_spec": metadata["model_spec"],
        "val_ler": val_ler,
        "val_accuracy": val_accuracy,
        "mwpm_val_ler": mwpm_val_ler,
        "mwpm_ratio": mwpm_ratio,
        "per_slice": [
            {
                "profile": manifest.profile,
                "distance": manifest.distance,
                "rounds": manifest.rounds,
                "p_error": manifest.p_error,
                "val_ler": val_ler,
                "mwpm_val_ler": mwpm_val_ler,
            }
        ],
    }
    write_metrics(result_path, payload)
    print(json.dumps(payload, sort_keys=True))
    return {"result_path": str(result_path), "summary": payload}


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation CLI."""

    args = parse_args(argv)
    run_eval(args)
    return 0
