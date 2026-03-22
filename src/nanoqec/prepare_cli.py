"""CLI for deterministic v0 data materialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from nanoqec.contracts import DATASET_SCHEMA_VERSION, DatasetManifest, DatasetSplit
from nanoqec.layout import (
    DEFAULT_CIRCUIT_NAME,
    DEFAULT_DATASET_ID,
    DEFAULT_DISTANCE,
    DEFAULT_P_ERROR,
    DEFAULT_PROFILE,
    DEFAULT_ROUNDS,
    DEFAULT_TRAIN_SEED,
    DEFAULT_TRAIN_SHOTS,
    DEFAULT_VAL_SEED,
    DEFAULT_VAL_SHOTS,
    build_circuit,
    extract_representation_metadata,
    mwpm_logical_error_rate,
    sample_detection_events,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--train-shots", type=int, default=DEFAULT_TRAIN_SHOTS)
    parser.add_argument("--val-shots", type=int, default=DEFAULT_VAL_SHOTS)
    parser.add_argument("--distance", type=int, default=DEFAULT_DISTANCE)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--p-error", type=float, default=DEFAULT_P_ERROR)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--val-seed", type=int, default=DEFAULT_VAL_SEED)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def run_prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Materialize the local v0 dataset and return a JSON summary."""

    if args.distance != DEFAULT_DISTANCE or args.rounds != DEFAULT_ROUNDS:
        raise ValueError("v0 only supports distance=3 and rounds=3")
    if not np.isclose(args.p_error, DEFAULT_P_ERROR):
        raise ValueError("v0 only supports p_error=0.005")

    workspace = args.workspace.resolve()
    dataset_dir = workspace / "data" / DEFAULT_DATASET_ID
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_dir / "manifest.json"
    train_path = dataset_dir / "train.npz"
    val_path = dataset_dir / "val.npz"

    if manifest_path.exists() and not args.force:
        manifest = DatasetManifest.load(manifest_path)
        return {
            "dataset_id": manifest.dataset_id,
            "manifest_path": str(manifest_path),
            "status": "reused",
        }

    circuit = build_circuit(distance=args.distance, rounds=args.rounds, p_error=args.p_error)
    representation = extract_representation_metadata(circuit)
    train_events, train_labels = sample_detection_events(
        circuit=circuit,
        shots=args.train_shots,
        seed=args.train_seed,
    )
    val_events, val_labels = sample_detection_events(
        circuit=circuit,
        shots=args.val_shots,
        seed=args.val_seed,
    )
    mwpm_val_ler = mwpm_logical_error_rate(circuit, val_events, val_labels)

    np.savez_compressed(train_path, detection_events=train_events, observable_flips=train_labels)
    np.savez_compressed(val_path, detection_events=val_events, observable_flips=val_labels)

    manifest = DatasetManifest(
        schema_version=DATASET_SCHEMA_VERSION,
        dataset_id=DEFAULT_DATASET_ID,
        profile=DEFAULT_PROFILE,
        circuit_name=DEFAULT_CIRCUIT_NAME,
        distance=args.distance,
        rounds=args.rounds,
        p_error=args.p_error,
        detector_count=int(train_events.shape[1]),
        observable_count=int(train_labels.shape[1]),
        representation=representation,
        splits={
            "train": DatasetSplit(path="train.npz", shots=args.train_shots, seed=args.train_seed),
            "val": DatasetSplit(path="val.npz", shots=args.val_shots, seed=args.val_seed),
        },
        baselines={"mwpm_val_ler": mwpm_val_ler},
    )
    manifest.write(manifest_path)
    return {
        "dataset_id": manifest.dataset_id,
        "manifest_path": str(manifest_path),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "mwpm_val_ler": mwpm_val_ler,
        "status": "created",
    }


def main(argv: list[str] | None = None) -> int:
    """Run the prepare CLI."""

    args = parse_args(argv)
    summary = run_prepare(args)
    print(json.dumps(summary, sort_keys=True))
    return 0
