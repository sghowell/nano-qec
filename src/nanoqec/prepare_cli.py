"""CLI for deterministic NanoQEC data materialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from nanoqec.contracts import (
    DATASET_SCHEMA_VERSION,
    DatasetArtifact,
    DatasetManifest,
    DatasetSlice,
)
from nanoqec.layout import (
    build_circuit,
    extract_representation_metadata,
    mwpm_logical_error_rate,
    sample_detection_events,
)
from nanoqec.profiles import (
    DEFAULT_CIRCUIT_NAME,
    DEFAULT_TRAIN_SEED,
    DEFAULT_VAL_SEED,
    available_profile_names,
    dataset_id_for_profile,
    get_profile,
    probability_tag,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--profile", choices=available_profile_names(), default="local-d3-v1")
    parser.add_argument("--train-shots", type=int, default=None)
    parser.add_argument("--val-shots", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def run_prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Materialize a local research profile and return a JSON summary."""

    workspace = args.workspace.resolve()
    profile = get_profile(args.profile)
    train_shots = args.train_shots or profile.default_train_shots
    val_shots = args.val_shots or profile.default_val_shots
    dataset_id = dataset_id_for_profile(profile, train_shots, val_shots)
    dataset_dir = workspace / "data" / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_dir / "manifest.json"

    if manifest_path.exists() and not args.force:
        manifest = DatasetManifest.load(manifest_path)
        return {
            "dataset_id": manifest.dataset_id,
            "manifest_path": str(manifest_path),
            "profile": manifest.profile,
            "slice_count": len(manifest.slices),
            "status": "reused",
        }

    representation: dict[str, Any] | None = None
    dataset_slices: list[DatasetSlice] = []
    mwpm_values: list[float] = []

    for index, p_error in enumerate(profile.p_errors):
        slice_id = probability_tag(p_error)
        train_seed = DEFAULT_TRAIN_SEED + index
        val_seed = DEFAULT_VAL_SEED + index
        circuit = build_circuit(
            distance=profile.distance,
            rounds=profile.rounds,
            p_error=p_error,
        )
        current_representation = extract_representation_metadata(circuit)
        if representation is None:
            representation = current_representation
        train_events, train_labels = sample_detection_events(
            circuit=circuit,
            shots=train_shots,
            seed=train_seed,
        )
        val_events, val_labels = sample_detection_events(
            circuit=circuit,
            shots=val_shots,
            seed=val_seed,
        )
        mwpm_val_ler = mwpm_logical_error_rate(circuit, val_events, val_labels)
        mwpm_values.append(mwpm_val_ler)

        train_path = dataset_dir / f"train_{slice_id}.npz"
        val_path = dataset_dir / f"val_{slice_id}.npz"
        np.savez_compressed(
            train_path,
            detection_events=train_events,
            observable_flips=train_labels,
        )
        np.savez_compressed(
            val_path,
            detection_events=val_events,
            observable_flips=val_labels,
        )

        dataset_slices.append(
            DatasetSlice(
                slice_id=slice_id,
                p_error=p_error,
                train=DatasetArtifact(
                    path=train_path.name,
                    shots=train_shots,
                    seed=train_seed,
                ),
                val=DatasetArtifact(
                    path=val_path.name,
                    shots=val_shots,
                    seed=val_seed,
                ),
                baselines={"mwpm_val_ler": mwpm_val_ler},
            )
        )

    if representation is None:
        raise ValueError("profile produced no dataset slices")

    manifest = DatasetManifest(
        schema_version=DATASET_SCHEMA_VERSION,
        dataset_id=dataset_id,
        profile=profile.name,
        circuit_name=DEFAULT_CIRCUIT_NAME,
        distance=profile.distance,
        rounds=profile.rounds,
        detector_count=int(len(representation["detector_coordinates"])),
        observable_count=1,
        representation=representation,
        p_error_values=[float(value) for value in profile.p_errors],
        slices=dataset_slices,
        aggregate_baselines={"mwpm_val_ler_mean": float(np.mean(mwpm_values))},
    )
    manifest.write(manifest_path)
    return {
        "dataset_id": manifest.dataset_id,
        "manifest_path": str(manifest_path),
        "profile": manifest.profile,
        "slice_count": len(manifest.slices),
        "distance": manifest.distance,
        "rounds": manifest.rounds,
        "status": "created",
    }


def main(argv: list[str] | None = None) -> int:
    """Run the prepare CLI."""

    args = parse_args(argv)
    summary = run_prepare(args)
    print(json.dumps(summary, sort_keys=True))
    return 0
