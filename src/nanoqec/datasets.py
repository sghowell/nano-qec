"""Dataset loading helpers for prepared NanoQEC profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nanoqec.contracts import DatasetManifest


@dataclass(slots=True)
class DatasetSliceArrays:
    """In-memory arrays for one prepared dataset slice."""

    slice_id: str
    p_error: float
    train_events: np.ndarray
    train_labels: np.ndarray
    val_events: np.ndarray
    val_labels: np.ndarray
    mwpm_val_ler: float
    train_positive_rate: float
    train_pos_weight: float


def compute_positive_class_weight(labels: np.ndarray) -> tuple[float, float]:
    """Return the positive-label rate and a clipped positive-class weight."""

    labels_1d = labels.reshape(-1).astype(np.float32, copy=False)
    positive_rate = float(labels_1d.mean())
    positive_count = max(int(labels_1d.sum()), 1)
    negative_count = max(int(labels_1d.size) - positive_count, 1)
    pos_weight = min(float(negative_count / positive_count), 32.0)
    return positive_rate, pos_weight


def load_profile_slices(manifest_path: Path, manifest: DatasetManifest) -> list[DatasetSliceArrays]:
    """Load all slice arrays described by a dataset manifest."""

    loaded_slices: list[DatasetSliceArrays] = []
    for dataset_slice in manifest.slices:
        train_npz = np.load(manifest.split_path(manifest_path, dataset_slice.slice_id, "train"))
        val_npz = np.load(manifest.split_path(manifest_path, dataset_slice.slice_id, "val"))
        train_positive_rate, train_pos_weight = compute_positive_class_weight(
            train_npz["observable_flips"]
        )
        loaded_slices.append(
            DatasetSliceArrays(
                slice_id=dataset_slice.slice_id,
                p_error=dataset_slice.p_error,
                train_events=train_npz["detection_events"],
                train_labels=train_npz["observable_flips"],
                val_events=val_npz["detection_events"],
                val_labels=val_npz["observable_flips"],
                mwpm_val_ler=float(dataset_slice.baselines["mwpm_val_ler"]),
                train_positive_rate=train_positive_rate,
                train_pos_weight=train_pos_weight,
            )
        )
    return loaded_slices
