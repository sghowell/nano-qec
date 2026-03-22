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


def load_profile_slices(manifest_path: Path, manifest: DatasetManifest) -> list[DatasetSliceArrays]:
    """Load all slice arrays described by a dataset manifest."""

    loaded_slices: list[DatasetSliceArrays] = []
    for dataset_slice in manifest.slices:
        train_npz = np.load(manifest.split_path(manifest_path, dataset_slice.slice_id, "train"))
        val_npz = np.load(manifest.split_path(manifest_path, dataset_slice.slice_id, "val"))
        loaded_slices.append(
            DatasetSliceArrays(
                slice_id=dataset_slice.slice_id,
                p_error=dataset_slice.p_error,
                train_events=train_npz["detection_events"],
                train_labels=train_npz["observable_flips"],
                val_events=val_npz["detection_events"],
                val_labels=val_npz["observable_flips"],
                mwpm_val_ler=float(dataset_slice.baselines["mwpm_val_ler"]),
            )
        )
    return loaded_slices
