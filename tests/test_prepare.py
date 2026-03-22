"""Tests for deterministic dataset preparation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nanoqec.contracts import DatasetManifest
from nanoqec.prepare_cli import main as prepare_main


def load_split(path: Path) -> dict[str, np.ndarray]:
    """Load a split from a compressed numpy archive."""

    loaded = np.load(path)
    return {
        "detection_events": loaded["detection_events"],
        "observable_flips": loaded["observable_flips"],
    }


def test_prepare_is_deterministic(tmp_path: Path, capsys) -> None:
    """Repeated forced preparation with the same seeds should reproduce identical artifacts."""

    argv = [
        "--workspace",
        str(tmp_path),
        "--train-shots",
        "64",
        "--val-shots",
        "32",
    ]
    assert prepare_main(argv) == 0
    first_summary = json.loads(capsys.readouterr().out.strip())
    manifest_path = Path(first_summary["manifest_path"])
    manifest = DatasetManifest.load(manifest_path)
    first_train = load_split(manifest_path.parent / "train.npz")
    first_val = load_split(manifest_path.parent / "val.npz")

    assert prepare_main([*argv, "--force"]) == 0
    second_summary = json.loads(capsys.readouterr().out.strip())
    second_manifest_path = Path(second_summary["manifest_path"])
    second_manifest = DatasetManifest.load(second_manifest_path)
    second_train = load_split(second_manifest_path.parent / "train.npz")
    second_val = load_split(second_manifest_path.parent / "val.npz")

    assert manifest.dataset_id == second_manifest.dataset_id
    assert manifest.detector_count == 24
    assert manifest.observable_count == 1
    assert manifest.representation["time_bucket_sizes"] == [4, 8, 8, 4]
    assert np.array_equal(first_train["detection_events"], second_train["detection_events"])
    assert np.array_equal(first_train["observable_flips"], second_train["observable_flips"])
    assert np.array_equal(first_val["detection_events"], second_val["detection_events"])
    assert np.array_equal(first_val["observable_flips"], second_val["observable_flips"])
