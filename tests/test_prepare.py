"""Tests for deterministic profile preparation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nanoqec.contracts import DatasetManifest
from nanoqec.prepare_cli import main as prepare_main


def load_npz(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path)
    return {
        "detection_events": loaded["detection_events"],
        "observable_flips": loaded["observable_flips"],
    }


def test_prepare_is_deterministic_across_profile_slices(tmp_path: Path, capsys) -> None:
    argv = [
        "--workspace",
        str(tmp_path),
        "--profile",
        "local-d3-v1",
        "--train-shots",
        "32",
        "--val-shots",
        "16",
    ]
    assert prepare_main(argv) == 0
    first_summary = json.loads(capsys.readouterr().out.strip())
    manifest_path = Path(first_summary["manifest_path"])
    manifest = DatasetManifest.load(manifest_path)
    first_slice = manifest.slices[0]
    first_train = load_npz(manifest.split_path(manifest_path, first_slice.slice_id, "train"))
    first_val = load_npz(manifest.split_path(manifest_path, first_slice.slice_id, "val"))

    assert prepare_main([*argv, "--force"]) == 0
    second_summary = json.loads(capsys.readouterr().out.strip())
    second_manifest_path = Path(second_summary["manifest_path"])
    second_manifest = DatasetManifest.load(second_manifest_path)
    second_slice = second_manifest.slices[0]
    second_train = load_npz(
        second_manifest.split_path(
            second_manifest_path,
            second_slice.slice_id,
            "train",
        )
    )
    second_val = load_npz(
        second_manifest.split_path(
            second_manifest_path,
            second_slice.slice_id,
            "val",
        )
    )

    assert manifest.profile == "local-d3-v1"
    assert manifest.detector_count == 24
    assert manifest.observable_count == 1
    assert manifest.representation["time_bucket_sizes"] == [4, 8, 8, 4]
    assert len(manifest.slices) == 5
    assert np.array_equal(first_train["detection_events"], second_train["detection_events"])
    assert np.array_equal(first_train["observable_flips"], second_train["observable_flips"])
    assert np.array_equal(first_val["detection_events"], second_val["detection_events"])
    assert np.array_equal(first_val["observable_flips"], second_val["observable_flips"])
