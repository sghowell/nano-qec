"""Shared contracts and JSON helpers for NanoQEC artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DATASET_SCHEMA_VERSION = "nanoqec.dataset.v2"
LEGACY_DATASET_SCHEMA_VERSION = "nanoqec.dataset.v1"
CHECKPOINT_SCHEMA_VERSION = "nanoqec.checkpoint.v2"
LEGACY_CHECKPOINT_SCHEMA_VERSION = "nanoqec.checkpoint.v1"
METRICS_SCHEMA_VERSION = "nanoqec.metrics.v2"
EXPERIMENT_SCHEMA_VERSION = "nanoqec.experiment.v2"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _ensure_keys(name: str, payload: dict[str, Any], required_keys: set[str]) -> None:
    missing = sorted(required_keys - payload.keys())
    if missing:
        raise ValueError(f"{name} is missing required keys: {', '.join(missing)}")


@dataclass(slots=True)
class DatasetArtifact:
    """Dataset split artifact metadata."""

    path: str
    shots: int
    seed: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DatasetArtifact:
        _ensure_keys("dataset artifact", payload, {"path", "shots", "seed"})
        return cls(
            path=str(payload["path"]),
            shots=int(payload["shots"]),
            seed=int(payload["seed"]),
        )


@dataclass(slots=True)
class DatasetSlice:
    """One physical-error slice within a prepared research profile."""

    slice_id: str
    p_error: float
    train: DatasetArtifact
    val: DatasetArtifact
    baselines: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DatasetSlice:
        _ensure_keys("dataset slice", payload, {"slice_id", "p_error", "train", "val", "baselines"})
        return cls(
            slice_id=str(payload["slice_id"]),
            p_error=float(payload["p_error"]),
            train=DatasetArtifact.from_dict(dict(payload["train"])),
            val=DatasetArtifact.from_dict(dict(payload["val"])),
            baselines=dict(payload["baselines"]),
        )


@dataclass(slots=True)
class DatasetManifest:
    """Dataset manifest contract."""

    schema_version: str
    dataset_id: str
    profile: str
    circuit_name: str
    distance: int
    rounds: int
    detector_count: int
    observable_count: int
    representation: dict[str, Any]
    p_error_values: list[float]
    slices: list[DatasetSlice]
    aggregate_baselines: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["slices"] = [
            {
                **asdict(dataset_slice),
                "train": asdict(dataset_slice.train),
                "val": asdict(dataset_slice.val),
            }
            for dataset_slice in self.slices
        ]
        return payload

    def write(self, path: Path) -> None:
        _write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> DatasetManifest:
        payload = _load_json(path)
        schema_version = str(payload.get("schema_version"))
        if schema_version == LEGACY_DATASET_SCHEMA_VERSION:
            return cls._from_legacy_v1(payload)
        required_keys = {
            "schema_version",
            "dataset_id",
            "profile",
            "circuit_name",
            "distance",
            "rounds",
            "detector_count",
            "observable_count",
            "representation",
            "p_error_values",
            "slices",
            "aggregate_baselines",
        }
        _ensure_keys("dataset manifest", payload, required_keys)
        manifest = cls(
            schema_version=str(payload["schema_version"]),
            dataset_id=str(payload["dataset_id"]),
            profile=str(payload["profile"]),
            circuit_name=str(payload["circuit_name"]),
            distance=int(payload["distance"]),
            rounds=int(payload["rounds"]),
            detector_count=int(payload["detector_count"]),
            observable_count=int(payload["observable_count"]),
            representation=dict(payload["representation"]),
            p_error_values=[float(value) for value in payload["p_error_values"]],
            slices=[
                DatasetSlice.from_dict(dict(slice_payload))
                for slice_payload in payload["slices"]
            ],
            aggregate_baselines=dict(payload["aggregate_baselines"]),
        )
        if manifest.schema_version != DATASET_SCHEMA_VERSION:
            raise ValueError(f"unsupported dataset manifest schema: {manifest.schema_version}")
        return manifest

    @classmethod
    def _from_legacy_v1(cls, payload: dict[str, Any]) -> DatasetManifest:
        _ensure_keys(
            "legacy dataset manifest",
            payload,
            {
                "schema_version",
                "dataset_id",
                "profile",
                "circuit_name",
                "distance",
                "rounds",
                "p_error",
                "detector_count",
                "observable_count",
                "representation",
                "splits",
                "baselines",
            },
        )
        splits = dict(payload["splits"])
        dataset_slice = DatasetSlice(
            slice_id="legacy-default",
            p_error=float(payload["p_error"]),
            train=DatasetArtifact.from_dict(dict(splits["train"])),
            val=DatasetArtifact.from_dict(dict(splits["val"])),
            baselines=dict(payload["baselines"]),
        )
        return cls(
            schema_version=DATASET_SCHEMA_VERSION,
            dataset_id=str(payload["dataset_id"]),
            profile=str(payload["profile"]),
            circuit_name=str(payload["circuit_name"]),
            distance=int(payload["distance"]),
            rounds=int(payload["rounds"]),
            detector_count=int(payload["detector_count"]),
            observable_count=int(payload["observable_count"]),
            representation=dict(payload["representation"]),
            p_error_values=[float(payload["p_error"])],
            slices=[dataset_slice],
            aggregate_baselines=dict(payload["baselines"]),
        )

    def slice_by_id(self, slice_id: str) -> DatasetSlice:
        for dataset_slice in self.slices:
            if dataset_slice.slice_id == slice_id:
                return dataset_slice
        raise KeyError(f"unknown dataset slice: {slice_id}")

    def split_path(self, manifest_path: Path, slice_id: str, split_name: str) -> Path:
        dataset_slice = self.slice_by_id(slice_id)
        artifact = dataset_slice.train if split_name == "train" else dataset_slice.val
        return manifest_path.parent / artifact.path


def write_metrics(path: Path, payload: dict[str, Any]) -> None:
    """Persist metrics JSON after checking the schema version."""

    _ensure_keys("metrics", payload, {"schema_version"})
    if payload["schema_version"] != METRICS_SCHEMA_VERSION:
        raise ValueError(f"unexpected metrics schema: {payload['schema_version']}")
    _write_json(path, payload)


def load_checkpoint_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the minimum checkpoint metadata contract."""

    required_keys = {
        "schema_version",
        "model_name",
        "model_spec",
        "train_config",
        "dataset_id",
        "train_seed",
        "git_sha",
    }
    _ensure_keys("checkpoint metadata", payload, required_keys)
    if payload["schema_version"] not in {
        LEGACY_CHECKPOINT_SCHEMA_VERSION,
        CHECKPOINT_SCHEMA_VERSION,
    }:
        raise ValueError(f"unexpected checkpoint schema: {payload['schema_version']}")
    return payload


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append a JSON line to a file, creating parent directories if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file if present."""

    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text().splitlines():
        if raw_line.strip():
            rows.append(json.loads(raw_line))
    return rows
