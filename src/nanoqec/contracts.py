"""Shared contracts and JSON helpers for NanoQEC artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DATASET_SCHEMA_VERSION = "nanoqec.dataset.v1"
CHECKPOINT_SCHEMA_VERSION = "nanoqec.checkpoint.v1"
METRICS_SCHEMA_VERSION = "nanoqec.metrics.v1"
EXPERIMENT_SCHEMA_VERSION = "nanoqec.experiment.v1"


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
class DatasetSplit:
    """Dataset split artifact metadata."""

    path: str
    shots: int
    seed: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DatasetSplit:
        _ensure_keys("dataset split", payload, {"path", "shots", "seed"})
        return cls(
            path=str(payload["path"]),
            shots=int(payload["shots"]),
            seed=int(payload["seed"]),
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
    p_error: float
    detector_count: int
    observable_count: int
    representation: dict[str, Any]
    splits: dict[str, DatasetSplit]
    baselines: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["splits"] = {name: asdict(split) for name, split in self.splits.items()}
        return payload

    def write(self, path: Path) -> None:
        _write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> DatasetManifest:
        payload = _load_json(path)
        required_keys = {
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
        }
        _ensure_keys("dataset manifest", payload, required_keys)
        splits = {
            split_name: DatasetSplit.from_dict(split_payload)
            for split_name, split_payload in payload["splits"].items()
        }
        manifest = cls(
            schema_version=str(payload["schema_version"]),
            dataset_id=str(payload["dataset_id"]),
            profile=str(payload["profile"]),
            circuit_name=str(payload["circuit_name"]),
            distance=int(payload["distance"]),
            rounds=int(payload["rounds"]),
            p_error=float(payload["p_error"]),
            detector_count=int(payload["detector_count"]),
            observable_count=int(payload["observable_count"]),
            representation=dict(payload["representation"]),
            splits=splits,
            baselines=dict(payload["baselines"]),
        )
        if manifest.schema_version != DATASET_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported dataset manifest schema: {manifest.schema_version}"
            )
        return manifest

    def split_path(self, manifest_path: Path, split_name: str) -> Path:
        split = self.splits[split_name]
        return manifest_path.parent / split.path


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
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
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
