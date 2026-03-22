"""Stim helpers and detector layout extraction."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pymatching
import stim

DEFAULT_CIRCUIT_NAME = "surface_code:rotated_memory_x"
DEFAULT_PROFILE = "local-d3-v0"
DEFAULT_DISTANCE = 3
DEFAULT_ROUNDS = 3
DEFAULT_P_ERROR = 0.005
DEFAULT_TRAIN_SEED = 20260321
DEFAULT_VAL_SEED = 20260322
DEFAULT_TRAIN_SHOTS = 512
DEFAULT_VAL_SHOTS = 256
DEFAULT_DATASET_ID = "local-d3-v0-d3-r3-p0p005"


def build_circuit(
    distance: int = DEFAULT_DISTANCE,
    rounds: int = DEFAULT_ROUNDS,
    p_error: float = DEFAULT_P_ERROR,
) -> stim.Circuit:
    """Build the fixed v0 surface-code circuit."""

    return stim.Circuit.generated(
        DEFAULT_CIRCUIT_NAME,
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p_error,
        before_round_data_depolarization=p_error,
        after_reset_flip_probability=p_error,
        before_measure_flip_probability=p_error,
    )


def dataset_id_for_v0() -> str:
    """Return the stable v0 dataset id."""

    return DEFAULT_DATASET_ID


def extract_representation_metadata(circuit: stim.Circuit) -> dict[str, Any]:
    """Extract canonical detector metadata for the stored flat representation."""

    coordinates = circuit.get_detector_coordinates()
    ordered_coords = [coordinates[index] for index in range(len(coordinates))]
    time_buckets: dict[float, list[int]] = {}
    for detector_index, raw_coords in enumerate(ordered_coords):
        time_coord = float(raw_coords[2] if len(raw_coords) > 2 else -1.0)
        time_buckets.setdefault(time_coord, []).append(detector_index)
    ordered_time_coords = sorted(time_buckets)
    time_bucket_indices = [time_buckets[time_coord] for time_coord in ordered_time_coords]
    counter = Counter(len(bucket) for bucket in time_bucket_indices)
    return {
        "kind": "flat_with_coordinates",
        "detector_coordinates": [[float(value) for value in coords] for coords in ordered_coords],
        "time_coordinates": ordered_time_coords,
        "time_bucket_indices": time_bucket_indices,
        "time_bucket_sizes": [len(bucket) for bucket in time_bucket_indices],
        "max_time_bucket_size": max(len(bucket) for bucket in time_bucket_indices),
        "time_bucket_size_histogram": {str(size): count for size, count in sorted(counter.items())},
    }


def sample_detection_events(
    circuit: stim.Circuit,
    shots: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample detector events and observable flips deterministically."""

    sampler = circuit.compile_detector_sampler(seed=seed)
    detection_events, observable_flips = sampler.sample(
        shots=shots,
        separate_observables=True,
    )
    return detection_events.astype(np.uint8), observable_flips.astype(np.uint8)


def mwpm_logical_error_rate(
    circuit: stim.Circuit,
    detection_events: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute the MWPM validation baseline."""

    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(detector_error_model)
    predictions = matching.decode_batch(detection_events)
    mismatch = np.any(predictions != labels, axis=1)
    return float(np.mean(mismatch))
