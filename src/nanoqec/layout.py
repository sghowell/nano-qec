"""Stim helpers and detector layout extraction."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pymatching
import stim

from nanoqec.profiles import DEFAULT_CIRCUIT_NAME


def build_circuit(
    distance: int,
    rounds: int,
    p_error: float,
) -> stim.Circuit:
    """Build a rotated-memory surface-code circuit."""

    return stim.Circuit.generated(
        DEFAULT_CIRCUIT_NAME,
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p_error,
        before_round_data_depolarization=p_error,
        after_reset_flip_probability=p_error,
        before_measure_flip_probability=p_error,
    )


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
    max_x = max(float(coords[0]) for coords in ordered_coords)
    max_y = max(float(coords[1]) for coords in ordered_coords)
    counter = Counter(len(bucket) for bucket in time_bucket_indices)
    return {
        "kind": "flat_with_coordinates",
        "detector_coordinates": [[float(value) for value in coords] for coords in ordered_coords],
        "normalized_xy": [
            [
                float(coords[0]) / max(max_x, 1.0),
                float(coords[1]) / max(max_y, 1.0),
            ]
            for coords in ordered_coords
        ],
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
