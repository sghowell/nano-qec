"""Named local research profiles for NanoQEC data preparation."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CIRCUIT_NAME = "surface_code:rotated_memory_x"
DEFAULT_TRAIN_SEED = 20260321
DEFAULT_VAL_SEED = 20260322


@dataclass(frozen=True, slots=True)
class ProfileSpec:
    """A deterministic local data-preparation profile."""

    name: str
    distance: int
    rounds: int
    p_errors: tuple[float, ...]
    default_train_shots: int
    default_val_shots: int
    description: str


PROFILES: dict[str, ProfileSpec] = {
    "local-d3-v1": ProfileSpec(
        name="local-d3-v1",
        distance=3,
        rounds=3,
        p_errors=(0.001, 0.003, 0.005, 0.007, 0.01),
        default_train_shots=8192,
        default_val_shots=256,
        description="Local d=3 research profile with a five-rate validation sweep.",
    ),
    "local-d5-v1": ProfileSpec(
        name="local-d5-v1",
        distance=5,
        rounds=5,
        p_errors=(0.001, 0.003, 0.005, 0.007, 0.01),
        default_train_shots=512,
        default_val_shots=256,
        description="Local d=5 research profile with the same five-rate sweep.",
    ),
}


def available_profile_names() -> list[str]:
    """Return sorted supported profile names."""

    return sorted(PROFILES)


def get_profile(name: str) -> ProfileSpec:
    """Return a named profile or raise a helpful error."""

    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown profile: {name}") from exc


def probability_tag(p_error: float) -> str:
    """Return a stable path-safe physical-error slug."""

    raw = f"{p_error:.4f}".rstrip("0").rstrip(".")
    return f"p{raw.replace('.', 'p')}"


def dataset_id_for_profile(profile: ProfileSpec, train_shots: int, val_shots: int) -> str:
    """Return the stable dataset identifier for a named profile and shot count."""

    return (
        f"{profile.name}-d{profile.distance}-r{profile.rounds}-"
        f"{len(profile.p_errors)}rates-train{train_shots}-val{val_shots}"
    )
