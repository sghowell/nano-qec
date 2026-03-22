# NanoQEC Local Implementation Spec

## Scope

This document freezes the current local implementation boundary for NanoQEC.

- Local single-host execution only
- `surface_code:rotated_memory_x`
- supported profiles: `local-d3-v1`, `local-d5-v1`
- one logical observable
- profile-level multi-rate validation sweeps
- deterministic cached training and validation data
- `uv` as the only package and task runner

Out of scope for the current local phase:

- mixed-distance training in a single run
- Prime Intellect integration
- cron or overnight scheduling
- automatic promotion to `main`

## Profiles

### `local-d3-v1`

- distance: `3`
- rounds: `3`
- physical error rates: `0.001`, `0.003`, `0.005`, `0.007`, `0.01`
- default train shots per slice: `1024`
- default val shots per slice: `256`

### `local-d5-v1`

- distance: `5`
- rounds: `5`
- physical error rates: `0.001`, `0.003`, `0.005`, `0.007`, `0.01`
- default train shots per slice: `512`
- default val shots per slice: `256`

## Canonical Data Representation

The canonical stored representation remains flat detector ordering plus
coordinate metadata from Stim. Each prepared profile stores a single shared
layout description plus per-error-rate slice artifacts.

Observed reference layout shapes:

- `d=3`, `rounds=3`: detector count `24`, time bucket sizes `[4, 8, 8, 4]`
- `d=5`, `rounds=5`: detector count `120`, time bucket sizes `[12, 24, 24, 24, 24, 12]`

The public dataset artifact schema stores the flat events plus metadata needed
to derive detector-aware time buckets internally. The AQ2 baseline consumes a
padded time-bucket view built from that metadata, but that transformation is
internal to the model stack. The current default decoder also conditions on the
profile physical error rate internally and applies a learned global decision
threshold after training.

## Reproducibility Rules

- Validation caches are fixed for a dataset id and slice seed.
- Training uses a separate training seed from data generation.
- Time budgets apply to training only and exclude cache generation.
- The same dataset manifest must identify the same artifact paths and metadata
  across repeated `prepare.py` runs unless `--force` is used.

## CLI Contracts

### `prepare.py`

Materializes deterministic multi-slice train/validation caches plus a manifest.

Required behavior:

- create `data/<dataset_id>/manifest.json`
- create `train_<slice_id>.npz` and `val_<slice_id>.npz` per physical error rate
- populate MWPM validation baselines per slice and an aggregate baseline mean
- print a JSON summary to stdout
- exit non-zero on invalid config or schema failure

### `train.py`

Consumes a dataset manifest and produces checkpoints, metrics JSON, and one
machine-parseable `RESULT` line.

Required behavior:

- load all slices from the dataset manifest
- train a stronger minimal AQ2-style model by default
- support at least one alternate model implementation to prove architecture
  replaceability
- periodically evaluate across all validation slices and save `best.pt`
- calibrate a global decision threshold on the training split before writing
  final metrics and checkpoints
- write checkpoint metadata sufficient for `eval.py` to reconstruct the model
- optionally append a runtime record to `results/experiments.jsonl`
- exit non-zero on contract violations or training failure

`RESULT` line contract:

```text
RESULT {"run_id":"...","val_ler":0.123,"mwpm_ratio":1.1,"kept":false}
```

The primary `val_ler` in the `RESULT` line is the aggregate mean validation LER
across all slices in the manifest profile.

### `eval.py`

Consumes a dataset manifest and checkpoint, then emits structured profile
evaluation metrics.

Required behavior:

- rebuild the model from checkpoint metadata, not hard-coded class assumptions
- use the checkpoint decision threshold when present
- compute aggregate and per-slice validation LER
- report per-slice MWPM comparisons
- write a JSON results file
- optionally write an `LER vs p` plot when plotting support is available
- print a JSON summary to stdout
- exit non-zero on missing or invalid artifacts

## Shared Schemas

### Dataset Manifest

Required fields:

- `schema_version`
- `dataset_id`
- `profile`
- `circuit_name`
- `distance`
- `rounds`
- `detector_count`
- `observable_count`
- `representation`
- `p_error_values`
- `slices`
- `aggregate_baselines`

### Checkpoint Metadata

Required fields:

- `schema_version`
- `model_name`
- `model_spec`
- `train_config`
- `dataset_id`
- `train_seed`
- `git_sha`

Current checkpoints should also include `decision_threshold`. `eval.py`
defaults to `0.5` when loading older local checkpoints that predate threshold
calibration.

### Runtime Experiment Record

Required fields:

- `schema_version`
- `run_id`
- `dataset_id`
- `profile`
- `hypothesis`
- `branch_name`
- `git_sha`
- `model_name`
- `model_spec`
- `config_snapshot`
- `metrics`
- `artifacts`
- `kept`
- `wall_time_sec`

## Promotion Rule

The current local phase may append runtime experiment records automatically, but
no branch may be promoted automatically. `kept` means the run beat the best
previous aggregate validation LER for the same dataset profile; it does not
grant merge permission by itself.
