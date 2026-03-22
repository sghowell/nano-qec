# NanoQEC V0 Implementation Spec

## Scope

This document freezes the v0 implementation boundary for the local readiness
harness.

- Local single-host execution only
- `surface_code:rotated_memory_x`
- `distance=3`
- `rounds=3`
- one logical observable
- `p_error=0.005`
- deterministic cached validation data
- `uv` as the only package and task runner

Out of scope for v0:

- `d=5`
- mixed-distance batches
- Prime Intellect integration
- cron or overnight scheduling
- automatic promotion to `main`

## Canonical Data Representation

The canonical stored representation is flat detector ordering plus coordinate
metadata from Stim.

Observed reference layout for `distance=3`, `rounds=3`:

- detector count: `24`
- observable count: `1`
- time coordinates: `0.0`, `1.0`, `2.0`, `3.0`
- detector counts per time coordinate: `4`, `8`, `8`, `4`

The public dataset contract stores flat detector events and the detector
coordinates needed to derive time-grouped views internally. The baseline AQ2
model may build a padded time-bucket tensor internally, but that transformation
is not part of the public dataset artifact schema.

## Reproducibility Rules

- Validation caches are fixed for a dataset id and seed.
- Training uses a separate training seed from data generation.
- Time budgets apply to training only and exclude cache generation.
- The same dataset manifest must identify the same artifact paths and metadata
  across repeated `prepare.py` runs unless `--force` is used.

## CLI Contracts

### `prepare.py`

Materializes deterministic train/validation caches plus a manifest.

Required behavior:

- create `data/<dataset_id>/train.npz`
- create `data/<dataset_id>/val.npz`
- create `data/<dataset_id>/manifest.json`
- populate MWPM validation baseline in the manifest
- print a JSON summary to stdout
- exit non-zero on invalid config or schema failure

Default contract:

- profile: `local-d3-v0`
- distance: `3`
- rounds: `3`
- physical error rate: `0.005`
- train seed: `20260321`
- val seed: `20260322`

### `train.py`

Consumes a dataset manifest and produces a checkpoint, metrics JSON, and one
machine-parseable `RESULT` line.

Required behavior:

- load the dataset manifest
- train a baseline minimal AQ2-style model by default
- support at least one alternate model implementation to prove architecture
  replaceability
- write checkpoint metadata sufficient for `eval.py` to reconstruct the model
- optionally append a runtime record to `results/experiments.jsonl`
- exit non-zero on contract violations or training failure

`RESULT` line contract:

```text
RESULT {"run_id":"...","val_ler":0.123,"mwpm_ratio":1.1,"kept":false}
```

### `eval.py`

Consumes a dataset manifest and checkpoint, then emits structured evaluation
metrics.

Required behavior:

- rebuild the model from checkpoint metadata, not hard-coded class assumptions
- compute validation LER
- report cached MWPM comparison
- write a JSON results file
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
- `p_error`
- `detector_count`
- `observable_count`
- `representation`
- `splits`
- `baselines`

### Checkpoint Metadata

Required fields:

- `schema_version`
- `model_name`
- `model_spec`
- `train_config`
- `dataset_id`
- `train_seed`
- `git_sha`

### Runtime Experiment Record

Required fields:

- `schema_version`
- `run_id`
- `dataset_id`
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

In v0, runtime experiment records may be appended automatically, but no branch
may be promoted automatically. `kept` means the run beat the best previously
recorded `val_ler` for the same dataset/evaluation profile; it does not grant
merge permission by itself.
