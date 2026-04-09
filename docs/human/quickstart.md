# Quickstart

This guide gets a human collaborator through a first NanoQEC run without
dirtying the repository. It uses a short `local-d3-v1` lane that writes
checkpoints and evaluation outputs to `/tmp`, then points to the promoted
benchmark lane once the basic pipeline works.

## Implemented Today

As of April 9, 2026, the validated first-run path below uses `uv`, Python 3.11,
`local-d3-v1`, a `30s` training budget, and temporary output directories under
`/tmp/nanoqec-quickstart`.

## Future Direction

The promoted research lane and long-horizon project direction are documented
elsewhere. This page is intentionally about a fast, successful first run, not
about reproducing the full promoted `8192 / 1024 / 180s` benchmark.

## Prerequisites

- a local clone of this repository
- Python `3.11`
- `uv`
- enough disk space for prepared datasets and temporary checkpoints

If you need install help, see the `uv` project docs and the repo’s
[pyproject.toml](../../pyproject.toml).

## First Run

Run these commands from the repo root:

```bash
uv sync --all-extras
uv run prepare.py --workspace . --profile local-d3-v1 --train-shots 1024 --val-shots 256
uv run train.py --workspace . --dataset-manifest data/local-d3-v1-d3-r3-5rates-train1024-val256/manifest.json --duration-seconds 30 --device auto --checkpoint-dir /tmp/nanoqec-quickstart/checkpoints --results-dir /tmp/nanoqec-quickstart/results/train --experiment-log /tmp/nanoqec-quickstart/results/experiments.jsonl --skip-experiment-log
uv run eval.py --workspace . --dataset-manifest data/local-d3-v1-d3-r3-5rates-train1024-val256/manifest.json --checkpoint /tmp/nanoqec-quickstart/checkpoints/best.pt --device auto --results-dir /tmp/nanoqec-quickstart/results/eval
```

## What Success Looks Like

After `prepare.py`:

- `data/local-d3-v1-d3-r3-5rates-train1024-val256/manifest.json` exists

After `train.py`:

- the terminal prints a final `RESULT {...}` line
- `/tmp/nanoqec-quickstart/checkpoints/best.pt` exists
- `/tmp/nanoqec-quickstart/results/train/<run-id>.json` exists

After `eval.py`:

- `/tmp/nanoqec-quickstart/results/eval/best-eval.json` exists
- `/tmp/nanoqec-quickstart/results/eval/best-eval-ler-vs-p.png` exists

## Common Failure Modes

### `uv` or Python version mismatch

If `uv sync --all-extras` fails because Python `3.11` is unavailable, install
Python `3.11` first and rerun the sync step.

### Missing accelerator support

`--device auto` resolves to `mps`, `cuda`, or `cpu` depending on the host. If
accelerator selection fails or is unstable on your machine, rerun `train.py`
and `eval.py` with `--device cpu`.

### Manifest path mismatch

If `train.py` complains that the manifest path does not exist, check that the
`prepare.py` command used the same `--train-shots` and `--val-shots` values that
appear in the manifest path.

### Plot generation problems

If `eval.py` fails during plotting on a constrained environment, rerun it with
`--skip-plot`. The metrics JSON is the important artifact.

## Canonical Research Lane

Once the short first run works, the canonical promoted `local-d3-v1` benchmark
lane is:

- `train8192 / val1024`
- `spacetime_gnn`
- `180s`

That lane is described in [results.md](./results.md). The exact protected
contracts behind it remain in [../implementation-v0.md](../implementation-v0.md).

## Next Reads

- [Results and evidence](./results.md)
- [Concepts](./concepts.md)
- [Contributing](./contributing.md)
