# Hermes Operations

## Purpose

This runbook defines how Hermes should operate inside the NanoQEC repository.

## Preconditions

- Hermes must have local terminal access to the repository.
- Hermes must use the repo-local docs as authority.
- Hermes must run `uv` commands from the repo root.
- Hermes must not rely on hidden state in `~/.hermes` to understand NanoQEC.

## Repo Discovery

Hermes should read these files before changing behavior:

1. `AGENTS.md`
2. `docs/implementation-v0.md`
3. `docs/hermes-ops.md`
4. any file directly referenced by those docs

## Allowed Mutations In V0

- code under `src/nanoqec/`
- thin root entrypoints `prepare.py`, `train.py`, and `eval.py`
- tests
- repo-local docs and adapters

## Disallowed Mutations In V0

- changing protected schemas without synchronized tests and doc updates
- changing the evaluation policy ad hoc
- automatic merges to `main`
- cloud, cron, or multi-host setup changes as part of a local v0 run

## Branch Workflow

- use `feature/<topic>` for harness or documentation work
- use `exp/<timestamp>-<slug>` for experiment branches
- commit in logical chunks
- run the documented validation commands before proposing a merge

## Dry-Run Workflow

```bash
uv sync --all-extras
uv run prepare.py --workspace .
uv run train.py --workspace . --dataset-manifest data/local-d3-v0-d3-r3-p0p005/manifest.json
uv run eval.py --workspace . --dataset-manifest data/local-d3-v0-d3-r3-p0p005/manifest.json --checkpoint checkpoints/latest.pt
```

If any command, path, or schema deviates from this runbook, Hermes must update
the repo docs in the same branch as the code change.
