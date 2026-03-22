# NanoQEC

NanoQEC is a harness-first repo for autonomous neural quantum error-correction
experiments. The repository, not the agent runtime, is the source of truth for
contracts, schemas, validation, and operating rules.

## Source Of Truth

- `AGENTS.md`: operational authority for humans and Hermes.
- `docs/implementation-v0.md`: current local implementation spec and CLI
  contracts.
- `docs/hermes-ops.md`: Hermes runbook and mutation policy.
- `docs/nanoqec-plan.md`: long-horizon architecture and research strategy.

## Quickstart

```bash
uv sync --all-extras
uv run prepare.py --workspace . --profile local-d3-v1
uv run train.py --workspace . --dataset-manifest data/local-d3-v1-d3-r3-5rates-train1024-val256/manifest.json
uv run eval.py --workspace . --dataset-manifest data/local-d3-v1-d3-r3-5rates-train1024-val256/manifest.json --checkpoint checkpoints/best.pt
```

## Validation

```bash
uv run ruff check .
uv run pytest
```

## Experiment Utilities

```bash
uv run python scripts/check_improvement.py --metrics-json results/train/<run>.json
uv run python scripts/plot_progress.py --experiment-log results/experiments.jsonl --output results/progress.png
```
