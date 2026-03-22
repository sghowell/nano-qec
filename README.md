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
- `archived/`: historical reference files that may lag the implementation.

## Quickstart

```bash
uv sync --all-extras
uv run prepare.py --workspace . --profile local-d3-v1
uv run train.py --workspace . --dataset-manifest data/local-d3-v1-d3-r3-5rates-train1024-val256/manifest.json
uv run eval.py --workspace . --dataset-manifest data/local-d3-v1-d3-r3-5rates-train1024-val256/manifest.json --checkpoint checkpoints/best.pt
```

## Using Hermes Directly

Run Hermes from the repo root so it can see the NanoQEC authority docs and
artifacts.

Preflight:

```bash
hermes model
hermes status
```

Interactive CLI/TUI session:

```bash
hermes chat
```

Suggested first prompt:

```text
Read AGENTS.md, docs/implementation-v0.md, and docs/hermes-ops.md, then summarize the current NanoQEC operating rules before making any changes.
```

One-shot repo-root query:

```bash
hermes chat -Q -q "Read AGENTS.md, docs/implementation-v0.md, and docs/hermes-ops.md from the current repo, then reply with a one-line readiness summary."
```

Resume a previous session:

```bash
hermes -c
hermes sessions browse
```

Run Hermes in an isolated git worktree:

```bash
hermes chat --worktree
```

Documented NanoQEC dry-run:

```bash
uv sync --all-extras
uv run prepare.py --workspace . --profile local-d3-v1
uv run train.py --workspace . --dataset-manifest data/local-d3-v1-d3-r3-5rates-train1024-val256/manifest.json
uv run eval.py --workspace . --dataset-manifest data/local-d3-v1-d3-r3-5rates-train1024-val256/manifest.json --checkpoint checkpoints/best.pt
```

See `docs/hermes-ops.md` for the full repo mutation policy and dry-run workflow.

## Validation

```bash
uv run ruff check .
uv run pytest
```

## Experiment Utilities

```bash
uv run python scripts/check_improvement.py --metrics-json results/train/<run>.json
uv run python scripts/plot_progress.py --experiment-log results/experiments.jsonl --output results/progress.png
uv run python scripts/tune_profile.py --workspace . --dataset-manifest data/local-d3-v1-d3-r3-5rates-train1024-val256/manifest.json --config baseline --config adamw --config lr1e3 --repeats 3 --duration-seconds 30 --eval-interval-seconds 5
```
