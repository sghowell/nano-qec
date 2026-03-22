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

Sample prompts for an autoresearch kickoff:

```text
Read AGENTS.md, docs/implementation-v0.md, and docs/hermes-ops.md. Then run one bounded local-d3-v1 decoder-improvement experiment on a short-lived exp branch. Stay inside the protected harness contracts, use the existing prepare/train/eval entrypoints, and do not merge automatically. When done, report the hypothesis, exact code changes, aggregate val_ler, aggregate mwpm_ratio, and whether the branch should be kept or discarded.
```

```text
Read the NanoQEC authority docs, inspect the most recent local-d3-v1 results, and propose the highest-signal next experiment to reduce aggregate mwpm_ratio. Before changing code, explain the hypothesis, the files you plan to edit, and the validation path you will run.
```

Sample prompts for querying recent runs:

```text
Read results/experiments.jsonl and summarize the 5 most recent runs. For each, report branch, hypothesis, aggregate val_ler, aggregate mwpm_ratio, kept/discarded, and the most important takeaway.
```

```text
Inspect the latest training and evaluation artifacts for local-d3-v1 and tell me which physical-error-rate slices are currently weakest relative to MWPM. Use concrete numbers and suggest one targeted follow-up experiment.
```

```text
Compare the current default decoder settings against the best recent run you can find in results/experiments.jsonl and results/train/. Tell me what changed, whether the newer run is reproducibly better, and what evidence is still missing before promotion.
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
uv run python scripts/tune_profile.py --workspace . --dataset-manifest data/local-d3-v1-d3-r3-5rates-train1024-val256/manifest.json --config baseline --config warmup_cosine --duration-seconds 30 --duration-seconds 60 --repeats 3 --eval-interval-seconds 5 --device mps
```

`train.py` also supports an optional time-based warmup+cosine learning-rate
schedule for fixed-budget runs:

```bash
uv run train.py --workspace . --dataset-manifest data/local-d3-v1-d3-r3-5rates-train1024-val256/manifest.json --scheduler warmup_cosine --warmup-fraction 0.1 --min-learning-rate-scale 0.1
```

Training metrics JSON now includes `eval_history`, which records periodic
aggregate validation snapshots across the run so plateauing and schedule effects
can be diagnosed after the fact.
