# Hermes Operations

## Purpose

This runbook defines how Hermes should operate inside the NanoQEC repository.

## Preconditions

- Hermes must have local terminal access to the repository.
- Hermes must use the repo-local docs as authority.
- Hermes must run `uv` commands from the repo root.
- Hermes must not rely on hidden state in `~/.hermes` to understand NanoQEC.
- Hermes should treat `best.pt` and the aggregate profile metrics as the default
  local comparison artifacts.

Before attempting a repo dry-run, Hermes itself must be ready:

- `hermes status` must show at least one working inference path.
- If the default provider is not logged in, fix it with `hermes model` before
  trying repo work.
- If a one-shot `hermes chat` call fails with a provider or model `400` error,
  treat that as a Hermes runtime/config issue, not a NanoQEC repo issue.

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

Recommended preflight:

```bash
hermes status
```

Direct interactive use from the repo root:

```bash
hermes chat
```

Suggested first prompt:

```text
Read AGENTS.md, docs/implementation-v0.md, and docs/hermes-ops.md, then summarize the NanoQEC operating rules before making any changes.
```

Sample autoresearch kickoff prompts:

```text
Read AGENTS.md, docs/implementation-v0.md, and docs/hermes-ops.md. Then run one bounded local-d3-v1 decoder-improvement experiment on a short-lived exp branch. Stay inside the protected harness contracts, use the existing prepare/train/eval entrypoints, and do not merge automatically. When done, report the hypothesis, exact code changes, aggregate val_ler, aggregate mwpm_ratio, and whether the branch should be kept or discarded.
```

```text
Read the NanoQEC authority docs, inspect the most recent local-d3-v1 results, and propose the highest-signal next experiment to reduce aggregate mwpm_ratio. Before changing code, explain the hypothesis, the files you plan to edit, and the validation path you will run.
```

Sample recent-run query prompts:

```text
Read results/experiments.jsonl and summarize the 5 most recent runs. For each, report branch, hypothesis, aggregate val_ler, aggregate mwpm_ratio, kept/discarded, and the most important takeaway.
```

```text
Inspect the latest training and evaluation artifacts for local-d3-v1 and tell me which physical-error-rate slices are currently weakest relative to MWPM. Use concrete numbers and suggest one targeted follow-up experiment.
```

Useful session helpers:

```bash
hermes -c
hermes sessions browse
hermes chat --worktree
```

If the configured provider is healthy, a minimal repo-root one-shot should be
able to read the authoritative docs:

```bash
hermes chat -Q -q "Read AGENTS.md, docs/implementation-v0.md, and docs/hermes-ops.md from the current repo, then reply with a one-line readiness summary."
```

Then the repo dry-run itself is:

```bash
uv sync --all-extras
uv run prepare.py --workspace . --profile local-d3-v1
uv run train.py --workspace . --dataset-manifest data/local-d3-v1-d3-r3-5rates-train8192-val256/manifest.json
uv run eval.py --workspace . --dataset-manifest data/local-d3-v1-d3-r3-5rates-train8192-val256/manifest.json --checkpoint checkpoints/best.pt
```

If any command, path, or schema deviates from this runbook, Hermes must update
the repo docs in the same branch as the code change.

For `local-d3-v1`, Hermes should treat train `8192` / val `1024` as the
primary research and promotion benchmark. Train `8192` / val `256` remains a
continuity check and should still be reported, but it is secondary when the two
benchmarks disagree.
