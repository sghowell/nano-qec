# NanoQEC

NanoQEC is a harness-first repo for autonomous neural quantum error-correction
experiments. The repository, not the agent runtime, is the source of truth for
contracts, schemas, validation, and operating rules.

## Source Of Truth

- [`AGENTS.md`](/Users/seanhowell/dev/nano-qec/AGENTS.md): operational authority
  for humans and Hermes.
- [`docs/implementation-v0.md`](/Users/seanhowell/dev/nano-qec/docs/implementation-v0.md):
  frozen v0 readiness spec and CLI contracts.
- [`docs/hermes-ops.md`](/Users/seanhowell/dev/nano-qec/docs/hermes-ops.md):
  Hermes runbook and mutation policy.
- [`docs/nanoqec-plan.md`](/Users/seanhowell/dev/nano-qec/docs/nanoqec-plan.md):
  long-horizon architecture and research strategy.

## Quickstart

```bash
uv sync --all-extras
uv run prepare.py --workspace .
uv run train.py --workspace . --dataset-manifest data/local-d3-v0-d3-r3-p0p005/manifest.json
uv run eval.py --workspace . --dataset-manifest data/local-d3-v0-d3-r3-p0p005/manifest.json --checkpoint checkpoints/latest.pt
```

## Validation

```bash
uv run ruff check .
uv run pytest
```
