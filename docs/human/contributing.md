# Contributing

This page is the human-facing contributor workflow for NanoQEC. It explains how
to make changes, what validation is expected, how to stay inside the repo’s
protected boundaries, and which docs to update when behavior changes.

## Implemented Today

As of April 9, 2026, NanoQEC expects contributors to work on short-lived
branches, validate locally with `uv`, and treat the repo-local authority docs
as the source of truth for behavior and protected interfaces.

## Future Direction

The repository may grow broader workflows later, but this page describes the
documented v0 contributor model only.

## Branch and Validation Discipline

- start from updated `main`
- use `feature/<topic>` for harness, docs, and infra work
- use `exp/<timestamp>-<slug>` for experiments
- keep commits logical and concise
- revalidate before merge

Default validation path:

```bash
uv sync --all-extras
uv run ruff check .
uv run pytest
```

## Protected Boundaries

Before changing behavior, read [../../AGENTS.md](../../AGENTS.md). In
particular, treat these as protected unless code, docs, and tests move together:

- `prepare.py` CLI behavior and dataset manifest schema
- `eval.py` CLI behavior and evaluation output schema
- shared artifact schemas and the `RESULT` line contract
- smoke tests and validation commands
- experiment promotion policy

## Documentation Expectations

- update human-facing docs when the human entry path changes
- update authority docs only when behavior or contracts actually change
- do not duplicate long protocol text across multiple files
- keep claims concrete: files, dates, commands, outputs

Use [review-checklist.md](./review-checklist.md) before merging a doc-heavy
change.

## Human-Facing Doc Flow

For this repo, the clean split is:

- `README.md` and `docs/human/`: human-first explanation and navigation
- `AGENTS.md`, `docs/implementation-v0.md`, `docs/hermes-ops.md`: exact
  operating rules and contracts

Do not let human-facing docs silently redefine protected behavior.

## Working With Runtime Artifacts

NanoQEC’s runtime artifacts are useful evidence, but they should not be treated
as casual scratch space. For quick experiments or docs validation, prefer
writing temporary outputs outside tracked paths when possible.

## Next Reads

- [Using Hermes](./using-hermes.md)
- [Docs review checklist](./review-checklist.md)
- [FAQ](./faq.md)
