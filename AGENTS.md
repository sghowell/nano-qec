# AGENTS.md

## Purpose

This file is the operational authority for NanoQEC. Hermes and humans must
follow the repo-local contracts here and in `docs/` rather than relying on
external memory or hidden instructions.

## Authority Order

1. This file.
2. `docs/implementation-v0.md`.
3. `docs/hermes-ops.md`.
4. Code, tests, and schema validators.
5. `docs/nanoqec-plan.md` for long-term strategy only.

If any of these drift, update the repo docs and code in the same branch before
merging.

## Operating Model

- Hermes is the primary operator for experiment execution.
- The repository is the source of truth for behavior, contracts, and guardrails.
- `SKILL.md` and `program.md` are adapters that point back to this repository.
- The current implementation target is local-only profile training using `uv`.
- Supported research profiles are `local-d3-v1` and `local-d5-v1`.
- Cloud GPUs, mixed-distance training, cron scheduling, and autonomous
  promotion remain deferred until the local harness is stable.

## Protected Boundaries

The following are protected in v0 and may only change with synchronized doc,
test, and schema updates:

- `prepare.py` CLI behavior and dataset manifest schema
- `eval.py` CLI behavior and evaluation output schema
- shared artifact schemas and the `RESULT` line contract
- smoke tests and validation commands
- experiment promotion policy

The following are intentionally mutable behind the protected interfaces:

- model architecture
- optimizer and scheduler choices
- loss shaping
- internal module layout under `src/nanoqec/`
- checkpoint contents beyond the required metadata contract

## Python Standards

- Use Python `3.11` and `uv` only. Do not use `pip`.
- Type-annotate public APIs.
- Prefer small functions with explicit inputs and outputs around I/O boundaries.
- Use `pathlib.Path` for filesystem paths.
- Keep configs explicit, serializable, and suitable for artifact metadata.
- Use `logging` for operational output. The only required ad hoc stdout line is
  the final machine-parseable `RESULT ...` line from `train.py`.
- Add concise docstrings to public modules, classes, and functions.
- Keep runtime artifacts out of git. Commit code, docs, tests, and harness
  changes only.

## Documentation Standards

- Update the authoritative docs in the same branch as any behavior change.
- Keep docs concrete: name files, commands, schemas, and failure conditions.
- Do not duplicate long protocol text across multiple files. Point to the
  authoritative file instead.

## Git Discipline

- Do non-trivial work on short-lived branches.
- Branch from updated `main`.
- Use `feature/<topic>` for harness, docs, and infra work.
- Use `exp/<timestamp>-<slug>` for experiment branches.
- Stage and commit in logical chunks with concise imperative commit messages.
- Revalidate on the branch before merging.
- Merge locally, re-run the critical checks on local `main`, then push.
- Delete merged or rejected branches and clean temporary artifacts afterward.
- Hermes may create runtime experiment records automatically, but it must not
  auto-merge to `main` in v0.

## Validation Commands

Use these commands as the default local validation path:

```bash
uv sync --all-extras
uv run ruff check .
uv run pytest
```

If CLI behavior or schemas change, add or update a test that proves the new
contract.
