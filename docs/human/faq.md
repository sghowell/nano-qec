# FAQ

This page answers the questions a new human reader is most likely to ask after
reading the README or skimming the results page. It focuses on practical
confusion points rather than deep implementation details.

## Implemented Today

As of April 9, 2026, NanoQEC is a local-first, harness-first neural
quantum-error-correction repo with two supported profiles and a documented
`local-d3-v1` promoted result.

## Future Direction

The repo’s future directions are intentionally separated into
[../nanoqec-plan.md](../nanoqec-plan.md). This FAQ is about present behavior.

## Does NanoQEC beat MWPM?

On the promoted `local-d3-v1` primary benchmark documented on March 24, 2026,
yes: the repeated mean aggregate MWPM ratio is `0.959` on the
`train8192 / val1024` lane. That does not mean every benchmark lane or future
profile beats MWPM.

## Why is the first quickstart run smaller than the headline benchmark?

Because human onboarding and benchmark claims have different goals. The
quickstart is a fast first run that proves the pipeline works without dirtying
the repo. The promoted result is a separate research benchmark with larger data
and a longer training budget.

## Why does the quickstart write to `/tmp`?

To keep a first run from overwriting tracked evaluation artifacts or leaving a
dirty worktree. Once you know the pipeline works, you can switch to the default
repo output directories for research runs.

## Which docs are authoritative?

The authoritative docs are:

- [../../AGENTS.md](../../AGENTS.md)
- [../implementation-v0.md](../implementation-v0.md)
- [../hermes-ops.md](../hermes-ops.md)

The human docs under `docs/human/` are a guide, not a replacement.

## Is `results/eval/best-eval.json` the promoted result?

No. It is a checked-in evaluation artifact and a useful schema example, but it
records an older smaller-scale evaluation with `aggregate_mwpm_ratio = 1.9032`.
The promoted result is documented in the overnight summaries and paper/blog
materials linked from [results.md](./results.md).

## Is NanoQEC only for Hermes?

No. Humans can run the full prepare/train/eval flow directly. Hermes is a tool
that operates inside the repo’s authority model, not the source of truth for
the project.

## Why are there two benchmark lanes?

Because the project separates a primary benchmark used for research and
promotion decisions from a continuity benchmark used as a secondary consistency
check. That split is part of the repo’s evidence discipline.

## Next Reads

- [Quickstart](./quickstart.md)
- [Results and evidence](./results.md)
- [Glossary](./glossary.md)
