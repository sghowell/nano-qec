# Concepts

This page defines the core NanoQEC vocabulary in plain English. It is meant for
human readers who want to understand the repo’s main ideas without diving
straight into schema definitions, training code, or operator-only runbooks.

## Implemented Today

As of April 9, 2026, the repository’s human-facing concepts center on two local
profiles, deterministic dataset preparation, decoder training, evaluation
against MWPM, and explicit benchmark policy for `local-d3-v1`.

## Future Direction

Future project phases may add new execution patterns or benchmark regimes, but
this page only describes the concepts that the repository documents today.

## Surface-Code Profile

A profile is a named experiment setting that fixes the code distance, the number
of rounds, and the physical error-rate sweep used for dataset preparation and
evaluation. In NanoQEC, `local-d3-v1` and `local-d5-v1` are the supported
profiles documented in [../implementation-v0.md](../implementation-v0.md).

## MWPM Baseline

MWPM stands for minimum-weight perfect matching. In this repository it is the
classical baseline used to evaluate whether a learned decoder is actually useful
on the same validation slices. A learned model does not get credit for being
interesting unless it can be compared directly to MWPM.

## Aggregate MWPM Ratio

The aggregate MWPM ratio is:

`aggregate learned-decoder logical error rate / aggregate MWPM logical error rate`

Lower is better. A value below `1.0` means the learned decoder beat MWPM on the
aggregate benchmark being reported.

## Primary vs Continuity Benchmark

For the promoted `local-d3-v1` story, NanoQEC distinguishes between:

- a primary benchmark used for research and promotion decisions
- a continuity benchmark used as a secondary consistency check

As of March 24, 2026, the promoted evidence trail treats `train8192 / val1024`
as primary and `train8192 / val256` as continuity. See [results.md](./results.md)
for the dates and sources.

## Dataset Manifest

The dataset manifest is the file produced by `prepare.py` that points to the
prepared slices and their metadata. It is the canonical handoff between data
preparation and the later `train.py` / `eval.py` stages.

## Checkpoint

A checkpoint is the serialized decoder state produced by `train.py`. NanoQEC
writes both a run-specific checkpoint and a `best.pt` checkpoint inside the
selected checkpoint directory. The checkpoint metadata records the model name,
model spec, dataset identity, and selected decision threshold.

## Evaluation Artifact

An evaluation artifact is the JSON output written by `eval.py` for a particular
checkpoint and manifest pair. It reports aggregate metrics, per-slice metrics,
and optionally a plot of logical error rate versus physical error rate.

## Why These Concepts Matter Together

NanoQEC is easiest to understand as a short loop:

1. prepare a deterministic profile
2. train a decoder on that manifest
3. evaluate it against MWPM
4. compare the result to the benchmark policy

The repo’s structure is designed to make that loop explicit and auditable.

## Next Reads

- [Architecture](./architecture.md)
- [Results and evidence](./results.md)
- [Glossary](./glossary.md)
