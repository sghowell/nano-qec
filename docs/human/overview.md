# Overview

NanoQEC is a compact research environment for improving neural decoders for
quantum error correction. It is built to make experiment loops legible and
reproducible: data preparation is deterministic, training and evaluation happen
through stable public entrypoints, benchmark policy is explicit, and artifacts
are written in machine-readable formats that can be audited after the fact.

## Implemented Today

As of April 9, 2026, the repository implements:

- deterministic profile preparation for `local-d3-v1` and `local-d5-v1`
- public `prepare.py`, `train.py`, and `eval.py` entrypoints
- a local single-host workflow built around `uv`
- operator-managed single-host cloud helpers that keep the same documented
  contracts
- a promoted `local-d3-v1` `spacetime_gnn` regime with repeated evidence below
  MWPM on the primary benchmark

## Future Direction

Future expansion remains in
[../nanoqec-plan.md](../nanoqec-plan.md), including broader autonomy, larger
research phases, and more ambitious execution environments. Those directions are
intentional future work, not promises about the present repository surface.

## Why This Problem Matters

Quantum error correction needs decoders that can infer likely logical failures
from noisy detector events. Minimum-weight perfect matching (MWPM) is a strong
classical baseline for many surface-code settings, so a learned decoder needs a
clear reason to exist: it must either beat MWPM or expose a path toward regimes
where learned structure matters. NanoQEC focuses on that comparison directly.

## What NanoQEC Is Optimized For

- fast local iteration on well-defined research profiles
- stable experimental contracts that survive repeated edits
- evidence-driven promotion instead of ad hoc “looks better” decisions
- agent-assisted experimentation that still stays inside repo authority

## What Is Deferred

The repository does not present itself as a general-purpose, fully autonomous
research platform. Multi-host training, provider-specific orchestration,
autonomous promotion, and cron-driven overnight autonomy remain future phases
and are intentionally deferred in the v0 docs.

## Supported Profiles and Benchmark Framing

- `local-d3-v1`: the main human-facing story and the source of the promoted
  MWPM-beating result
- `local-d5-v1`: a supported larger-distance profile for local or single-host
  cloud experiments

For `local-d3-v1`, the repo’s human-facing results story is anchored to a
primary `train8192 / val1024` benchmark and a continuity `train8192 / val256`
benchmark. See [results.md](./results.md) for the evidence trail and dates.

## Where Strategy Lives

[../nanoqec-plan.md](../nanoqec-plan.md) is the long-range architecture and
research note. Use it for project direction, not for exact present-day
contracts.

## Where Exact Rules Live

The exact operational rules and protected interfaces remain in:

- [../../AGENTS.md](../../AGENTS.md)
- [../implementation-v0.md](../implementation-v0.md)
- [../hermes-ops.md](../hermes-ops.md)

## Next Reads

- [Quickstart](./quickstart.md)
- [Concepts](./concepts.md)
- [Results and evidence](./results.md)
