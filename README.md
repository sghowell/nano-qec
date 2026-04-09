```
    ███    ██  █████  ███    ██  ██████   ██████  ███████  ██████
    ████   ██ ██   ██ ████   ██ ██    ██ ██    ██ ██      ██
    ██ ██  ██ ███████ ██ ██  ██ ██    ██ ██    ██ █████   ██
    ██  ██ ██ ██   ██ ██  ██ ██ ██    ██ ██ ▄▄ ██ ██      ██
    ██   ████ ██   ██ ██   ████  ██████   ██████  ███████  ██████
                                             ▀▀
    ───────────────────────────────────────────────────────────────
    neural quantum error correction
```

[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](./pyproject.toml)
[![CI](https://github.com/sghowell/nano-qec/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sghowell/nano-qec/actions/workflows/ci.yml)
[![Managed with uv](https://img.shields.io/badge/managed%20with-uv-DE5FE9?logo=uv&logoColor=white)](./uv.lock)
[![Implementation v0](https://img.shields.io/badge/spec-implementation%20v0-FF6F00)](./docs/implementation-v0.md)
[![Hermes Operated](https://img.shields.io/badge/operator-Hermes-111111)](./docs/hermes-ops.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-2EA44F)](./LICENSE)

NanoQEC is a harness-first research repository for neural quantum
error-correction experiments. It combines deterministic data preparation,
stable `prepare.py` / `train.py` / `eval.py` entrypoints, explicit benchmark
policy, and machine-readable artifacts so humans and agents can improve local
surface-code decoders without drifting the protocol.

## Implemented Today

As of April 9, 2026, this repository implements:

- single-host local runs with `uv`
- operator-managed single-host cloud GPU runs that preserve the same public
  CLI and artifact contracts
- two supported research profiles: `local-d3-v1` and `local-d5-v1`
- reproducible dataset preparation, training, evaluation, and experiment logs
- a promoted `local-d3-v1` graph-native decoder regime documented in the repo

Future direction remains in [docs/nanoqec-plan.md](./docs/nanoqec-plan.md). The
authoritative operational and contract docs remain
[AGENTS.md](./AGENTS.md),
[docs/implementation-v0.md](./docs/implementation-v0.md), and
[docs/hermes-ops.md](./docs/hermes-ops.md).

## Headline Result

As of March 24, 2026, NanoQEC’s promoted `local-d3-v1` regime is a
`spacetime_gnn` decoder with `6` graph blocks, `feedforward_mult=6`,
`8192` training shots, and a `180s` training budget. The repeated
primary-benchmark evidence recorded in
[results/overnight/hermes-d3-final-promotion-20260324.md](./results/overnight/hermes-d3-final-promotion-20260324.md)
reports aggregate MWPM ratios `0.9420`, `0.9565`, and `0.9783`, for a mean of
`0.959` on the `train8192 / val1024` benchmark.

See [docs/human/results.md](./docs/human/results.md) for the benchmark policy,
evidence trail, figures, and limitations.

## Quickstart

This first run keeps the repository clean by writing checkpoints and evaluation
outputs to `/tmp`.

```bash
uv sync --all-extras
uv run prepare.py --workspace . --profile local-d3-v1 --train-shots 1024 --val-shots 256
uv run train.py --workspace . --dataset-manifest data/local-d3-v1-d3-r3-5rates-train1024-val256/manifest.json --duration-seconds 30 --device auto --checkpoint-dir /tmp/nanoqec-quickstart/checkpoints --results-dir /tmp/nanoqec-quickstart/results/train --experiment-log /tmp/nanoqec-quickstart/results/experiments.jsonl --skip-experiment-log
uv run eval.py --workspace . --dataset-manifest data/local-d3-v1-d3-r3-5rates-train1024-val256/manifest.json --checkpoint /tmp/nanoqec-quickstart/checkpoints/best.pt --device auto --results-dir /tmp/nanoqec-quickstart/results/eval
```

Successful completion should leave you with:

- `data/local-d3-v1-d3-r3-5rates-train1024-val256/manifest.json`
- `/tmp/nanoqec-quickstart/checkpoints/best.pt`
- `/tmp/nanoqec-quickstart/results/train/<run-id>.json`
- `/tmp/nanoqec-quickstart/results/eval/best-eval.json`

For a fuller walkthrough, failure recovery, and the canonical promoted benchmark
lane, start with [docs/human/quickstart.md](./docs/human/quickstart.md).

## Human Docs

- [Start here](./docs/human/index.md)
- [Project overview](./docs/human/overview.md)
- [Quickstart](./docs/human/quickstart.md)
- [Concepts](./docs/human/concepts.md)
- [Results and evidence](./docs/human/results.md)
- [Architecture](./docs/human/architecture.md)
- [Using Hermes safely](./docs/human/using-hermes.md)
- [Contributing](./docs/human/contributing.md)
- [FAQ](./docs/human/faq.md)
- [Glossary](./docs/human/glossary.md)

## Authority Docs

These remain the source of truth for behavior and policy:

- [AGENTS.md](./AGENTS.md)
- [docs/implementation-v0.md](./docs/implementation-v0.md)
- [docs/hermes-ops.md](./docs/hermes-ops.md)

Human-facing docs summarize those contracts for readability. They do not
override them.
