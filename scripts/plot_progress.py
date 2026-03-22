"""Render a progress plot from the runtime experiment log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-log", type=Path, default=Path("results/experiments.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_ids: list[str] = []
    val_lers: list[float] = []
    mwpm_ratios: list[float] = []
    for line in args.experiment_log.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        metrics = row.get("metrics", {})
        primary_val_ler = metrics.get("aggregate_val_ler", metrics.get("val_ler"))
        primary_mwpm_ratio = metrics.get("aggregate_mwpm_ratio", metrics.get("mwpm_ratio"))
        if primary_val_ler is None or primary_mwpm_ratio is None:
            continue
        run_ids.append(str(row.get("run_id")))
        val_lers.append(float(primary_val_ler))
        mwpm_ratios.append(float(primary_mwpm_ratio))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(range(len(run_ids)), val_lers, marker="o")
    axes[0].set_ylabel("Aggregate val LER")
    axes[0].grid(alpha=0.3)
    axes[1].plot(range(len(run_ids)), mwpm_ratios, marker="s")
    axes[1].set_ylabel("Aggregate MWPM ratio")
    axes[1].set_xlabel("Experiment index")
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(args.output)
    plt.close(figure)
    print(json.dumps({"output": str(args.output), "points": len(run_ids)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
