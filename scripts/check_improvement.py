"""Compare a metrics artifact against prior experiment history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--experiment-log", type=Path, default=Path("results/experiments.jsonl"))
    return parser.parse_args(argv)


def load_primary_val_ler(payload: dict[str, Any]) -> float:
    if "aggregate_val_ler" in payload:
        return float(payload["aggregate_val_ler"])
    if "val_ler" in payload:
        return float(payload["val_ler"])
    raise ValueError("metrics payload does not contain a primary validation LER")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics_payload = json.loads(args.metrics_json.read_text())
    candidate_val_ler = load_primary_val_ler(metrics_payload)
    previous_best: float | None = None
    if args.experiment_log.exists():
        for line in args.experiment_log.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            metrics = row.get("metrics", {})
            if row.get("dataset_id") != metrics_payload.get("dataset_id"):
                continue
            try:
                row_primary = load_primary_val_ler(metrics)
            except ValueError:
                continue
            previous_best = (
                row_primary
                if previous_best is None
                else min(previous_best, row_primary)
            )
    improved = previous_best is None or candidate_val_ler < previous_best
    print(
        json.dumps(
            {
                "dataset_id": metrics_payload.get("dataset_id"),
                "candidate_val_ler": candidate_val_ler,
                "previous_best_val_ler": previous_best,
                "improved": improved,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
