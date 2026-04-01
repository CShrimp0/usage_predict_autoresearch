#!/usr/bin/env python3
"""Aggregate feature stability and metrics from one or more run directories."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from selection.stability import aggregate_importances
from utils.io import load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze existing run directories.")
    parser.add_argument("--run-dir", action="append", required=True, help="Path to a completed run directory.")
    parser.add_argument("--output-dir", required=True, help="Directory to write aggregated analysis.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    importance_frames = []
    for run_dir in args.run_dir:
        run_path = Path(run_dir)
        metrics = load_json(run_path / "metrics.json")
        metrics_rows.append({"run_dir": str(run_path), **metrics.get("outer_cv_pooled", metrics.get("test", metrics))})
        importance_path = run_path / "feature_importance.csv"
        if importance_path.exists():
            frame = pd.read_csv(importance_path)
            if "importance" not in frame.columns and "mean_importance" in frame.columns:
                frame["importance"] = frame["mean_importance"]
            frame["run_dir"] = str(run_path)
            importance_frames.append(frame)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False)

    aggregated_importance = aggregate_importances(importance_frames)
    aggregated_importance.to_csv(output_dir / "feature_importance_summary.csv", index=False)
    save_json({"n_runs": len(args.run_dir)}, output_dir / "summary.json")


if __name__ == "__main__":
    main()
