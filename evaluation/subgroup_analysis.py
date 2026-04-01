"""Subgroup and age-bin error analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.metrics import compute_regression_metrics


def analyze_age_bins(
    frame: pd.DataFrame,
    age_column: str = "age",
    pred_column: str = "prediction",
    bin_edges: list[float] | None = None,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Compute metrics by age bins."""
    data = frame.copy()
    if bin_edges:
        data["age_bin"] = pd.cut(data[age_column], bins=bin_edges, include_lowest=True)
    else:
        data["age_bin"] = pd.qcut(data[age_column], q=n_bins, duplicates="drop")
    rows = []
    for label, group in data.groupby("age_bin", observed=False):
        if group.empty:
            continue
        metrics = compute_regression_metrics(group[age_column], group[pred_column])
        rows.append({"age_bin": str(label), "n": int(len(group)), **metrics})
    return pd.DataFrame(rows)


def analyze_subgroups(
    frame: pd.DataFrame,
    subgroup_columns: list[str],
    age_column: str = "age",
    pred_column: str = "prediction",
) -> pd.DataFrame:
    """Compute metrics by subgroup values."""
    rows = []
    for column in subgroup_columns:
        if column not in frame.columns:
            continue
        for value, group in frame.groupby(column, dropna=False):
            metrics = compute_regression_metrics(group[age_column], group[pred_column])
            rows.append(
                {
                    "subgroup_column": column,
                    "subgroup_value": str(value),
                    "n": int(len(group)),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)
