"""Selection and importance stability summaries across folds."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


def summarize_selected_features(selected_feature_lists: list[list[str]]) -> pd.DataFrame:
    """Count how often each feature is selected across folds."""
    counter: Counter[str] = Counter()
    for features in selected_feature_lists:
        counter.update(features)
    total_folds = max(len(selected_feature_lists), 1)
    rows = [
        {
            "feature": feature,
            "count": count,
            "selection_frequency": count / total_folds,
        }
        for feature, count in counter.most_common()
    ]
    return pd.DataFrame(rows)


def aggregate_importances(importance_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate feature importances across folds."""
    if not importance_frames:
        return pd.DataFrame(columns=["feature", "mean_importance", "std_importance", "n_folds"])
    combined = pd.concat(importance_frames, ignore_index=True)
    if "importance" not in combined.columns:
        return pd.DataFrame(columns=["feature", "mean_importance", "std_importance", "n_folds"])
    grouped = combined.groupby("feature")["importance"]
    return (
        grouped.agg(mean_importance="mean", std_importance="std", n_folds="count")
        .reset_index()
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )
