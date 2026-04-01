"""Filter-based feature selection methods."""

from __future__ import annotations

from sklearn.feature_selection import SelectKBest, SelectPercentile, f_regression, mutual_info_regression


def build_filter_selector(selection_config: dict):
    """Construct a univariate feature selector."""
    method = selection_config.get("method", "univariate_k_best")
    params = selection_config.get("params", {}) or {}
    score_name = params.get("score_func", "mutual_info")
    score_func = mutual_info_regression if score_name == "mutual_info" else f_regression

    if method == "univariate_k_best":
        return SelectKBest(score_func=score_func, k=params.get("k", 50))
    if method == "univariate_percentile":
        return SelectPercentile(score_func=score_func, percentile=params.get("percentile", 20))
    raise ValueError(f"Unsupported filter selection method: {method}")
