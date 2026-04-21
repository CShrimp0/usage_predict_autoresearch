"""Wrapper-based feature selection methods."""

from __future__ import annotations

from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVR


def build_wrapper_selector(selection_config: dict, random_state: int):
    """Construct a wrapper feature selector."""
    method = selection_config.get("method", "rfe_linear_svr")
    params = selection_config.get("params", {}) or {}

    if method == "rfe_linear_svr":
        estimator = LinearSVR(
            C=params.get("C", 1.0),
            epsilon=params.get("epsilon", 0.0),
            loss=params.get("loss", "epsilon_insensitive"),
            dual=params.get("dual", "auto"),
            max_iter=params.get("max_iter", 5000),
            random_state=random_state,
        )
        return RFE(
            estimator=estimator,
            n_features_to_select=params.get("n_features_to_select"),
            step=params.get("step", 0.1),
            importance_getter=params.get("importance_getter", "auto"),
        )

    raise ValueError(f"Unsupported wrapper selection method: {method}")
