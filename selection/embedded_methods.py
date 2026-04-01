"""Embedded feature selection methods."""

from __future__ import annotations

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet, Lasso


def build_embedded_selector(selection_config: dict, random_state: int):
    """Construct an embedded feature selector."""
    method = selection_config.get("method", "lasso")
    params = selection_config.get("params", {}) or {}
    threshold = params.get("threshold", "median")

    if method == "lasso":
        estimator = Lasso(alpha=params.get("alpha", 0.01), max_iter=params.get("max_iter", 5000), random_state=random_state)
    elif method == "elasticnet":
        estimator = ElasticNet(
            alpha=params.get("alpha", 0.01),
            l1_ratio=params.get("l1_ratio", 0.5),
            max_iter=params.get("max_iter", 5000),
            random_state=random_state,
        )
    elif method == "tree":
        estimator = ExtraTreesRegressor(
            n_estimators=params.get("n_estimators", 500),
            max_depth=params.get("max_depth"),
            random_state=random_state,
            n_jobs=params.get("n_jobs"),
        )
    else:
        raise ValueError(f"Unsupported embedded selection method: {method}")

    return SelectFromModel(estimator=estimator, threshold=threshold)
