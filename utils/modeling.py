"""Shared model fitting helpers."""

from __future__ import annotations

from copy import deepcopy

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from models.registry import get_search_space
from preprocessing.pipelines import build_regression_pipeline
from preprocessing.split import make_group_cv_splitter


def build_search_estimator(pipeline, config: dict):
    """Wrap a pipeline with optional grid/random search."""
    search_config = config.get("search", {}) or {}
    if not search_config.get("enabled", False):
        return pipeline

    model_name = config["model"]["name"]
    user_grid = (search_config.get("param_grid", {}) or {}).get(model_name, {})
    search_space = user_grid or get_search_space(model_name)
    if not search_space:
        return pipeline

    splitter_config = deepcopy(config)
    splitter_config["split"] = deepcopy(config["split"])
    splitter_config["split"]["n_splits"] = int(search_config.get("cv_n_splits", 5))
    cv_splitter = make_group_cv_splitter(splitter_config)
    common_kwargs = {
        "estimator": pipeline,
        "cv": cv_splitter,
        "scoring": search_config.get("scoring", "neg_mean_absolute_error"),
        "refit": search_config.get("refit", "neg_mean_absolute_error"),
        "n_jobs": search_config.get("n_jobs"),
        "verbose": int(search_config.get("verbose", 0)),
    }
    if search_config.get("method", "grid") == "random":
        return RandomizedSearchCV(
            param_distributions=search_space,
            n_iter=int(search_config.get("n_iter", 20)),
            random_state=int(config.get("seed", 42)),
            **common_kwargs,
        )
    return GridSearchCV(param_grid=search_space, **common_kwargs)


def build_pipeline_and_search(column_spec, config: dict):
    """Create the base pipeline and optional search wrapper."""
    pipeline = build_regression_pipeline(column_spec, config, random_state=int(config.get("seed", 42)))
    return build_search_estimator(pipeline, config)


def unwrap_best_estimator(estimator):
    """Return the fitted pipeline regardless of whether search was used."""
    return getattr(estimator, "best_estimator_", estimator)


def extract_best_params(estimator) -> dict:
    """Return best params if search was used, otherwise the configured params."""
    return getattr(estimator, "best_params_", {})
