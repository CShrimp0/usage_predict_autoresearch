"""Utilities for tracking feature names through sklearn pipelines."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.pipeline import Pipeline


def get_step_feature_names(step, input_features: Sequence[str]) -> np.ndarray:
    """Get feature names after applying one fitted transformer."""
    if hasattr(step, "get_feature_names_out"):
        return np.asarray(step.get_feature_names_out(input_features), dtype=object)
    return np.asarray(input_features, dtype=object)


def get_pipeline_feature_names(fitted_pipeline: Pipeline, input_features: Sequence[str]) -> np.ndarray:
    """Propagate feature names through a fitted pipeline, excluding the final estimator."""
    features = np.asarray(input_features, dtype=object)
    for _, step in fitted_pipeline.steps[:-1]:
        features = get_step_feature_names(step, features)
    return features


def transform_features(fitted_pipeline: Pipeline, x_frame):
    """Transform data with all steps except the final estimator."""
    transformed = x_frame
    for _, step in fitted_pipeline.steps[:-1]:
        transformed = step.transform(transformed)
    return transformed
