"""Model importance extraction utilities."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.inspection import permutation_importance

from explain.coefficients import coefficient_importance
from utils.feature_names import get_pipeline_feature_names, transform_features

LOGGER = logging.getLogger("usage_predict_feature_engineering")


def extract_selected_feature_names(fitted_pipeline, input_columns: list[str]) -> list[str]:
    """Return the final selected/transformed feature names from a fitted pipeline."""
    return list(get_pipeline_feature_names(fitted_pipeline, input_columns))


def compute_importance(
    fitted_pipeline,
    x_eval,
    y_eval,
    input_columns: list[str],
    config: dict,
) -> pd.DataFrame:
    """Compute model-specific and permutation-based feature importance."""
    estimator = fitted_pipeline.named_steps["model"]
    feature_names = extract_selected_feature_names(fitted_pipeline, input_columns)
    transformed = transform_features(fitted_pipeline, x_eval)
    importance_frames: list[pd.DataFrame] = []

    if hasattr(estimator, "coef_"):
        importance_frames.append(coefficient_importance(feature_names, estimator.coef_))

    if hasattr(estimator, "feature_importances_"):
        frame = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": estimator.feature_importances_,
                "importance_type": "impurity",
            }
        )
        importance_frames.append(frame.sort_values("importance", ascending=False).reset_index(drop=True))

    permutation_config = config.get("explain", {}).get("permutation_importance", {}) or {}
    if permutation_config.get("enabled", True) and len(feature_names) > 0:
        result = permutation_importance(
            estimator,
            transformed,
            y_eval,
            n_repeats=int(permutation_config.get("n_repeats", 20)),
            random_state=int(config.get("seed", 42)),
            scoring=config.get("search", {}).get("refit", "neg_mean_absolute_error"),
            n_jobs=permutation_config.get("n_jobs"),
        )
        frame = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": result.importances_mean,
                "importance_std": result.importances_std,
                "importance_type": "permutation",
            }
        )
        importance_frames.append(frame.sort_values("importance", ascending=False).reset_index(drop=True))

    if not importance_frames:
        LOGGER.warning("No importance method applicable for model '%s'.", estimator.__class__.__name__)
        return pd.DataFrame(columns=["feature", "importance", "importance_type"])

    combined = pd.concat(importance_frames, ignore_index=True, sort=False)
    combined = combined.sort_values(["importance_type", "importance"], ascending=[True, False]).reset_index(drop=True)
    return combined
