"""Optional SHAP helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("usage_predict_feature_engineering")


def compute_shap_importance(estimator, transformed_x, feature_names) -> pd.DataFrame:
    """Compute mean absolute SHAP values when shap is available."""
    try:
        import shap
    except ImportError:
        LOGGER.warning("shap is not installed; SHAP importance will be skipped.")
        return pd.DataFrame(columns=["feature", "importance", "importance_type"])

    try:
        explainer = shap.Explainer(estimator, transformed_x)
        shap_values = explainer(transformed_x)
        values = getattr(shap_values, "values", shap_values)
        importance = np.mean(np.abs(values), axis=0)
    except Exception as exc:  # pragma: no cover - optional dependency path
        LOGGER.warning("SHAP computation failed: %s", exc)
        return pd.DataFrame(columns=["feature", "importance", "importance_type"])

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance,
                "importance_type": "shap_mean_abs",
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_shap_topk(
    shap_frame: pd.DataFrame,
    output_dir: str | Path,
    top_k: int = 20,
    prefix: str = "shap",
) -> pd.DataFrame:
    """Save SHAP top-k results and return the saved frame."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    top_frame = shap_frame.sort_values("importance", ascending=False).head(top_k).reset_index(drop=True)
    top_frame.to_csv(output_path / f"{prefix}_top{top_k}.csv", index=False)
    return top_frame
