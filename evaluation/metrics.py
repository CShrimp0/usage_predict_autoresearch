"""Regression metrics for age prediction."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute core regression metrics used by the project."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "pearson_r": np.nan,
            "spearman_rho": np.nan,
            "bias": np.nan,
            "residual_std": np.nan,
        }
    residuals = y_pred - y_true
    pearson = np.nan
    spearman = np.nan
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson = pearsonr(y_true, y_pred)[0]
        spearman = spearmanr(y_true, y_pred)[0]
    try:
        r2 = float(r2_score(y_true, y_pred))
    except ValueError:
        r2 = np.nan
    try:
        rmse = float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": r2,
        "pearson_r": float(pearson),
        "spearman_rho": float(spearman),
        "bias": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0,
    }


def residual_summary(y_true, y_pred) -> dict[str, float]:
    """Summarize residual trend diagnostics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_pred - y_true
    if len(y_true) < 2:
        return {
            "residual_vs_age_slope": np.nan,
            "residual_vs_age_intercept": np.nan,
            "pred_vs_true_slope": np.nan,
            "pred_vs_true_intercept": np.nan,
        }
    slope, intercept = np.polyfit(y_true, residuals, deg=1)
    pred_slope, pred_intercept = np.polyfit(y_true, y_pred, deg=1)
    return {
        "residual_vs_age_slope": float(slope),
        "residual_vs_age_intercept": float(intercept),
        "pred_vs_true_slope": float(pred_slope),
        "pred_vs_true_intercept": float(pred_intercept),
    }
