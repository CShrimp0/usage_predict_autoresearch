"""Plotting helpers for experiment outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_predicted_vs_true(frame: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(frame["age"], frame["prediction"], alpha=0.75, s=28)
    min_val = min(frame["age"].min(), frame["prediction"].min())
    max_val = max(frame["age"].max(), frame["prediction"].max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black")
    if len(frame) >= 2:
        slope, intercept = np.polyfit(frame["age"], frame["prediction"], deg=1)
        x = np.linspace(min_val, max_val, 100)
        ax.plot(x, slope * x + intercept, color="tab:red", linewidth=2)
    ax.set_xlabel("True Age")
    ax.set_ylabel("Predicted Age")
    ax.set_title("Predicted vs True Age")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_bland_altman(frame: pd.DataFrame, path: str | Path) -> None:
    pred = frame["prediction"].to_numpy()
    true = frame["age"].to_numpy()
    mean_age = (pred + true) / 2.0
    diff = pred - true
    bias = diff.mean()
    std = diff.std(ddof=1) if len(diff) > 1 else 0.0
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mean_age, diff, alpha=0.75, s=28)
    ax.axhline(bias, color="tab:red", linestyle="-", label="Bias")
    ax.axhline(bias + 1.96 * std, color="tab:gray", linestyle="--", label="95% LoA")
    ax.axhline(bias - 1.96 * std, color="tab:gray", linestyle="--")
    ax.set_xlabel("Mean of True/Predicted Age")
    ax.set_ylabel("Prediction Error")
    ax.set_title("Bland-Altman Plot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_residual_hist(frame: pd.DataFrame, path: str | Path) -> None:
    residuals = frame["prediction"] - frame["age"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=20, color="tab:blue", alpha=0.85)
    ax.axvline(residuals.mean(), color="tab:red", linestyle="--", label="Bias")
    ax.set_xlabel("Residual (Pred - True)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Histogram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_residual_vs_age(frame: pd.DataFrame, path: str | Path) -> None:
    residuals = frame["prediction"] - frame["age"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(frame["age"], residuals, alpha=0.75, s=28)
    if len(frame) >= 2:
        slope, intercept = np.polyfit(frame["age"], residuals, deg=1)
        x = np.linspace(frame["age"].min(), frame["age"].max(), 100)
        ax.plot(x, slope * x + intercept, color="tab:red", linewidth=2)
    ax.axhline(0.0, color="black", linestyle="--")
    ax.set_xlabel("True Age")
    ax.set_ylabel("Residual (Pred - True)")
    ax.set_title("Residual vs Age")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_age_bin_error(age_bin_metrics: pd.DataFrame, path: str | Path) -> None:
    if age_bin_metrics.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(age_bin_metrics["age_bin"].astype(str), age_bin_metrics["mae"], color="tab:orange")
    ax.set_xlabel("Age Bin")
    ax.set_ylabel("MAE")
    ax.set_title("Age-bin Error")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_feature_importance_top20(frame: pd.DataFrame, path: str | Path) -> None:
    if frame.empty:
        return
    top = frame.sort_values("importance", ascending=False).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top["feature"], top["importance"], color="tab:green")
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 Feature Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
