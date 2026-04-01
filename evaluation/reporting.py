"""Artifact writing and run summary generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.io import save_json


def save_core_outputs(
    output_dir: str | Path,
    metrics: dict,
    predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
    selected_features: list[str],
    split_info: dict,
    age_bin_metrics: pd.DataFrame | None = None,
    subgroup_metrics: pd.DataFrame | None = None,
) -> None:
    """Write the standard output files required by the project."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_json(metrics, output_path / "metrics.json")
    predictions.to_csv(output_path / "predictions.csv", index=False)
    feature_importance.to_csv(output_path / "feature_importance.csv", index=False)
    pd.DataFrame({"feature": selected_features}).to_csv(output_path / "selected_features.csv", index=False)
    save_json(split_info, output_path / "split_info.json")
    if age_bin_metrics is not None:
        age_bin_metrics.to_csv(output_path / "age_bin_metrics.csv", index=False)
    if subgroup_metrics is not None:
        subgroup_metrics.to_csv(output_path / "subgroup_metrics.csv", index=False)


def build_run_summary(
    metrics: dict,
    split_info: dict,
    top_features: pd.DataFrame,
    diagnostics: dict,
    model_name: str,
) -> str:
    """Create a concise Markdown run summary."""
    lines = [
        "# Run Summary",
        "",
        f"- Model: `{model_name}`",
        f"- MAE: `{metrics.get('mae', float('nan')):.4f}`",
        f"- RMSE: `{metrics.get('rmse', float('nan')):.4f}`",
        f"- R2: `{metrics.get('r2', float('nan')):.4f}`",
        f"- Pearson r: `{metrics.get('pearson_r', float('nan')):.4f}`",
        f"- Spearman rho: `{metrics.get('spearman_rho', float('nan')):.4f}`",
        f"- Bias: `{metrics.get('bias', float('nan')):.4f}`",
        "",
        "## Split",
        "",
    ]
    for key, value in split_info.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## RTM Diagnostics", ""])
    for key, value in diagnostics.items():
        lines.append(f"- {key}: `{value:.4f}`")
    lines.extend(["", "## Top Features", ""])
    if top_features.empty:
        lines.append("- No feature importance available.")
    else:
        for _, row in top_features.head(10).iterrows():
            importance = row["importance"]
            feature = row["feature"]
            if hasattr(importance, "iloc"):
                importance = importance.iloc[0]
            if hasattr(feature, "iloc"):
                feature = feature.iloc[0]
            lines.append(f"- `{feature}`: `{float(importance):.6f}`")
    return "\n".join(lines) + "\n"


def save_run_summary(summary: str, path: str | Path) -> None:
    """Write the markdown run summary."""
    Path(path).write_text(summary, encoding="utf-8")
