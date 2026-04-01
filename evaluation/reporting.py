"""Artifact writing and run summary generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.io import save_json


def make_output_layout(output_dir: str | Path) -> dict[str, Path]:
    """Create a structured output layout for one run."""
    root = Path(output_dir)
    layout = {
        "root": root,
        "tables": root / "tables",
        "figures": root / "figures",
        "models": root / "models",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


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
    layout = make_output_layout(output_dir)
    save_json(metrics, layout["tables"] / "metrics.json")
    predictions.to_csv(layout["tables"] / "predictions.csv", index=False)
    save_readable_predictions(predictions, layout["root"] / "predictions_readable.csv")
    feature_importance.to_csv(layout["tables"] / "feature_importance.csv", index=False)
    pd.DataFrame({"feature": selected_features}).to_csv(layout["tables"] / "selected_features.csv", index=False)
    save_json(split_info, layout["tables"] / "split_info.json")
    if age_bin_metrics is not None:
        age_bin_metrics.to_csv(layout["tables"] / "age_bin_metrics.csv", index=False)
    if subgroup_metrics is not None:
        subgroup_metrics.to_csv(layout["tables"] / "subgroup_metrics.csv", index=False)
    save_json(
        {
            "quick_view_files": {
                "summary": "run_summary.md",
                "readable_predictions": "predictions_readable.csv",
                "tables_dir": "tables/",
                "figures_dir": "figures/",
                "models_dir": "models/",
            }
        },
        layout["root"] / "results_overview.json",
    )


def save_readable_predictions(predictions: pd.DataFrame, path: str | Path) -> None:
    """Write a more human-readable prediction table."""
    frame = predictions.copy()
    if "prediction" in frame.columns and "age" in frame.columns:
        frame["pred_age"] = frame["prediction"].round(3)
        frame["true_age"] = frame["age"].round(3)
        frame["error_signed"] = (frame["prediction"] - frame["age"]).round(3)
        frame["error_abs"] = (frame["prediction"] - frame["age"]).abs().round(3)
    rename_map = {
        "sample_id": "sample_id",
        "subject_id": "subject_id",
        "split": "split",
        "fold": "fold",
        "sex": "sex",
        "bmi": "bmi",
    }
    preferred_columns = [
        "sample_id",
        "subject_id",
        "split",
        "fold",
        "true_age",
        "pred_age",
        "error_signed",
        "error_abs",
        "sex",
        "bmi",
    ]
    remaining = [column for column in frame.columns if column not in preferred_columns and column not in {"prediction", "age"}]
    ordered = [column for column in preferred_columns if column in frame.columns] + remaining
    frame = frame[ordered].sort_values(
        by=[column for column in ["error_abs", "subject_id", "sample_id"] if column in frame.columns],
        ascending=[False, True, True][: len([column for column in ["error_abs", "subject_id", "sample_id"] if column in frame.columns])],
    )
    frame.to_csv(path, index=False)


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
        "## Where To Look",
        "",
        "- Quick summary: `run_summary.md`",
        "- Human-readable predictions: `predictions_readable.csv`",
        "- Detailed tables: `tables/`",
        "- Plots: `figures/`",
        "- Saved models: `models/`",
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
