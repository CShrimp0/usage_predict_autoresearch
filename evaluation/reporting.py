"""Artifact writing and rich run-summary generation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
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
    raw_feature_columns: list[str] | None = None,
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
    if raw_feature_columns is not None:
        feature_rows = []
        for column in raw_feature_columns:
            parts = column.split("__")
            feature_rows.append(
                {
                    "feature": column,
                    "scope": parts[0] if len(parts) > 0 else "",
                    "group": parts[1] if len(parts) > 1 else "",
                    "name": "__".join(parts[2:]) if len(parts) > 2 else "",
                }
            )
        pd.DataFrame(feature_rows).to_csv(layout["tables"] / "extracted_feature_names.csv", index=False)
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
                "shap_dir": "shap/ (if enabled)",
                "extracted_feature_names": "tables/extracted_feature_names.csv",
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
    sort_columns = [column for column in ["error_abs", "fold", "subject_id", "sample_id"] if column in frame.columns]
    ascending = []
    for column in sort_columns:
        ascending.append(False if column == "error_abs" else True)
    frame = frame[ordered].sort_values(by=sort_columns, ascending=ascending)
    frame.to_csv(path, index=False)


def _format_scalar(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return "NaN"
        return f"{float(value):.4f}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value)


def _dataframe_to_markdown(frame: pd.DataFrame, columns: Iterable[str] | None = None, max_rows: int = 10) -> str:
    if frame is None or frame.empty:
        return "_无可展示内容_"
    data = frame.copy()
    if columns is not None:
        keep = [column for column in columns if column in data.columns]
        data = data[keep]
    data = data.head(max_rows)
    headers = list(data.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in data.iterrows():
        lines.append("| " + " | ".join(_format_scalar(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def _pick_primary_importance_table(feature_importance: pd.DataFrame) -> pd.DataFrame:
    if feature_importance is None or feature_importance.empty:
        return pd.DataFrame(columns=["feature", "importance"])
    if "importance_type" not in feature_importance.columns:
        return feature_importance.sort_values("importance", ascending=False).reset_index(drop=True)
    priority = ["permutation", "coefficient_abs", "impurity"]
    for kind in priority:
        subset = feature_importance[feature_importance["importance_type"] == kind].copy()
        if not subset.empty:
            return subset.sort_values("importance", ascending=False).reset_index(drop=True)
    return feature_importance.sort_values("importance", ascending=False).reset_index(drop=True)


def _build_feature_group_summary(raw_feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in raw_feature_columns:
        parts = column.split("__")
        scope = parts[0] if len(parts) > 0 else "unknown"
        group = parts[1] if len(parts) > 1 else "unknown"
        rows.append({"scope": scope, "group": group})
    if not rows:
        return pd.DataFrame(columns=["scope", "group", "n_features"])
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(["scope", "group"])
        .size()
        .reset_index(name="n_features")
        .sort_values(["scope", "group"])
        .reset_index(drop=True)
    )


def _build_error_table(predictions: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    if predictions is None or predictions.empty:
        return pd.DataFrame()
    frame = predictions.copy()
    frame["error_signed"] = frame["prediction"] - frame["age"]
    frame["error_abs"] = np.abs(frame["error_signed"])
    columns = [column for column in ["sample_id", "subject_id", "fold", "split", "age", "prediction", "error_signed", "error_abs", "sex", "bmi"] if column in frame.columns]
    return frame.sort_values("error_abs", ascending=False)[columns].head(top_k).reset_index(drop=True)


def _metric_table_from_mapping(mapping: dict[str, float], title: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": list(mapping.keys()),
            title: [mapping[key] for key in mapping.keys()],
        }
    )


def _interpret_metrics(metrics: dict, diagnostics: dict) -> list[str]:
    lines = []
    mae = metrics.get("mae", np.nan)
    bias = metrics.get("bias", np.nan)
    pred_slope = diagnostics.get("pred_vs_true_slope", np.nan)
    residual_slope = diagnostics.get("residual_vs_age_slope", np.nan)

    lines.append(f"平均绝对误差 MAE 为 {mae:.2f} 岁，表示单个样本预测年龄与真实年龄平均相差约 {mae:.2f} 岁。")
    if np.isfinite(bias):
        if bias > 0.5:
            lines.append(f"Bias 为 {bias:.2f}，整体上偏向预测得更老。")
        elif bias < -0.5:
            lines.append(f"Bias 为 {bias:.2f}，整体上偏向预测得更年轻。")
        else:
            lines.append(f"Bias 为 {bias:.2f}，整体系统性偏差不明显。")
    if np.isfinite(pred_slope):
        if pred_slope < 0.9:
            lines.append(f"`pred vs true` 斜率为 {pred_slope:.3f}，小于 1，存在一定回归到均值现象。")
        elif pred_slope > 1.1:
            lines.append(f"`pred vs true` 斜率为 {pred_slope:.3f}，大于 1，模型对年龄变化较为敏感。")
        else:
            lines.append(f"`pred vs true` 斜率为 {pred_slope:.3f}，接近 1，校准关系较为合理。")
    if np.isfinite(residual_slope):
        if residual_slope < -0.05:
            lines.append("残差随年龄呈负斜率，说明年轻样本更容易被预测偏老、年长样本更容易被预测偏年轻。")
        elif residual_slope > 0.05:
            lines.append("残差随年龄呈正斜率，说明年轻样本更容易被预测偏年轻、年长样本更容易被预测偏老。")
        else:
            lines.append("残差随年龄变化趋势较弱，未见明显年龄相关系统误差。")
    return lines


def _figure_section(lines: list[str], run_dir: Path, shap_enabled: bool) -> None:
    figure_candidates = [
        ("预测值 vs 真实值", "figures/predicted_vs_true.png"),
        ("残差 vs 年龄", "figures/residual_vs_age.png"),
        ("Bland-Altman 图", "figures/bland_altman.png"),
        ("年龄分桶误差", "figures/age_bin_error.png"),
        ("普通重要性 Top20", "figures/feature_importance_top20.png"),
    ]
    lines.extend(["## 图表总览", ""])
    for title, rel_path in figure_candidates:
        if (run_dir / rel_path).exists():
            lines.append(f"### {title}")
            lines.append("")
            lines.append(f"![{title}]({rel_path})")
            lines.append("")
    if shap_enabled:
        shap_plots = sorted((run_dir / "shap").glob("shap_top*.png"))
        for path in shap_plots:
            rel_path = path.relative_to(run_dir).as_posix()
            lines.append("### SHAP 重要性图")
            lines.append("")
            lines.append(f"![SHAP 重要性图]({rel_path})")
            lines.append("")


def build_run_summary(
    run_dir: str | Path,
    config: dict,
    mode: str,
    primary_metrics: dict,
    split_info: dict,
    diagnostics: dict,
    predictions: pd.DataFrame,
    raw_feature_columns: list[str],
    selected_features: list[str],
    feature_importance: pd.DataFrame,
    age_bin_metrics: pd.DataFrame | None = None,
    subgroup_metrics: pd.DataFrame | None = None,
    validation_metrics: dict | None = None,
    fold_metrics_df: pd.DataFrame | None = None,
    shap_top: pd.DataFrame | None = None,
) -> str:
    """Create a rich Chinese Markdown report for one run."""
    run_path = Path(run_dir)
    model_name = config.get("model", {}).get("name", "unknown_model")
    experiment_name = config.get("experiment", {}).get("name", "experiment")
    feature_mode = config.get("data", {}).get("feature_mode", "image_only")
    age_filter = config.get("data", {}).get("age_filter", {})
    shap_requested = bool(config.get("explain", {}).get("shap", {}).get("enabled", False))
    shap_available = shap_top is not None and not shap_top.empty
    primary_importance = _pick_primary_importance_table(feature_importance)
    feature_group_summary = _build_feature_group_summary(raw_feature_columns)
    error_table = _build_error_table(predictions, top_k=10)

    lines = [
        "# 本次实验结果总览",
        "",
        "## 一句话结论",
        "",
        f"本次使用 `{model_name}` 在 `{experiment_name}` 设置下完成 `{mode}` 评估。主结果 MAE 为 `{primary_metrics.get('mae', np.nan):.3f}` 岁，RMSE 为 `{primary_metrics.get('rmse', np.nan):.3f}`，R2 为 `{primary_metrics.get('r2', np.nan):.3f}`。",
        "",
        "## 实验设置",
        "",
        f"- 实验名：`{experiment_name}`",
        f"- 模型：`{model_name}`",
        f"- 运行模式：`{mode}`",
        f"- 特征模式：`{feature_mode}`",
        f"- 年龄过滤：`{age_filter.get('min_age', 'None')}` 到 `{age_filter.get('max_age', 'None')}` 岁",
        f"- SHAP：`{'已生成' if shap_available else ('已请求但未生成' if shap_requested else '未启用')}`",
        f"- 原始显式特征数：`{len(raw_feature_columns)}`",
        f"- 最终进入模型的特征数：`{len(selected_features)}`",
        f"- 评估样本数：`{len(predictions)}`",
        f"- 评估受试者数：`{predictions['subject_id'].nunique() if 'subject_id' in predictions.columns else 'N/A'}`",
        "",
        "## 结果解读",
        "",
    ]
    for sentence in _interpret_metrics(primary_metrics, diagnostics):
        lines.append(f"- {sentence}")

    lines.extend(["", "## 核心指标", ""])
    if validation_metrics is not None:
        metric_table = _metric_table_from_mapping(primary_metrics, "test")
        metric_table["validation"] = [validation_metrics.get(metric, np.nan) for metric in metric_table["metric"]]
        metric_table = metric_table[["metric", "validation", "test"]]
        lines.append(_dataframe_to_markdown(metric_table, max_rows=len(metric_table)))
    else:
        metric_table = _metric_table_from_mapping(primary_metrics, "outer_cv_pooled")
        lines.append(_dataframe_to_markdown(metric_table, max_rows=len(metric_table)))
        if fold_metrics_df is not None and not fold_metrics_df.empty:
            lines.extend(["", "外层各折结果均值/标准差：", ""])
            summary_rows = []
            metric_columns = [column for column in fold_metrics_df.columns if column != "fold"]
            for column in metric_columns:
                summary_rows.append(
                    {
                        "metric": column,
                        "mean": fold_metrics_df[column].mean(),
                        "std": fold_metrics_df[column].std(),
                    }
                )
            lines.append(_dataframe_to_markdown(pd.DataFrame(summary_rows), max_rows=len(summary_rows)))

    lines.extend(["", "## 数据与划分", ""])
    split_rows = [{"item": key, "value": str(value)} for key, value in split_info.items()]
    lines.append(_dataframe_to_markdown(pd.DataFrame(split_rows), max_rows=len(split_rows)))

    lines.extend(["", "## 提取到的特征组成", ""])
    lines.append(_dataframe_to_markdown(feature_group_summary, max_rows=len(feature_group_summary)))

    lines.extend(["", "## 最重要的普通特征", ""])
    if primary_importance.empty:
        lines.append("_未生成普通 feature importance。_")
    else:
        lines.append(_dataframe_to_markdown(primary_importance, columns=["feature", "importance", "importance_type"], max_rows=10))

    lines.extend(["", "## Top SHAP 特征", ""])
    if shap_available:
        lines.append("以下表格展示本次运行中全局平均绝对 SHAP 最高的前 10 个特征。")
        lines.append("")
        lines.append(_dataframe_to_markdown(shap_top, columns=["feature", "importance"], max_rows=min(10, len(shap_top))))
    elif shap_requested:
        lines.append("_本次配置请求了 SHAP，但没有生成结果。常见原因是未安装 `shap` 包，或当前模型/输入触发了 SHAP 计算失败。_")
    else:
        lines.append("_本次未启用 SHAP。_")

    lines.extend(["", "## 年龄分桶误差", ""])
    lines.append(_dataframe_to_markdown(age_bin_metrics, columns=["age_bin", "n", "mae", "rmse", "bias"], max_rows=10))

    lines.extend(["", "## 亚组分析", ""])
    lines.append(_dataframe_to_markdown(subgroup_metrics, columns=["subgroup_column", "subgroup_value", "n", "mae", "rmse", "bias"], max_rows=10))

    lines.extend(["", "## 误差最大的样本 Top10", ""])
    lines.append(_dataframe_to_markdown(error_table, columns=["sample_id", "subject_id", "fold", "split", "age", "prediction", "error_signed", "error_abs"], max_rows=10))

    _figure_section(lines, run_path, shap_enabled=shap_available)

    lines.extend(
        [
            "## 文件导航",
            "",
            "- `run_summary.md`：当前这份中文总报告。",
            "- `predictions_readable.csv`：按误差排序的样本级预测结果，适合人工阅读。",
            "- `tables/metrics.json`：完整指标明细。",
            "- `tables/features_raw.csv`：全部提取到的显式特征。",
            "- `tables/extracted_feature_names.csv`：全部提取到的显式特征名，不含具体特征值。",
            "- `tables/selected_features.csv`：最终进入模型的特征名。",
            "- `tables/feature_importance.csv`：普通 importance 结果。",
            "- `shap/`：SHAP 专用结果目录，仅在成功生成 SHAP 时存在。",
            "- `figures/`：图像版评估结果。",
            "- `models/`：训练好的模型文件。",
            "",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def save_run_summary(summary: str, path: str | Path) -> None:
    """Write the markdown run summary."""
    Path(path).write_text(summary, encoding="utf-8")
