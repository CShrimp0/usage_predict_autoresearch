#!/usr/bin/env python3
"""Aggregate historical hard-error patterns from predictions_readable files."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS_ROOT = REPO_ROOT / "outputs"
DEFAULT_ANALYSIS_DIR = DEFAULT_OUTPUTS_ROOT / "hard_error_history"

FIELD_ALIASES = {
    "sample_id": ["sample_id", "sample", "SampleID"],
    "subject_id": ["subject_id", "subject", "SubjectID"],
    "fold": ["fold", "Fold"],
    "true_age": ["true_age", "age", "Age", "target", "y_true"],
    "pred_age": ["pred_age", "prediction", "pred", "y_pred"],
    "error_signed": ["error_signed", "signed_error", "error"],
    "error_abs": ["error_abs", "abs_error", "absolute_error"],
    "sex": ["sex", "Sex"],
    "bmi": ["bmi", "BMI"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", default=str(DEFAULT_OUTPUTS_ROOT))
    parser.add_argument("--analysis-dir", default=str(DEFAULT_ANALYSIS_DIR))
    return parser.parse_args()


def _pick_existing_column(frame: pd.DataFrame, aliases: list[str]) -> str | None:
    for candidate in aliases:
        if candidate in frame.columns:
            return candidate
    return None


def _mode_or_nan(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    modes = non_null.mode()
    return modes.iloc[0] if not modes.empty else non_null.iloc[0]


def _infer_run_short(run_name: str) -> str:
    match = re.match(r"(run_\d{8}_\d{6})", run_name)
    return match.group(1) if match else run_name


def _infer_run_family(run_name: str) -> str:
    if "fusion_whole_roi" in run_name:
        return "fusion_whole_roi"
    if "roi_only" in run_name:
        return "roi_only"
    return "whole_image"


def _format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NaN"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return "NaN"
    return f"{numeric:.{digits}f}"


def _format_pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _direction_from_sign(mean_sign: float, consistency: float) -> str:
    if consistency >= 0.95:
        return "pred_older" if mean_sign > 0 else "pred_younger"
    if consistency >= 0.75:
        return "mostly_pred_older" if mean_sign > 0 else "mostly_pred_younger"
    return "mixed"


def _markdown_table(frame: pd.DataFrame, max_rows: int = 10) -> str:
    if frame.empty:
        return "_No rows._"
    data = frame.head(max_rows).copy()
    columns = list(data.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in data.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                if math.isnan(value):
                    values.append("NaN")
                else:
                    values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def standardize_predictions(frame: pd.DataFrame, run_name: str) -> pd.DataFrame:
    standardized = pd.DataFrame()
    for field, aliases in FIELD_ALIASES.items():
        column = _pick_existing_column(frame, aliases)
        if column is not None:
            standardized[field] = frame[column]

    required = {"sample_id", "subject_id", "true_age", "pred_age"}
    missing = sorted(required - set(standardized.columns))
    if missing:
        raise KeyError(f"{run_name}: missing required columns after alias matching: {missing}")

    if "error_signed" not in standardized.columns:
        standardized["error_signed"] = pd.to_numeric(standardized["pred_age"], errors="coerce") - pd.to_numeric(
            standardized["true_age"], errors="coerce"
        )
    if "error_abs" not in standardized.columns:
        standardized["error_abs"] = pd.to_numeric(standardized["error_signed"], errors="coerce").abs()
    if "fold" not in standardized.columns:
        standardized["fold"] = np.nan
    if "sex" not in standardized.columns:
        standardized["sex"] = np.nan
    if "bmi" not in standardized.columns:
        standardized["bmi"] = np.nan

    standardized["sample_id"] = standardized["sample_id"].astype(str)
    standardized["subject_id"] = standardized["subject_id"].astype(str)
    for column in ["fold", "true_age", "pred_age", "error_signed", "error_abs", "bmi"]:
        standardized[column] = pd.to_numeric(standardized[column], errors="coerce")
    standardized["sex"] = standardized["sex"].astype("string")
    standardized["run"] = run_name
    standardized["run_short"] = _infer_run_short(run_name)
    standardized["run_family"] = _infer_run_family(run_name)
    return standardized[
        [
            "sample_id",
            "subject_id",
            "run",
            "run_short",
            "run_family",
            "fold",
            "true_age",
            "pred_age",
            "error_signed",
            "error_abs",
            "sex",
            "bmi",
        ]
    ]


def load_history(outputs_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(outputs_root.rglob("predictions_readable.csv")):
        if path.parent == outputs_root / "hard_error_history":
            continue
        run_name = path.parent.name
        frame = pd.read_csv(path)
        frames.append(standardize_predictions(frame, run_name))
    if not frames:
        raise FileNotFoundError(f"No predictions_readable.csv files found under {outputs_root}")
    history = pd.concat(frames, ignore_index=True)
    history = history.dropna(subset=["sample_id", "subject_id", "true_age", "pred_age", "error_signed", "error_abs"])
    return history


def add_high_error_flags(history: pd.DataFrame) -> pd.DataFrame:
    run_thresholds = (
        history.groupby("run", as_index=False)["error_abs"]
        .quantile(0.9)
        .rename(columns={"error_abs": "run_top10pct_threshold"})
    )
    flagged = history.merge(run_thresholds, on="run", how="left")
    flagged["is_top10pct"] = flagged["error_abs"] >= flagged["run_top10pct_threshold"]
    flagged["is_ge20"] = flagged["error_abs"] >= 20.0
    return flagged


def _add_direction_columns(summary: pd.DataFrame, positive_col: str, negative_col: str, mean_signed_col: str) -> pd.DataFrame:
    total_nonzero = summary[positive_col] + summary[negative_col]
    summary["sign_consistency"] = np.where(
        total_nonzero > 0,
        np.maximum(summary[positive_col], summary[negative_col]) / total_nonzero,
        0.0,
    )
    mean_sign = np.sign(summary[mean_signed_col].fillna(0.0))
    summary["direction"] = [
        _direction_from_sign(sign, consistency)
        for sign, consistency in zip(mean_sign, summary["sign_consistency"], strict=False)
    ]
    return summary


def build_sample_summary(history: pd.DataFrame) -> pd.DataFrame:
    sample_run = (
        history.groupby(["sample_id", "subject_id", "run"], as_index=False)
        .agg(
            run_short=("run_short", "first"),
            run_family=("run_family", "first"),
            fold=("fold", "first"),
            true_age=("true_age", "median"),
            pred_age=("pred_age", "mean"),
            error_signed=("error_signed", "mean"),
            error_abs=("error_abs", "mean"),
            sex=("sex", _mode_or_nan),
            bmi=("bmi", "median"),
            is_top10pct=("is_top10pct", "max"),
            is_ge20=("is_ge20", "max"),
        )
        .sort_values(["sample_id", "run"])
        .reset_index(drop=True)
    )

    def _family_count(frame: pd.DataFrame, flag_column: str) -> int:
        return int(frame.loc[frame[flag_column], "run_family"].nunique())

    def _run_list(frame: pd.DataFrame, flag_column: str) -> str:
        values = frame.loc[frame[flag_column], "run"].tolist()
        return ",".join(values)

    sample_summary = (
        sample_run.groupby(["sample_id", "subject_id"], as_index=False)
        .agg(
            true_age=("true_age", "median"),
            sex=("sex", _mode_or_nan),
            bmi=("bmi", "median"),
            runs_present=("run", "nunique"),
            runs_top10pct=("is_top10pct", "sum"),
            runs_ge20=("is_ge20", "sum"),
        )
    )
    family_stats = (
        sample_run.groupby(["sample_id", "subject_id"])
        .apply(
            lambda frame: pd.Series(
                {
                    "families_top10pct": _family_count(frame, "is_top10pct"),
                    "families_ge20": _family_count(frame, "is_ge20"),
                    "runs_top10pct_list": _run_list(frame, "is_top10pct"),
                    "runs_ge20_list": _run_list(frame, "is_ge20"),
                    "positive_error_runs": int((frame["error_signed"] > 0).sum()),
                    "negative_error_runs": int((frame["error_signed"] < 0).sum()),
                    "mean_pred_age": float(frame["pred_age"].mean()),
                    "pred_age_min": float(frame["pred_age"].min()),
                    "pred_age_max": float(frame["pred_age"].max()),
                    "mean_error_abs": float(frame["error_abs"].mean()),
                    "median_error_abs": float(frame["error_abs"].median()),
                    "max_error_abs": float(frame["error_abs"].max()),
                    "min_error_abs": float(frame["error_abs"].min()),
                    "mean_error_signed": float(frame["error_signed"].mean()),
                    "median_error_signed": float(frame["error_signed"].median()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    sample_summary = sample_summary.merge(family_stats, on=["sample_id", "subject_id"], how="left")
    sample_summary["pct_runs_top10pct"] = sample_summary.apply(
        lambda row: _format_pct(row["runs_top10pct"], row["runs_present"]),
        axis=1,
    )
    sample_summary["pct_runs_ge20"] = sample_summary.apply(
        lambda row: _format_pct(row["runs_ge20"], row["runs_present"]),
        axis=1,
    )
    sample_summary["all_runs_top10pct"] = sample_summary["runs_top10pct"] == sample_summary["runs_present"]
    sample_summary["all_runs_ge20"] = sample_summary["runs_ge20"] == sample_summary["runs_present"]
    sample_summary = _add_direction_columns(sample_summary, "positive_error_runs", "negative_error_runs", "mean_error_signed")
    sample_summary = sample_summary.sort_values(
        ["runs_ge20", "runs_top10pct", "mean_error_abs", "max_error_abs"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    sample_summary.insert(0, "rank", np.arange(1, len(sample_summary) + 1))
    return sample_summary


def build_subject_summary(history: pd.DataFrame, sample_summary: pd.DataFrame) -> pd.DataFrame:
    subject_run = (
        history.groupby(["run", "subject_id"], as_index=False)
        .agg(
            run_short=("run_short", "first"),
            run_family=("run_family", "first"),
            true_age=("true_age", "median"),
            mean_pred_age=("pred_age", "mean"),
            sex=("sex", _mode_or_nan),
            bmi=("bmi", "median"),
            n_samples=("sample_id", "nunique"),
            mean_error_abs=("error_abs", "mean"),
            median_error_abs=("error_abs", "median"),
            max_error_abs=("error_abs", "max"),
            mean_error_signed=("error_signed", "mean"),
            n_sample_top10pct=("is_top10pct", "sum"),
            n_sample_ge20=("is_ge20", "sum"),
        )
        .sort_values(["subject_id", "run"])
        .reset_index(drop=True)
    )
    subject_thresholds = (
        subject_run.groupby("run", as_index=False)["mean_error_abs"]
        .quantile(0.9)
        .rename(columns={"mean_error_abs": "subject_top10pct_threshold"})
    )
    subject_run = subject_run.merge(subject_thresholds, on="run", how="left")
    subject_run["is_subject_top10pct"] = subject_run["mean_error_abs"] >= subject_run["subject_top10pct_threshold"]
    subject_run["any_sample_ge20"] = subject_run["max_error_abs"] >= 20.0

    subject_sample_counts = (
        sample_summary.groupby("subject_id", as_index=False)
        .agg(
            n_unique_samples=("sample_id", "nunique"),
            n_samples_all_runs_top10pct=("all_runs_top10pct", "sum"),
            n_samples_all_runs_ge20=("all_runs_ge20", "sum"),
            top_sample_ids=(
                "sample_id",
                lambda s: ",".join(
                    sample_summary.loc[s.index]
                    .sort_values(["runs_ge20", "runs_top10pct", "mean_error_abs"], ascending=[False, False, False])
                    .head(5)["sample_id"]
                    .tolist()
                ),
            ),
        )
    )

    def _subject_lists(frame: pd.DataFrame, flag_column: str) -> tuple[int, str]:
        flagged = frame.loc[frame[flag_column]]
        return int(flagged["run_family"].nunique()), ",".join(flagged["run"].tolist())

    subject_summary = (
        subject_run.groupby("subject_id", as_index=False)
        .agg(
            true_age=("true_age", "median"),
            mean_pred_age=("mean_pred_age", "mean"),
            sex=("sex", _mode_or_nan),
            bmi=("bmi", "median"),
            runs_present=("run", "nunique"),
            runs_subject_top10pct=("is_subject_top10pct", "sum"),
            runs_any_sample_ge20=("any_sample_ge20", "sum"),
            mean_subject_error_abs=("mean_error_abs", "mean"),
            median_subject_error_abs=("mean_error_abs", "median"),
            max_subject_error_abs=("max_error_abs", "max"),
            mean_subject_error_signed=("mean_error_signed", "mean"),
            positive_error_runs=("mean_error_signed", lambda s: int((s > 0).sum())),
            negative_error_runs=("mean_error_signed", lambda s: int((s < 0).sum())),
        )
    )

    subject_family_stats = (
        subject_run.groupby("subject_id")
        .apply(
            lambda frame: pd.Series(
                {
                    "families_subject_top10pct": _subject_lists(frame, "is_subject_top10pct")[0],
                    "families_any_sample_ge20": _subject_lists(frame, "any_sample_ge20")[0],
                    "runs_subject_top10pct_list": _subject_lists(frame, "is_subject_top10pct")[1],
                    "runs_any_sample_ge20_list": _subject_lists(frame, "any_sample_ge20")[1],
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    subject_summary = subject_summary.merge(subject_family_stats, on="subject_id", how="left")
    subject_summary = subject_summary.merge(subject_sample_counts, on="subject_id", how="left")
    subject_summary["pct_runs_subject_top10pct"] = subject_summary.apply(
        lambda row: _format_pct(row["runs_subject_top10pct"], row["runs_present"]),
        axis=1,
    )
    subject_summary["pct_runs_any_sample_ge20"] = subject_summary.apply(
        lambda row: _format_pct(row["runs_any_sample_ge20"], row["runs_present"]),
        axis=1,
    )
    subject_summary["all_runs_subject_top10pct"] = (
        subject_summary["runs_subject_top10pct"] == subject_summary["runs_present"]
    )
    subject_summary["all_runs_any_sample_ge20"] = (
        subject_summary["runs_any_sample_ge20"] == subject_summary["runs_present"]
    )
    subject_summary = _add_direction_columns(
        subject_summary,
        "positive_error_runs",
        "negative_error_runs",
        "mean_subject_error_signed",
    )
    subject_summary = subject_summary.sort_values(
        ["runs_any_sample_ge20", "runs_subject_top10pct", "mean_subject_error_abs", "max_subject_error_abs"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    subject_summary.insert(0, "rank", np.arange(1, len(subject_summary) + 1))
    return subject_summary


def _age_bin_summary(history: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    bins = [0.0, 35.0, 60.0, float("inf")]
    labels = ["young(<35)", "mid(35-60)", "old(>=60)"]
    subset = history.loc[mask].copy()
    subset["age_bin"] = pd.cut(subset["true_age"], bins=bins, labels=labels, right=False)
    summary = (
        subset.groupby("age_bin", observed=False)
        .agg(
            n=("sample_id", "size"),
            mean_error_signed=("error_signed", "mean"),
            mean_error_abs=("error_abs", "mean"),
        )
        .reset_index()
    )
    return summary


def _sex_rate_summary(history: pd.DataFrame, flag_column: str) -> pd.DataFrame:
    frame = history.copy()
    summary = (
        frame.groupby("sex", dropna=False)
        .agg(total=("sample_id", "size"), flagged=(flag_column, "sum"))
        .reset_index()
    )
    summary["flag_rate"] = summary["flagged"] / summary["total"]
    return summary


def build_summary_markdown(history: pd.DataFrame, sample_summary: pd.DataFrame, subject_summary: pd.DataFrame) -> str:
    n_runs = int(history["run"].nunique())
    n_samples = int(history["sample_id"].nunique())
    n_subjects = int(history["subject_id"].nunique())
    total_rows = int(len(history))

    unique_samples_top10 = int((sample_summary["runs_top10pct"] > 0).sum())
    unique_samples_ge20 = int((sample_summary["runs_ge20"] > 0).sum())
    unique_subjects_top10 = int((subject_summary["runs_subject_top10pct"] > 0).sum())
    unique_subjects_ge20 = int((subject_summary["runs_any_sample_ge20"] > 0).sum())

    samples_top10_all_runs = int(sample_summary["all_runs_top10pct"].sum())
    samples_ge20_all_runs = int(sample_summary["all_runs_ge20"].sum())
    subjects_top10_all_runs = int(subject_summary["all_runs_subject_top10pct"].sum())
    subjects_ge20_all_runs = int(subject_summary["all_runs_any_sample_ge20"].sum())

    samples_top10_ge10 = int((sample_summary["runs_top10pct"] >= 10).sum())
    samples_ge20_ge10 = int((sample_summary["runs_ge20"] >= 10).sum())
    subjects_top10_ge10 = int((subject_summary["runs_subject_top10pct"] >= 10).sum())
    subjects_ge20_ge10 = int((subject_summary["runs_any_sample_ge20"] >= 10).sum())

    slope_all = float(np.polyfit(history["true_age"], history["error_signed"], deg=1)[0])
    top10_history = history.loc[history["is_top10pct"]]
    ge20_history = history.loc[history["is_ge20"]]
    slope_top10 = float(np.polyfit(top10_history["true_age"], top10_history["error_signed"], deg=1)[0])
    slope_ge20 = float(np.polyfit(ge20_history["true_age"], ge20_history["error_signed"], deg=1)[0])

    age_all = _age_bin_summary(history, history["error_abs"] >= 0)
    age_top10 = _age_bin_summary(history, history["is_top10pct"])
    age_ge20 = _age_bin_summary(history, history["is_ge20"])

    sex_top10 = _sex_rate_summary(history, "is_top10pct")
    sex_ge20 = _sex_rate_summary(history, "is_ge20")

    bmi_corr_abs = float(history[["bmi", "error_abs"]].corr().iloc[0, 1])
    bmi_corr_signed = float(history[["bmi", "error_signed"]].corr().iloc[0, 1])

    top_subjects = subject_summary.loc[
        :,
        [
            "subject_id",
            "true_age",
            "sex",
            "runs_any_sample_ge20",
            "runs_subject_top10pct",
            "mean_subject_error_abs",
            "mean_subject_error_signed",
            "direction",
            "top_sample_ids",
        ],
    ].head(10)
    top_samples = sample_summary.loc[
        :,
        [
            "sample_id",
            "subject_id",
            "true_age",
            "runs_ge20",
            "runs_top10pct",
            "mean_error_abs",
            "mean_error_signed",
            "direction",
        ],
    ].head(10)

    lines = [
        "# 历次 Run 高误差样本历史总结",
        "",
        "## 范围",
        "",
        f"- 共扫描 `{n_runs}` 个历史 run、`{total_rows}` 条预测记录（全部来自 `predictions_readable.csv`）。",
        f"- 覆盖 `{n_samples}` 个唯一 sample、`{n_subjects}` 个唯一 subject。",
        "- 并行使用两种高误差定义：",
        "- 每个 run 内 `error_abs` 前 10%",
        "- `error_abs >= 20`",
        "",
        "## 结论",
        "",
        f"- 高误差明显集中，而不是均匀散落。全部 `{n_samples}` 个 sample 里，只有 `{unique_samples_top10}` 个 sample 曾进入任一 run 的 top 10%，只有 `{unique_samples_ge20}` 个 sample 曾达到 `error_abs >= 20`。",
        f"- 跨 run 复发性很强。共有 `{samples_top10_all_runs}` 个 sample 在全部 `{n_runs}` 个 run 中都属于 top 10%，`{samples_ge20_all_runs}` 个 sample 在全部 `{n_runs}` 个 run 中都达到 `error_abs >= 20`。subject 层面则有 `{subjects_top10_all_runs}` 个 subject 在全部 run 中都属于 subject-top-10%，`{subjects_ge20_all_runs}` 个 subject 在全部 run 中都至少出现一个 `>=20` 的 sample。",
        f"- 这不是少数偶发错误。共有 `{samples_top10_ge10}` 个 sample 和 `{samples_ge20_ge10}` 个 sample 在至少 10 个 run 中持续高误差；subject 层面也有 `{subjects_top10_ge10}` 个和 `{subjects_ge20_ge10}` 个 subject 达到同样水平。",
        f"- 最明显的系统性模式是“回归到均值”。残差对年龄的斜率整体为 `{_format_float(slope_all)}`，在高误差子集里更明显：top 10% 子集为 `{_format_float(slope_top10)}`，`error_abs >= 20` 子集为 `{_format_float(slope_ge20)}`。",
        f"- 年轻样本常被预测偏老，老年样本常被预测偏年轻。在 top 10% 子集里，年轻组的平均 signed error 为 `{_format_float(age_top10.loc[age_top10['age_bin'] == 'young(<35)', 'mean_error_signed'].iloc[0])}`，老年组为 `{_format_float(age_top10.loc[age_top10['age_bin'] == 'old(>=60)', 'mean_error_signed'].iloc[0])}`。",
        f"- hardest cases 在不同 run family 之间也很稳定，而不是只在某一类模型里出现。共有 `{int((sample_summary['families_top10pct'] == 3).sum())}` 个 sample 在 `whole_image / roi_only / fusion_whole_roi` 三个 family 中都进入过 top 10%，`{int((sample_summary['families_ge20'] == 3).sum())}` 个 sample 在三个 family 中都出现过 `>=20`。",
        f"- sex 只表现出轻度不对称，弱于年龄效应。top 10% 比例为 F `{_format_float(sex_top10.loc[sex_top10['sex'] == 'F', 'flag_rate'].iloc[0] * 100.0, 1)}%`、M `{_format_float(sex_top10.loc[sex_top10['sex'] == 'M', 'flag_rate'].iloc[0] * 100.0, 1)}%`；`>=20` 比例为 F `{_format_float(sex_ge20.loc[sex_ge20['sex'] == 'F', 'flag_rate'].iloc[0] * 100.0, 1)}%`、M `{_format_float(sex_ge20.loc[sex_ge20['sex'] == 'M', 'flag_rate'].iloc[0] * 100.0, 1)}%`。",
        f"- BMI 没看到明显线性信号，`corr(BMI, error_abs) = {_format_float(bmi_corr_abs)}`，`corr(BMI, error_signed) = {_format_float(bmi_corr_signed)}`。",
        "",
        "## 最顽固的 Hard Subjects",
        "",
        _markdown_table(top_subjects),
        "",
        "## 最顽固的 Hard Samples",
        "",
        _markdown_table(top_samples),
        "",
        "## 年龄模式",
        "",
        "### 全部预测",
        "",
        _markdown_table(age_all),
        "",
        "### 每个 run 内 top 10%",
        "",
        _markdown_table(age_top10),
        "",
        "### `error_abs >= 20`",
        "",
        _markdown_table(age_ge20),
        "",
        "## 解释",
        "",
        "- 最顽固的失败不是偶发事件。对顶级 hard subject 来说，同一个 subject 的多张图像在几乎所有 run 中都朝同一个方向出错。",
        "- 因此问题至少有两层：一层是模型层面的系统偏差，另一层是 subject 层面的异常。强烈的年龄残差斜率已经清楚说明模型存在“回归到均值”的系统性偏差。",
        "- 但最难的 subject 同时也更像 subject 级异常，而不是单张图像质量问题。比如 `1068`、`1396`、`1473`、`451` 这几个 subject，三张图在全部 15 个 run 中几乎都持续大错，而且符号一致。",
        "- 这种模式很难用“某一张图刚好质量差”来解释，更像是 subject 级别的问题，例如 label / pairing 错配、采集域偏移，或者该 subject 的外观确实偏离了当前特征工程能表达的分布。",
        "- 也存在少量中间年龄段的持续异常 subject，例如约 37 岁的 `1006` 被反复预测偏老，所以问题不只发生在年龄两端。但整体上，年龄仍然是最强的系统偏差来源。",
        "",
        "## 输出文件",
        "",
        "- `hard_samples_summary.csv`：sample-level 跨 run 复发汇总。",
        "- `hard_subjects_summary.csv`：subject-level 跨 run 复发汇总。",
    ]
    return "\n".join(lines).strip() + "\n"


def write_outputs(
    analysis_dir: Path,
    summary_markdown: str,
    sample_summary: pd.DataFrame,
    subject_summary: pd.DataFrame,
) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "hard_error_history_summary.md").write_text(summary_markdown, encoding="utf-8")
    sample_summary.to_csv(analysis_dir / "hard_samples_summary.csv", index=False)
    subject_summary.to_csv(analysis_dir / "hard_subjects_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_root).resolve()
    analysis_dir = Path(args.analysis_dir).resolve()

    history = add_high_error_flags(load_history(outputs_root))
    sample_summary = build_sample_summary(history)
    subject_summary = build_subject_summary(history, sample_summary)
    summary_markdown = build_summary_markdown(history, sample_summary, subject_summary)
    write_outputs(analysis_dir, summary_markdown, sample_summary, subject_summary)

    print(f"Wrote summary to {analysis_dir / 'hard_error_history_summary.md'}")
    print(f"Wrote sample summary to {analysis_dir / 'hard_samples_summary.csv'}")
    print(f"Wrote subject summary to {analysis_dir / 'hard_subjects_summary.csv'}")


if __name__ == "__main__":
    main()
