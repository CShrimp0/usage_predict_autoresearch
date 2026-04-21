#!/usr/bin/env python3
"""Build a compact manual-audit shortlist from hard subject/sample summaries."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "outputs" / "hard_error_history"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", default=str(DEFAULT_ANALYSIS_DIR))
    return parser.parse_args()


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "NaN"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return "NaN"
    return f"{numeric:.{digits}f}"


def _pick_top_subjects(subjects: pd.DataFrame) -> pd.DataFrame:
    filtered = subjects.copy()
    filtered = filtered[
        (filtered["all_runs_any_sample_ge20"] == True)
        & (filtered["sign_consistency"] >= 0.95)
    ].copy()
    filtered = filtered.sort_values(
        [
            "all_runs_subject_top10pct",
            "runs_any_sample_ge20",
            "runs_subject_top10pct",
            "mean_subject_error_abs",
            "max_subject_error_abs",
        ],
        ascending=[False, False, False, False, False],
    ).head(10)
    filtered = filtered.reset_index(drop=True)
    filtered.insert(0, "audit_rank", range(1, len(filtered) + 1))
    return filtered


def _audit_reason(row: pd.Series) -> str:
    age = float(row["true_age"])
    direction = str(row["direction"])
    n_samples = int(row["n_unique_samples"])
    if age < 35 and "older" in direction:
        return f"年轻 subject 在全部 run 中稳定被预测偏老，且 {n_samples} 张图一起持续大错，优先排查标签/配对与采集差异。"
    if age >= 60 and "younger" in direction:
        return f"老年 subject 在全部 run 中稳定被预测偏年轻，且 {n_samples} 张图一起持续大错，优先排查标签/配对与年龄相关系统偏差。"
    if 35 <= age < 60:
        return f"非年龄极端但跨全部 run 持续大错，更像 subject 级异常或数据问题，而不只是常规年龄回归偏差。"
    return f"跨全部 run 与多个 run family 持续同向大错，且 {n_samples} 张图共同失败，优先人工核查原图与元数据。"


def build_shortlist_csv(subjects: pd.DataFrame) -> pd.DataFrame:
    shortlist = subjects.copy()
    shortlist["why_audit_first"] = shortlist.apply(_audit_reason, axis=1)
    return shortlist[
        [
            "audit_rank",
            "subject_id",
            "true_age",
            "sex",
            "bmi",
            "top_sample_ids",
            "mean_pred_age",
            "mean_subject_error_abs",
            "direction",
            "why_audit_first",
        ]
    ]


def build_sample_shortlist(samples: pd.DataFrame, shortlisted_subjects: pd.DataFrame) -> pd.DataFrame:
    subject_order = {subject_id: idx for idx, subject_id in enumerate(shortlisted_subjects["subject_id"].tolist(), start=1)}
    picked = samples[samples["subject_id"].isin(shortlisted_subjects["subject_id"])].copy()
    picked["subject_rank"] = picked["subject_id"].map(subject_order)
    picked = picked.sort_values(
        ["subject_rank", "runs_ge20", "runs_top10pct", "mean_error_abs", "max_error_abs"],
        ascending=[True, False, False, False, False],
    )
    picked = picked.groupby("subject_id", as_index=False, group_keys=False).head(3).reset_index(drop=True)
    return picked[
        [
            "subject_rank",
            "subject_id",
            "sample_id",
            "true_age",
            "mean_pred_age",
            "mean_error_abs",
            "direction",
        ]
    ]


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_无数据_"
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                if math.isnan(value):
                    values.append("NaN")
                else:
                    values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_markdown(shortlist_csv: pd.DataFrame, sample_shortlist: pd.DataFrame) -> str:
    subject_table = shortlist_csv.rename(
        columns={
            "audit_rank": "rank",
            "subject_id": "subject_id",
            "true_age": "true_age",
            "sex": "sex",
            "bmi": "bmi",
            "top_sample_ids": "top_sample_ids",
            "mean_pred_age": "mean_pred_age",
            "mean_subject_error_abs": "mean_subject_error_abs",
            "direction": "direction",
            "why_audit_first": "why_audit_first",
        }
    )
    sample_table = sample_shortlist.rename(
        columns={
            "subject_rank": "subject_rank",
            "subject_id": "subject_id",
            "sample_id": "sample_id",
            "true_age": "true_age",
            "mean_pred_age": "mean_pred_age",
            "mean_error_abs": "mean_error_abs",
            "direction": "direction",
        }
    )
    lines = [
        "# Hard Subject Audit Shortlist",
        "",
        "## Subject Shortlist",
        "",
        _markdown_table(subject_table),
        "",
        "## Corresponding Sample Shortlist",
        "",
        _markdown_table(sample_table),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir).resolve()
    subjects = pd.read_csv(analysis_dir / "hard_subjects_summary.csv")
    samples = pd.read_csv(analysis_dir / "hard_samples_summary.csv")

    shortlisted_subjects = _pick_top_subjects(subjects)
    shortlist_csv = build_shortlist_csv(shortlisted_subjects)
    sample_shortlist = build_sample_shortlist(samples, shortlisted_subjects)

    shortlist_csv.to_csv(analysis_dir / "hard_subject_audit_shortlist.csv", index=False)
    (analysis_dir / "hard_subject_audit_shortlist.md").write_text(
        build_markdown(shortlist_csv, sample_shortlist),
        encoding="utf-8",
    )

    print(f"Wrote {analysis_dir / 'hard_subject_audit_shortlist.csv'}")
    print(f"Wrote {analysis_dir / 'hard_subject_audit_shortlist.md'}")


if __name__ == "__main__":
    main()
