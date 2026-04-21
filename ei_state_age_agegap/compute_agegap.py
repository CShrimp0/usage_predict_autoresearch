#!/usr/bin/env python3
"""Define EI-based state age and compute prediction-vs-state-age gap MAE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


AGE_CANDIDATES = ("age", "Age", "true_age", "y_true", "年龄")
PRED_CANDIDATES = ("prediction", "pred_age", "predicted_age", "y_pred", "预测年龄")
EI_CANDIDATES = (
    "EI",
    "ei",
    "echo_intensity",
    "echo intensity",
    "EchoIntensity",
    "mean_echo_intensity",
    "灰度",
    "亮度",
)
ID_CANDIDATES = ("sample_id", "subject_id", "ID", "id", "Local ID", "ID_norm_file4", "ID1_file4")


def normalize_name(value: str) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def read_table(path: Path, sheet: str | int | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=0 if sheet is None else sheet)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def find_column(
    columns: Iterable[str],
    explicit: str | None,
    candidates: Iterable[str],
    label: str,
    *,
    optional: bool = False,
    allow_contains: bool = False,
) -> str | None:
    column_list = list(columns)
    if explicit:
        if explicit in column_list:
            return explicit
        raise KeyError(f"{label} column {explicit!r} not found. Available columns: {column_list}")

    normalized_to_column = {normalize_name(column): column for column in column_list}
    for candidate in candidates:
        if candidate in column_list:
            return candidate
        normalized = normalize_name(candidate)
        if normalized in normalized_to_column:
            return normalized_to_column[normalized]

    if allow_contains:
        matches: list[str] = []
        normalized_candidates = [normalize_name(candidate) for candidate in candidates]
        for column in column_list:
            normalized_column = normalize_name(column)
            if any(candidate and candidate in normalized_column for candidate in normalized_candidates):
                matches.append(column)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise KeyError(
                f"Multiple possible {label} columns found: {matches}. "
                f"Pass --{label.replace(' ', '-').lower()}-column explicitly."
            )

    if optional:
        return None
    raise KeyError(f"Could not infer {label} column. Available columns: {column_list}")


def parse_merge_keys(values: list[str] | None, left_columns: Iterable[str], right_columns: Iterable[str]) -> tuple[list[str], list[str]]:
    if values:
        left_keys: list[str] = []
        right_keys: list[str] = []
        for value in values:
            if ":" in value:
                left_key, right_key = value.split(":", 1)
            else:
                left_key = value
                right_key = value
            left_keys.append(left_key)
            right_keys.append(right_key)
        return left_keys, right_keys

    left_set = set(left_columns)
    right_set = set(right_columns)
    common = [column for column in ID_CANDIDATES if column in left_set and column in right_set]
    if not common:
        raise KeyError(
            "Could not infer merge keys between prediction table and EI source. "
            "Pass --merge-key left_col:right_col."
        )
    return [common[0]], [common[0]]


def merge_ei_source(
    predictions: pd.DataFrame,
    source: pd.DataFrame,
    ei_column: str,
    merge_keys: list[str] | None,
) -> tuple[pd.DataFrame, str]:
    left_keys, right_keys = parse_merge_keys(merge_keys, predictions.columns, source.columns)
    missing_left = [key for key in left_keys if key not in predictions.columns]
    missing_right = [key for key in right_keys if key not in source.columns]
    if missing_left:
        raise KeyError(f"Merge key(s) missing in prediction table: {missing_left}")
    if missing_right:
        raise KeyError(f"Merge key(s) missing in EI source: {missing_right}")

    merged_ei_column = ei_column if ei_column not in predictions.columns else f"{ei_column}_ei_source"
    source_subset = source[right_keys + [ei_column]].copy()
    source_subset[ei_column] = pd.to_numeric(source_subset[ei_column], errors="coerce")
    source_subset = source_subset.dropna(subset=[ei_column])
    source_subset = source_subset.groupby(right_keys, as_index=False)[ei_column].mean()
    source_subset = source_subset.rename(columns={ei_column: merged_ei_column})

    merged = predictions.merge(
        source_subset,
        left_on=left_keys,
        right_on=right_keys,
        how="left",
        validate="many_to_one",
        suffixes=("", "_ei_source"),
    )
    extra_right_keys = [key for key in right_keys if key not in left_keys and key in merged.columns]
    if extra_right_keys:
        merged = merged.drop(columns=extra_right_keys)
    return merged, merged_ei_column


def build_state_age(
    ei_values: pd.Series,
    min_age: float,
    max_age: float,
    *,
    method: str,
    reverse: bool,
) -> pd.Series:
    if method == "rank":
        scaled = ei_values.rank(method="average", pct=True)
        if len(scaled) > 1:
            scaled = (scaled - scaled.min()) / (scaled.max() - scaled.min())
        else:
            scaled = pd.Series(0.5, index=ei_values.index)
    else:
        ei_min = float(ei_values.min())
        ei_max = float(ei_values.max())
        if np.isclose(ei_min, ei_max):
            scaled = pd.Series(0.5, index=ei_values.index)
        else:
            scaled = (ei_values - ei_min) / (ei_max - ei_min)

    if reverse:
        scaled = 1.0 - scaled
    return min_age + scaled * (max_age - min_age)


def summarize_gap(frame: pd.DataFrame) -> dict[str, float | int]:
    gap = frame["agegap"].to_numpy(dtype=float)
    abs_gap = np.abs(gap)
    return {
        "n": int(len(frame)),
        "agegap_mae": float(abs_gap.mean()) if len(abs_gap) else np.nan,
        "agegap_rmse": float(np.sqrt(np.mean(np.square(gap)))) if len(gap) else np.nan,
        "agegap_bias": float(gap.mean()) if len(gap) else np.nan,
        "agegap_median_abs": float(np.median(abs_gap)) if len(abs_gap) else np.nan,
        "agegap_std": float(np.std(gap, ddof=1)) if len(gap) > 1 else 0.0,
    }


def write_markdown_summary(metrics: dict, output_path: Path) -> None:
    lines = [
        "# EI State Age Agegap Summary",
        "",
        f"- Input rows used: `{metrics['row_level']['n']}`",
        f"- Row-level agegap MAE: `{metrics['row_level']['agegap_mae']:.6g}`",
        f"- Row-level agegap bias: `{metrics['row_level']['agegap_bias']:.6g}`",
        f"- Agegap definition: `pred_age - state_age`",
        f"- State-age range: `{metrics['config']['min_age']}` to `{metrics['config']['max_age']}`",
        f"- EI mapping method: `{metrics['config']['method']}`",
        f"- Reverse EI direction: `{metrics['config']['reverse']}`",
    ]
    subject_level = metrics.get("subject_level")
    if subject_level:
        lines.extend(
            [
                "",
                f"- Subject-level n: `{subject_level['n']}`",
                f"- Subject-level agegap MAE: `{subject_level['agegap_mae']:.6g}`",
                f"- Subject-level agegap bias: `{subject_level['agegap_bias']:.6g}`",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Map EI to a state-age scale for 18-100 year-old rows, compute "
            "agegap = pred_age - state_age, and report MAE."
        )
    )
    parser.add_argument("--input", required=True, help="Prediction table containing pred age and optionally EI.")
    parser.add_argument("--sheet", default=None, help="Excel sheet for --input if it is xlsx/xls.")
    parser.add_argument("--ei-source", default=None, help="Optional separate table containing EI.")
    parser.add_argument("--ei-sheet", default=None, help="Excel sheet for --ei-source if it is xlsx/xls.")
    parser.add_argument("--ei-column", default=None, help="EI column name. Required if EI cannot be inferred.")
    parser.add_argument("--age-column", default=None, help="Chronological age column used only for 18-100 filtering.")
    parser.add_argument("--pred-column", default=None, help="Predicted age column.")
    parser.add_argument(
        "--merge-key",
        action="append",
        default=None,
        help="Join key for --ei-source. Use key for same name or left_col:right_col. Repeat for composite keys.",
    )
    parser.add_argument("--output-dir", default="outputs/ei_state_age_agegap", help="Directory for result files.")
    parser.add_argument("--min-age", type=float, default=18.0, help="Minimum chronological age to include.")
    parser.add_argument("--max-age", type=float, default=100.0, help="Maximum chronological age to include.")
    parser.add_argument(
        "--method",
        choices=("linear", "rank"),
        default="linear",
        help="How to scale EI to state_age. linear uses EI min/max; rank uses EI percentile rank.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Use when higher EI should indicate younger state age instead of older state age.",
    )
    parser.add_argument(
        "--subject-column",
        default=None,
        help="Optional subject/person id column for an additional person-level MAE table. Auto-detects subject_id.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = read_table(input_path, args.sheet)
    source_path = Path(args.ei_source) if args.ei_source else None
    if source_path:
        source = read_table(source_path, args.ei_sheet)
        source_ei_column = find_column(
            source.columns,
            args.ei_column,
            EI_CANDIDATES,
            "EI",
            allow_contains=True,
        )
        frame, ei_column = merge_ei_source(frame, source, source_ei_column, args.merge_key)
    else:
        ei_column = find_column(frame.columns, args.ei_column, EI_CANDIDATES, "EI", allow_contains=True)

    age_column = find_column(frame.columns, args.age_column, AGE_CANDIDATES, "age")
    pred_column = find_column(frame.columns, args.pred_column, PRED_CANDIDATES, "pred")
    subject_column = find_column(
        frame.columns,
        args.subject_column,
        ("subject_id", "ID", "id", "Local ID"),
        "subject",
        optional=True,
    )

    working = frame.copy()
    working[age_column] = pd.to_numeric(working[age_column], errors="coerce")
    working[pred_column] = pd.to_numeric(working[pred_column], errors="coerce")
    working[ei_column] = pd.to_numeric(working[ei_column], errors="coerce")
    n_before = len(working)
    working = working.dropna(subset=[age_column, pred_column, ei_column]).copy()
    n_after_numeric = len(working)
    working = working[(working[age_column] >= args.min_age) & (working[age_column] <= args.max_age)].copy()
    if working.empty:
        raise ValueError("No rows left after numeric cleaning and age filtering.")

    working["state_age"] = build_state_age(
        working[ei_column],
        args.min_age,
        args.max_age,
        method=args.method,
        reverse=args.reverse,
    )
    working["agegap"] = working[pred_column] - working["state_age"]
    working["agegap_abs"] = working["agegap"].abs()

    metrics: dict[str, object] = {
        "config": {
            "input": str(input_path),
            "ei_source": str(source_path) if source_path else None,
            "age_column": age_column,
            "pred_column": pred_column,
            "ei_column": ei_column,
            "subject_column": subject_column,
            "min_age": args.min_age,
            "max_age": args.max_age,
            "method": args.method,
            "reverse": args.reverse,
        },
        "data": {
            "rows_input": int(n_before),
            "rows_after_numeric_cleaning": int(n_after_numeric),
            "rows_after_age_filter": int(len(working)),
            "ei_min": float(working[ei_column].min()),
            "ei_max": float(working[ei_column].max()),
            "state_age_min": float(working["state_age"].min()),
            "state_age_max": float(working["state_age"].max()),
        },
        "row_level": summarize_gap(working),
    }

    working.to_csv(output_dir / "agegap_results.csv", index=False)

    if subject_column:
        numeric_columns = [age_column, pred_column, ei_column, "state_age"]
        subject_frame = working.groupby(subject_column, as_index=False)[numeric_columns].mean()
        subject_frame["agegap"] = subject_frame[pred_column] - subject_frame["state_age"]
        subject_frame["agegap_abs"] = subject_frame["agegap"].abs()
        subject_frame.to_csv(output_dir / "agegap_subject_results.csv", index=False)
        metrics["subject_level"] = summarize_gap(subject_frame)

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown_summary(metrics, output_dir / "summary.md")
    print(json.dumps(metrics["row_level"], ensure_ascii=False, indent=2))
    if "subject_level" in metrics:
        print(json.dumps({"subject_level": metrics["subject_level"]}, ensure_ascii=False, indent=2))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
