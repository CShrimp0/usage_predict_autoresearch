#!/usr/bin/env python3
"""Benchmark lightweight feature sets for state-age fitting under unified group CV."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import pearsonr
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataio.load_images import load_grayscale_image, load_mask
from preprocessing.split import RegressionStratifiedGroupKFold

try:
    from skimage.feature import local_binary_pattern

    HAS_LBP = True
except Exception:
    HAS_LBP = False


PRED_COLUMN_CANDIDATES = ["pred_age", "prediction", "predicted_age", "y_pred"]
TRUE_AGE_CANDIDATES = ["true_age", "age", "y_true"]
SUBJECT_CANDIDATES = ["subject_id", "ID", "id"]
SAMPLE_CANDIDATES = ["sample_id", "image_id", "instance_id"]


@dataclass
class FeatureRunResult:
    feature_set: str
    model: str
    n_features: int
    sample_gap_mae: float
    subject_gap_mae: float
    sample_gap_rmse: float
    subject_gap_rmse: float
    state_age_vs_true_mae: float
    state_age_vs_true_corr: float
    state_age_std: float
    state_age_min: float
    state_age_max: float
    status: str
    note: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def resolve_column(columns: list[str], candidates: list[str], explicit: str | None, label: str) -> str:
    if explicit:
        if explicit in columns:
            return explicit
        raise KeyError(f"{label} column {explicit!r} is missing. Available: {columns}")
    normalized = {normalize_name(c): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        n_cand = normalize_name(cand)
        if n_cand in normalized:
            return normalized[n_cand]
    raise KeyError(f"Could not infer {label} column. Available: {columns}")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(pearsonr(x, y)[0])


def infer_mask_path(row: pd.Series) -> Path | None:
    for key in ("roi_path", "mask_path"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            candidate = Path(value)
            if candidate.exists():
                return candidate
    image_path = row.get("image_path")
    if not isinstance(image_path, str) or not image_path:
        return None
    candidate = Path(image_path.replace("/Images/", "/Masks/"))
    return candidate if candidate.exists() else None


def extract_basic_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}mean": np.nan,
            f"{prefix}median": np.nan,
            f"{prefix}std": np.nan,
            f"{prefix}iqr": np.nan,
            f"{prefix}min": np.nan,
            f"{prefix}max": np.nan,
            f"{prefix}p10": np.nan,
            f"{prefix}p90": np.nan,
        }
    p10 = float(np.percentile(values, 10))
    p90 = float(np.percentile(values, 90))
    p25 = float(np.percentile(values, 25))
    p75 = float(np.percentile(values, 75))
    return {
        f"{prefix}mean": float(np.mean(values)),
        f"{prefix}median": float(np.median(values)),
        f"{prefix}std": float(np.std(values)),
        f"{prefix}iqr": float(p75 - p25),
        f"{prefix}min": float(np.min(values)),
        f"{prefix}max": float(np.max(values)),
        f"{prefix}p10": p10,
        f"{prefix}p90": p90,
    }


def extract_partition_first_order(
    image: np.ndarray,
    mask: np.ndarray,
    axis: str,
    n_bins: int,
    prefix: str,
) -> dict[str, float]:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return {}
    if axis == "depth":
        coords = rows.astype(float)
    elif axis == "width":
        coords = cols.astype(float)
    else:
        raise ValueError(f"Unsupported axis: {axis}")

    c_min = float(coords.min())
    c_max = float(coords.max())
    span = max(c_max - c_min, 1e-6)
    norm = (coords - c_min) / span
    bin_idx = np.floor(norm * n_bins).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    out: dict[str, float] = {}
    for b in range(n_bins):
        select = bin_idx == b
        values = image[rows[select], cols[select]] if np.any(select) else np.array([], dtype=float)
        out.update(extract_basic_stats(values, f"{prefix}bin{b}__"))
    return out


def extract_partition_texture(
    image: np.ndarray,
    mask: np.ndarray,
    axis: str,
    n_bins: int,
    prefix: str,
) -> dict[str, float]:
    if not HAS_LBP:
        return {}
    rows, cols = np.where(mask)
    if rows.size == 0:
        return {}
    if axis == "depth":
        coords = rows.astype(float)
    elif axis == "width":
        coords = cols.astype(float)
    else:
        raise ValueError(f"Unsupported axis: {axis}")

    c_min = float(coords.min())
    c_max = float(coords.max())
    span = max(c_max - c_min, 1e-6)
    norm = (coords - c_min) / span
    bin_idx = np.floor(norm * n_bins).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    quant = np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)
    lbp = local_binary_pattern(quant, P=8, R=1, method="uniform")
    out: dict[str, float] = {}
    for b in range(n_bins):
        select = bin_idx == b
        if not np.any(select):
            out[f"{prefix}bin{b}__lbp_entropy"] = np.nan
            out[f"{prefix}bin{b}__lbp_mean"] = np.nan
            continue
        vals = lbp[rows[select], cols[select]]
        hist, _ = np.histogram(vals, bins=10, range=(0, 10), density=True)
        hist = hist[hist > 0]
        entropy = float(-(hist * np.log2(hist)).sum()) if hist.size else np.nan
        out[f"{prefix}bin{b}__lbp_entropy"] = entropy
        out[f"{prefix}bin{b}__lbp_mean"] = float(np.mean(vals))
    return out


def extract_extra_features_one_row(row: pd.Series) -> tuple[dict[str, float], str | None]:
    sample_id = row["sample_id"]
    image_path = row.get("image_path")
    if not isinstance(image_path, str) or not image_path:
        return {"sample_id": sample_id}, "missing_image_path"
    image_file = Path(image_path)
    if not image_file.exists():
        return {"sample_id": sample_id}, "image_not_found"
    mask_file = infer_mask_path(row)
    if mask_file is None:
        return {"sample_id": sample_id}, "mask_not_found"

    try:
        image = load_grayscale_image(image_file)
        mask = load_mask(mask_file)
        if image.shape != mask.shape:
            mask = load_mask(mask_file, resize=(image.shape[1], image.shape[0]))
    except Exception as exc:
        return {"sample_id": sample_id}, f"load_error:{type(exc).__name__}"

    if not np.any(mask):
        return {"sample_id": sample_id}, "empty_mask"

    out: dict[str, float] = {"sample_id": sample_id}
    roi_values = image[mask]

    # A5: depth-normalized first-order in ROI
    row_mean = image.mean(axis=1, keepdims=True)
    row_std = image.std(axis=1, keepdims=True)
    row_std[row_std < 1e-6] = 1.0
    depth_norm = (image - row_mean) / row_std
    depth_values = depth_norm[mask]
    out.update(extract_basic_stats(depth_values, "depthnorm__roi__"))

    # C2: mask depth-position descriptors
    rows, cols = np.where(mask)
    h, w = mask.shape
    rmin, rmax = rows.min(), rows.max()
    cmin, cmax = cols.min(), cols.max()
    box_h = float(rmax - rmin + 1)
    box_w = float(cmax - cmin + 1)
    area = float(mask.sum())
    eroded = ndimage.binary_erosion(mask)
    border = mask & (~eroded)
    peri = float(border.sum())
    circularity = float(4.0 * np.pi * area / max(peri * peri, 1e-6))

    out.update(
        {
            "mask_depth__centroid": float(rows.mean() / max(h, 1)),
            "mask_depth__min": float(rmin / max(h, 1)),
            "mask_depth__max": float(rmax / max(h, 1)),
            "mask_depth__span": float(box_h / max(h, 1)),
            "mask_depth__superficial_frac": float(np.mean(rows < (h * 0.5))),
            "mask_depth__deep_frac": float(np.mean(rows >= (h * 0.5))),
            "mask_width__centroid": float(cols.mean() / max(w, 1)),
            "mask_width__span": float(box_w / max(w, 1)),
            "mask_shape__bbox_width": box_w,
            "mask_shape__bbox_height": box_h,
            "mask_shape__aspect_ratio": float(box_w / max(box_h, 1e-6)),
            "mask_shape__circularity": circularity,
        }
    )

    # D1/D2: partition first-order
    out.update(
        extract_partition_first_order(
            image=image,
            mask=mask,
            axis="depth",
            n_bins=2,
            prefix="part_depth2__",
        )
    )
    out.update(
        extract_partition_first_order(
            image=image,
            mask=mask,
            axis="depth",
            n_bins=4,
            prefix="part_depth4__",
        )
    )
    out.update(
        extract_partition_first_order(
            image=image,
            mask=mask,
            axis="width",
            n_bins=2,
            prefix="part_width2__",
        )
    )
    out.update(
        extract_partition_first_order(
            image=image,
            mask=mask,
            axis="width",
            n_bins=4,
            prefix="part_width4__",
        )
    )

    # D3: partition texture (LBP-only lightweight)
    out.update(
        extract_partition_texture(
            image=image,
            mask=mask,
            axis="depth",
            n_bins=2,
            prefix="part_tex_depth2__",
        )
    )
    out.update(
        extract_partition_texture(
            image=image,
            mask=mask,
            axis="depth",
            n_bins=4,
            prefix="part_tex_depth4__",
        )
    )
    out.update(
        extract_partition_texture(
            image=image,
            mask=mask,
            axis="width",
            n_bins=2,
            prefix="part_tex_width2__",
        )
    )
    out.update(
        extract_partition_texture(
            image=image,
            mask=mask,
            axis="width",
            n_bins=4,
            prefix="part_tex_width4__",
        )
    )

    return out, None


def build_or_load_extra_features(df: pd.DataFrame, cache_path: Path) -> tuple[pd.DataFrame, list[str]]:
    if cache_path.exists():
        loaded = pd.read_csv(cache_path)
        return loaded, []

    errors: dict[str, int] = {}
    rows: list[dict[str, float]] = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        out, err = extract_extra_features_one_row(row)
        rows.append(out)
        if err:
            errors[err] = errors.get(err, 0) + 1
        if i % 300 == 0 or i == total:
            print(f"[extra_features] processed {i}/{total}")

    frame = pd.DataFrame(rows)
    frame.to_csv(cache_path, index=False)

    notes = [f"{k}: {v}" for k, v in sorted(errors.items(), key=lambda x: x[0])]
    return frame, notes


def choose_first_order_columns(columns: list[str], scope: str) -> tuple[list[str], list[str]]:
    wanted = ["mean", "median", "std", "iqr", "min", "max", "p10", "p25", "p75", "p90", "skewness", "kurtosis"]
    fallback = {"p10": ["p10", "p5"], "p90": ["p90", "p95"]}
    selected: list[str] = []
    notes: list[str] = []

    for base in wanted:
        candidates = fallback.get(base, [base])
        picked = None
        for cand in candidates:
            col = f"{scope}__intensity__{cand}"
            if col in columns:
                picked = col
                if cand != base:
                    notes.append(f"{scope}:{base}-> {cand}")
                break
        if picked is not None:
            selected.append(picked)
        else:
            notes.append(f"{scope}:{base} missing")
    return selected, notes


def build_feature_sets(df: pd.DataFrame) -> tuple[dict[str, list[str]], list[str]]:
    columns = list(df.columns)
    notes: list[str] = []

    roi_first_order, notes_roi = choose_first_order_columns(columns, "roi")
    whole_first_order, notes_whole = choose_first_order_columns(columns, "whole_image")
    notes.extend(notes_roi)
    notes.extend(notes_whole)

    roi_mean = "roi__intensity__mean"
    texture_cols = [c for c in columns if "__texture__" in c]
    glcm_cols = [c for c in texture_cols if "__glcm__" in c]
    lbp_cols = [c for c in texture_cols if "__lbp__" in c]
    glrlm_glszm_cols = [c for c in texture_cols if ("__glrlm__" in c or "__glszm__" in c)]

    morphology_cols = [c for c in columns if c.startswith("roi__morphology__")]
    morphology_extra_basic = [
        c
        for c in columns
        if c.startswith("mask_shape__")
    ]
    morphology_depth_cols = [
        c
        for c in columns
        if c.startswith("mask_depth__") or c.startswith("mask_width__")
    ]

    depthnorm_cols = [c for c in columns if c.startswith("depthnorm__roi__")]

    partition_depth_cols = [c for c in columns if c.startswith("part_depth2__") or c.startswith("part_depth4__")]
    partition_width_cols = [c for c in columns if c.startswith("part_width2__") or c.startswith("part_width4__")]
    partition_texture_cols = [c for c in columns if c.startswith("part_tex_")]

    metadata_cols = [c for c in ["meta__sex_male", "height_cm", "weight_kg", "bmi"] if c in columns]

    feature_sets: dict[str, list[str]] = {
        "A1_ei_only_baseline": [roi_mean] if roi_mean in columns else [],
        "A2_roi_first_order": roi_first_order,
        "A3_whole_first_order": whole_first_order,
        "A4_roi_plus_whole_first_order": sorted(set(roi_first_order + whole_first_order)),
        "A5_depthnorm_roi_first_order": depthnorm_cols,
        "B1_glcm_only": glcm_cols,
        "B2_lbp_only": lbp_cols,
        "B3_glrlm_glszm_only": glrlm_glszm_cols,
        "B4_texture_only": texture_cols,
        "B5_first_order_plus_texture": sorted(set(roi_first_order + whole_first_order + texture_cols)),
        "C1_morphology_basic": sorted(set(morphology_cols + morphology_extra_basic)),
        "C2_morphology_depth_position": morphology_depth_cols,
        "C3_morphology_only": sorted(set(morphology_cols + morphology_extra_basic + morphology_depth_cols)),
        "C4_first_order_plus_morphology": sorted(set(roi_first_order + morphology_cols + morphology_extra_basic + morphology_depth_cols)),
        "C5_first_order_texture_morphology": sorted(
            set(roi_first_order + whole_first_order + texture_cols + morphology_cols + morphology_extra_basic + morphology_depth_cols)
        ),
        "D1_partition_first_order_depth": partition_depth_cols,
        "D2_partition_first_order_width": partition_width_cols,
        "D3_partition_first_order_plus_partition_texture": sorted(set(partition_depth_cols + partition_width_cols + partition_texture_cols)),
        "D4_roi_plus_partition": sorted(set(roi_first_order + partition_depth_cols + partition_width_cols + partition_texture_cols)),
        "E1_metadata_only": metadata_cols,
        "E2_first_order_plus_metadata": sorted(set(roi_first_order + metadata_cols)),
        "E3_texture_plus_metadata": sorted(set(texture_cols + metadata_cols)),
        "E4_full_plus_metadata": sorted(
            set(
                roi_first_order
                + whole_first_order
                + texture_cols
                + morphology_cols
                + morphology_extra_basic
                + morphology_depth_cols
                + partition_depth_cols
                + partition_width_cols
                + partition_texture_cols
                + metadata_cols
            )
        ),
    }
    return feature_sets, notes


def build_model(model_name: str, seed: int) -> Pipeline:
    if model_name == "ridge":
        model = RidgeCV(alphas=np.logspace(-3, 3, 31))
    elif model_name == "elasticnet":
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            alphas=np.logspace(-3, 1, 25),
            cv=5,
            max_iter=8000,
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def build_group_folds(df: pd.DataFrame, n_splits: int, seed: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    splitter = RegressionStratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
        bin_strategy="quantile",
        n_bins=5,
    )
    x_dummy = np.zeros((len(df), 1), dtype=float)
    y = df["true_age"].to_numpy(dtype=float)
    groups = df["subject_id"].to_numpy()

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    fold_id = np.zeros(len(df), dtype=int)
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(x_dummy, y=y, groups=groups), start=1):
        folds.append((train_idx, test_idx))
        fold_id[test_idx] = fold_idx
    return folds, fold_id


def run_cv_predict(
    df: pd.DataFrame,
    feature_cols: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    model_name: str,
    seed: int,
) -> np.ndarray:
    pred = np.full(len(df), np.nan, dtype=float)
    x_all = df[feature_cols]
    y_all = df["true_age"].to_numpy(dtype=float)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for train_idx, test_idx in folds:
            model = build_model(model_name=model_name, seed=seed)
            model.fit(x_all.iloc[train_idx], y_all[train_idx])
            pred[test_idx] = model.predict(x_all.iloc[test_idx])
    return pred


def evaluate_predictions(df: pd.DataFrame, state_age: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    out = df[["sample_id", "subject_id", "true_age", "pred_age", "fold"]].copy()
    out["state_age"] = state_age
    out["gap_pred_true"] = out["pred_age"] - out["true_age"]
    out["gap_pred_state"] = out["pred_age"] - out["state_age"]

    sample_gap_mae = float(mean_absolute_error(out["pred_age"], out["state_age"]))
    sample_gap_rmse = rmse(out["pred_age"].to_numpy(), out["state_age"].to_numpy())

    subject_out = (
        out.groupby("subject_id", as_index=False)[["true_age", "pred_age", "state_age", "gap_pred_true", "gap_pred_state"]]
        .mean()
    )
    subject_gap_mae = float(mean_absolute_error(subject_out["pred_age"], subject_out["state_age"]))
    subject_gap_rmse = rmse(subject_out["pred_age"].to_numpy(), subject_out["state_age"].to_numpy())

    state_age_vs_true_mae = float(mean_absolute_error(out["true_age"], out["state_age"]))
    state_age_vs_true_corr = safe_corr(out["true_age"].to_numpy(), out["state_age"].to_numpy())
    state_age_std = float(out["state_age"].std(ddof=1))
    state_age_min = float(out["state_age"].min())
    state_age_max = float(out["state_age"].max())

    metrics = {
        "sample_gap_mae": sample_gap_mae,
        "subject_gap_mae": subject_gap_mae,
        "sample_gap_rmse": sample_gap_rmse,
        "subject_gap_rmse": subject_gap_rmse,
        "state_age_vs_true_mae": state_age_vs_true_mae,
        "state_age_vs_true_corr": state_age_vs_true_corr,
        "state_age_std": state_age_std,
        "state_age_min": state_age_min,
        "state_age_max": state_age_max,
        "n_samples": int(len(out)),
        "n_subjects": int(subject_out["subject_id"].nunique()),
    }
    return out, metrics


def save_scatter(x: np.ndarray, y: np.ndarray, path: Path, title: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(6.5, 6))
    plt.scatter(x, y, s=10, alpha=0.55)
    lo = float(np.nanmin([np.nanmin(x), np.nanmin(y)]))
    hi = float(np.nanmax([np.nanmax(x), np.nanmax(y)]))
    plt.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_bar(values: pd.DataFrame, path: Path, title: str) -> None:
    fig_h = max(5.0, 0.3 * len(values))
    plt.figure(figsize=(10, fig_h))
    plt.barh(values["feature_set"], values["subject_gap_mae"], color="#3b82f6")
    plt.gca().invert_yaxis()
    plt.xlabel("subject_gap_mae")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_summary(
    run_dir: Path,
    leaderboard: pd.DataFrame,
    baseline_row: pd.Series,
    best_row: pd.Series,
    skipped: list[str],
    feature_notes: list[str],
    extra_notes: list[str],
    input_paths: dict[str, str],
    pred_true_sample_mae: float,
    pred_true_subject_mae: float,
) -> None:
    ridge_board = leaderboard[leaderboard["model"] == "ridge"].sort_values("subject_gap_mae").reset_index(drop=True)
    top5 = ridge_board.head(5)

    degeneration_flags = []
    if float(best_row["state_age_std"]) < 4.0:
        degeneration_flags.append("state_age 标准差偏小")
    if float(best_row["state_age_vs_true_corr"]) < 0.35:
        degeneration_flags.append("state_age 与真实年龄相关性偏低")
    if float(best_row["state_age_max"]) - float(best_row["state_age_min"]) < 20.0:
        degeneration_flags.append("state_age 动态范围偏窄")

    lines: list[str] = []
    lines.append("# state_age 特征基准实验总结")
    lines.append("")
    lines.append("## 输入")
    lines.append(f"- pred_age 输入: {input_paths['pred']}")
    lines.append(f"- 缓存特征表: {input_paths['feature_table']}")
    lines.append(f"- 图像根目录: {input_paths['images']}")
    lines.append(f"- mask 根目录: {input_paths['masks']}")
    lines.append("")
    lines.append("## 主结论")
    lines.append(
        f"- Ridge 主榜第一: {best_row['feature_set']} | sample_gap_mae={best_row['sample_gap_mae']:.4f}, subject_gap_mae={best_row['subject_gap_mae']:.4f}"
    )
    lines.append(
        f"- EI-only baseline: sample_gap_mae={baseline_row['sample_gap_mae']:.4f}, subject_gap_mae={baseline_row['subject_gap_mae']:.4f}"
    )
    lines.append(
        f"- 相比 EI-only 的绝对提升: sample={best_row['gain_vs_ei_sample']:.4f}, subject={best_row['gain_vs_ei_subject']:.4f}"
    )
    lines.append(
        f"- 相比 pred_age vs true_age 的缩小量: sample={best_row['gain_vs_true_sample']:.4f}, subject={best_row['gain_vs_true_subject']:.4f}"
    )
    lines.append("")
    lines.append("## Top 5 (Ridge, 按 subject_gap_mae 升序)")
    lines.append("")
    lines.append("| rank | feature_set | n_features | sample_gap_mae | subject_gap_mae | gain_vs_ei_subject | state_age_vs_true_corr |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for idx, (_, row) in enumerate(top5.iterrows(), start=1):
        lines.append(
            f"| {idx} | {row['feature_set']} | {int(row['n_features'])} | {row['sample_gap_mae']:.4f} | {row['subject_gap_mae']:.4f} | {row['gain_vs_ei_subject']:.4f} | {row['state_age_vs_true_corr']:.4f} |"
        )
    lines.append("")
    lines.append("## sanity check")
    lines.append(f"- pred_age vs true_age MAE: sample={pred_true_sample_mae:.4f}, subject={pred_true_subject_mae:.4f}")
    lines.append(f"- 最优方案 state_age_vs_true_mae={best_row['state_age_vs_true_mae']:.4f}")
    lines.append(f"- 最优方案 state_age_vs_true_corr={best_row['state_age_vs_true_corr']:.4f}")
    lines.append(f"- 最优方案 state_age_std={best_row['state_age_std']:.4f}")
    lines.append(f"- 最优方案 state_age 范围=[{best_row['state_age_min']:.4f}, {best_row['state_age_max']:.4f}]")
    if degeneration_flags:
        lines.append(f"- 退化告警: {'; '.join(degeneration_flags)}")
    else:
        lines.append("- 未发现明显退化迹象。")

    if skipped:
        lines.append("")
        lines.append("## 跳过项")
        for item in skipped:
            lines.append(f"- {item}")

    if feature_notes:
        lines.append("")
        lines.append("## 特征构建备注")
        for item in sorted(set(feature_notes)):
            lines.append(f"- {item}")

    if extra_notes:
        lines.append("")
        lines.append("## 图像+mask 现算备注")
        for item in extra_notes:
            lines.append(f"- {item}")

    lines.append("")
    lines.append("## 输出清单")
    lines.append("- leaderboard.csv")
    lines.append("- feature_sets/<feature_set>/results.csv")
    lines.append("- figures/*.png")
    lines.append("- inputs_used.json")

    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark feature sets for state_age fitting under a unified subject-level CV protocol."
    )
    parser.add_argument(
        "--pred-file",
        default="outputs/run_20260411_003506_ta_healthy_nested_cv_fusion_whole_roi_masked_cached_ridge/predictions_readable.csv",
        help="Prediction file containing pred_age.",
    )
    parser.add_argument(
        "--feature-table",
        default="outputs/cache_feature_tables/ta_healthy_whole_plus_roi_original_size.csv",
        help="Cached feature table path.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/state_age_feature_benchmark",
        help="Root directory for benchmark outputs.",
    )
    parser.add_argument("--run-name", default=None, help="Optional run name.")
    parser.add_argument("--min-age", type=float, default=18.0)
    parser.add_argument("--max-age", type=float, default=100.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--elasticnet-sets",
        default="A1_ei_only_baseline,A2_roi_first_order,B4_texture_only,B5_first_order_plus_texture,C5_first_order_texture_morphology,D4_roi_plus_partition,E4_full_plus_metadata",
        help="Comma-separated feature_set names for ElasticNet robustness runs.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    run_name = args.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_state_age_feature_benchmark"
    run_dir = (PROJECT_ROOT / args.output_root / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "feature_sets").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "tables").mkdir(exist_ok=True)

    pred_path = (PROJECT_ROOT / args.pred_file).resolve()
    feature_path = (PROJECT_ROOT / args.feature_table).resolve()

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature table not found: {feature_path}")

    pred_df = pd.read_csv(pred_path)
    feat_df = pd.read_csv(feature_path)

    pred_columns = pred_df.columns.tolist()
    feat_columns = feat_df.columns.tolist()

    sample_col = resolve_column(pred_columns, SAMPLE_CANDIDATES, "sample_id" if "sample_id" in pred_columns else None, "sample")
    subject_col = resolve_column(pred_columns, SUBJECT_CANDIDATES, None, "subject")
    true_col = resolve_column(pred_columns, TRUE_AGE_CANDIDATES, None, "true_age")
    pred_col = resolve_column(pred_columns, PRED_COLUMN_CANDIDATES, None, "pred_age")

    if "sample_id" not in feat_df.columns:
        raise KeyError("Feature table must contain sample_id.")
    if "subject_id" not in feat_df.columns:
        raise KeyError("Feature table must contain subject_id.")

    pred_core = pred_df[[sample_col, subject_col, true_col, pred_col]].copy()
    pred_core = pred_core.rename(
        columns={sample_col: "sample_id", subject_col: "subject_id", true_col: "true_age", pred_col: "pred_age"}
    )
    pred_core = pred_core.drop_duplicates(subset=["sample_id"], keep="first")

    feat_df = feat_df.drop_duplicates(subset=["sample_id"], keep="first")

    merged = pred_core.merge(feat_df, on=["sample_id", "subject_id"], how="inner", suffixes=("", "_feat"))

    if merged.empty:
        raise ValueError("No overlap after merging prediction file and feature table.")

    # Metadata cleanup
    if "sex" in merged.columns:
        merged["meta__sex_male"] = (
            merged["sex"].astype(str).str.strip().str.upper().map({"M": 1.0, "F": 0.0})
        )

    merged["true_age"] = pd.to_numeric(merged["true_age"], errors="coerce")
    merged["pred_age"] = pd.to_numeric(merged["pred_age"], errors="coerce")
    for col in ["height_cm", "weight_kg", "bmi"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged = merged.dropna(subset=["sample_id", "subject_id", "true_age", "pred_age"]).copy()
    merged = merged[(merged["true_age"] >= args.min_age) & (merged["true_age"] <= args.max_age)].copy()

    if len(merged) < 50:
        raise ValueError(f"Too few usable rows after filtering: {len(merged)}")

    # Compute optional image+mask extra features
    extra_cache = run_dir / "tables" / "extra_features.csv"
    extra_features, extra_notes = build_or_load_extra_features(merged, extra_cache)
    merged = merged.merge(extra_features, on="sample_id", how="left")

    feature_sets, feature_notes = build_feature_sets(merged)

    folds, fold_id = build_group_folds(merged, n_splits=args.n_splits, seed=args.seed)
    merged["fold"] = fold_id

    pred_true_sample_mae = float(mean_absolute_error(merged["pred_age"], merged["true_age"]))
    subject_base = merged.groupby("subject_id", as_index=False)[["true_age", "pred_age"]].mean()
    pred_true_subject_mae = float(mean_absolute_error(subject_base["pred_age"], subject_base["true_age"]))

    leaderboard_rows: list[dict[str, Any]] = []
    skipped: list[str] = []

    elasticnet_set_names = {x.strip() for x in args.elasticnet_sets.split(",") if x.strip()}

    for feature_set_name, raw_cols in feature_sets.items():
        cols = [c for c in dict.fromkeys(raw_cols) if c in merged.columns]
        if not cols:
            skipped.append(f"{feature_set_name}: no available columns")
            continue

        numeric_frame = merged[cols].apply(pd.to_numeric, errors="coerce")
        valid_ratio = 1.0 - float(numeric_frame.isna().all(axis=1).mean())
        if valid_ratio < 0.5:
            skipped.append(f"{feature_set_name}: too many fully-missing rows (valid_ratio={valid_ratio:.3f})")
            continue

        for model_name in ["ridge", "elasticnet"]:
            if model_name == "elasticnet" and feature_set_name not in elasticnet_set_names:
                continue

            try:
                state_age = run_cv_predict(
                    df=merged,
                    feature_cols=cols,
                    folds=folds,
                    model_name=model_name,
                    seed=args.seed,
                )
                result_df, metrics = evaluate_predictions(merged, state_age)
                fs_dir = run_dir / "feature_sets" / feature_set_name
                fs_dir.mkdir(exist_ok=True)
                if model_name == "ridge":
                    result_path = fs_dir / "results.csv"
                    subject_path = fs_dir / "subject_results.csv"
                else:
                    result_path = fs_dir / f"results_{model_name}.csv"
                    subject_path = fs_dir / f"subject_results_{model_name}.csv"

                result_df.to_csv(result_path, index=False)
                result_df.groupby("subject_id", as_index=False)[
                    ["true_age", "pred_age", "state_age", "gap_pred_true", "gap_pred_state"]
                ].mean().to_csv(subject_path, index=False)

                leaderboard_rows.append(
                    {
                        "feature_set": feature_set_name,
                        "model": model_name,
                        "n_features": int(len(cols)),
                        "sample_gap_mae": metrics["sample_gap_mae"],
                        "subject_gap_mae": metrics["subject_gap_mae"],
                        "sample_gap_rmse": metrics["sample_gap_rmse"],
                        "subject_gap_rmse": metrics["subject_gap_rmse"],
                        "gain_vs_ei_sample": np.nan,
                        "gain_vs_ei_subject": np.nan,
                        "gain_vs_true_sample": float(pred_true_sample_mae - metrics["sample_gap_mae"]),
                        "gain_vs_true_subject": float(pred_true_subject_mae - metrics["subject_gap_mae"]),
                        "state_age_vs_true_mae": metrics["state_age_vs_true_mae"],
                        "state_age_vs_true_corr": metrics["state_age_vs_true_corr"],
                        "state_age_std": metrics["state_age_std"],
                        "state_age_min": metrics["state_age_min"],
                        "state_age_max": metrics["state_age_max"],
                        "status": "ok",
                        "note": "",
                    }
                )
            except Exception as exc:
                skipped.append(f"{feature_set_name}/{model_name}: {type(exc).__name__}: {exc}")

    if not leaderboard_rows:
        raise RuntimeError("No feature set finished successfully.")

    leaderboard = pd.DataFrame(leaderboard_rows)

    # gain_vs_ei by model
    for model_name in sorted(leaderboard["model"].unique()):
        mask = leaderboard["model"] == model_name
        model_df = leaderboard[mask]
        baseline = model_df[model_df["feature_set"] == "A1_ei_only_baseline"]
        if baseline.empty:
            skipped.append(f"{model_name}: baseline A1_ei_only_baseline missing, gain_vs_ei unavailable")
            continue
        baseline_sample = float(baseline.iloc[0]["sample_gap_mae"])
        baseline_subject = float(baseline.iloc[0]["subject_gap_mae"])
        leaderboard.loc[mask, "gain_vs_ei_sample"] = baseline_sample - leaderboard.loc[mask, "sample_gap_mae"]
        leaderboard.loc[mask, "gain_vs_ei_subject"] = baseline_subject - leaderboard.loc[mask, "subject_gap_mae"]

    leaderboard = leaderboard.sort_values(["model", "subject_gap_mae", "sample_gap_mae"]).reset_index(drop=True)
    leaderboard.to_csv(run_dir / "leaderboard.csv", index=False)

    ridge_board = leaderboard[leaderboard["model"] == "ridge"].sort_values("subject_gap_mae").reset_index(drop=True)
    if ridge_board.empty:
        raise RuntimeError("Ridge results are empty. Cannot build main summary.")

    baseline_row = ridge_board[ridge_board["feature_set"] == "A1_ei_only_baseline"]
    if baseline_row.empty:
        raise RuntimeError("Ridge baseline A1_ei_only_baseline is missing.")
    baseline_row = baseline_row.iloc[0]
    best_row = ridge_board.iloc[0]

    # Figures
    best_set = best_row["feature_set"]
    best_results = pd.read_csv(run_dir / "feature_sets" / best_set / "results.csv")
    baseline_results = pd.read_csv(run_dir / "feature_sets" / "A1_ei_only_baseline" / "results.csv")

    save_scatter(
        x=baseline_results["true_age"].to_numpy(),
        y=baseline_results["pred_age"].to_numpy(),
        path=run_dir / "figures" / "pred_age_vs_true_age.png",
        title="pred_age vs true_age",
        xlabel="true_age",
        ylabel="pred_age",
    )
    save_scatter(
        x=best_results["state_age"].to_numpy(),
        y=best_results["pred_age"].to_numpy(),
        path=run_dir / "figures" / "pred_age_vs_state_age_best.png",
        title=f"pred_age vs state_age ({best_set})",
        xlabel="state_age",
        ylabel="pred_age",
    )
    save_scatter(
        x=best_results["true_age"].to_numpy(),
        y=best_results["state_age"].to_numpy(),
        path=run_dir / "figures" / "state_age_vs_true_age_best.png",
        title=f"state_age vs true_age ({best_set})",
        xlabel="true_age",
        ylabel="state_age",
    )

    bar_df = ridge_board[["feature_set", "subject_gap_mae"]].copy()
    save_bar(bar_df, run_dir / "figures" / "subject_gap_mae_by_feature_set.png", "subject-level gap_mae by feature set (Ridge)")

    plt.figure(figsize=(7, 5.5))
    residual = best_results["state_age"].to_numpy() - best_results["true_age"].to_numpy()
    plt.scatter(best_results["true_age"], residual, s=10, alpha=0.55)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("true_age")
    plt.ylabel("state_age - true_age")
    plt.title(f"Residual vs Age ({best_set})")
    plt.tight_layout()
    plt.savefig(run_dir / "figures" / "best_state_age_residual_vs_age.png", dpi=180)
    plt.close()

    inputs_used = {
        "pred": str(pred_path),
        "feature_table": str(feature_path),
        "images": "/home/szdx/LNX/data/TA/Healthy/Images",
        "masks": "/home/szdx/LNX/data/TA/Healthy/Masks",
        "n_rows_after_merge_age_filter": int(len(merged)),
        "n_subjects": int(merged["subject_id"].nunique()),
        "pred_true_sample_mae": pred_true_sample_mae,
        "pred_true_subject_mae": pred_true_subject_mae,
        "lbp_available": HAS_LBP,
    }
    (run_dir / "inputs_used.json").write_text(json.dumps(inputs_used, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    write_summary(
        run_dir=run_dir,
        leaderboard=leaderboard,
        baseline_row=baseline_row,
        best_row=best_row,
        skipped=skipped,
        feature_notes=feature_notes,
        extra_notes=extra_notes,
        input_paths=inputs_used,
        pred_true_sample_mae=pred_true_sample_mae,
        pred_true_subject_mae=pred_true_subject_mae,
    )

    print("=== state_age feature benchmark complete ===")
    print(f"run_dir: {run_dir}")
    print("Top-5 Ridge by subject_gap_mae:")
    show_cols = [
        "feature_set",
        "n_features",
        "sample_gap_mae",
        "subject_gap_mae",
        "gain_vs_ei_sample",
        "gain_vs_ei_subject",
        "state_age_vs_true_corr",
        "state_age_std",
    ]
    print(ridge_board[show_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
