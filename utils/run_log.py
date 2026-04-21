"""Helpers for writing a concise one-line cross-run experiment log."""

from __future__ import annotations

import json
import math
import re
from csv import reader as csv_reader
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.config import load_yaml
from utils.io import load_json

LOG_HEADER = (
    "usage_predict_feature_engineering run log\n"
    "mae | run | r2 | feat(sel/raw) | special\n"
)
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
MISSING = object()


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _format_number(value: Any) -> str:
    if value is None or _is_nan(value):
        return "NaN"
    return f"{float(value):.3f}"


def _format_aligned_mae(value: Any) -> str:
    if value is None or _is_nan(value):
        return "NaN"
    numeric = float(value)
    if numeric >= 0:
        return f"{numeric:06.3f}"
    return f"{numeric:07.3f}"


def _stringify_value(value: Any) -> str:
    if value is MISSING:
        return "missing"
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if _is_nan(value):
            return "NaN"
        return f"{value:.6g}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _get_by_path(mapping: dict[str, Any], dotted_path: str) -> Any:
    cursor: Any = mapping
    for key in dotted_path.split("."):
        if not isinstance(cursor, dict) or key not in cursor:
            return MISSING
        cursor = cursor[key]
    return cursor


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _short_list(value: Any) -> str:
    if not isinstance(value, list):
        return _stringify_value(value)
    return "+".join(str(item) for item in value)


def _collect_special_tags(config: dict[str, Any], default_config: dict[str, Any]) -> list[str]:
    tags: list[str] = []

    experiment = _get_by_path(config, "experiment.name")
    if experiment is not MISSING:
        tags.append(f"exp={experiment}")

    adapter = _get_by_path(config, "data.adapter")
    if adapter is not MISSING:
        tags.append(f"adapter={adapter}")

    feature_mode = _get_by_path(config, "data.feature_mode")
    if feature_mode is not MISSING and feature_mode != _get_by_path(default_config, "data.feature_mode"):
        mode_alias = {
            "image_only": "img",
            "metadata_only": "meta",
            "image_plus_metadata": "img+meta",
        }.get(str(feature_mode), str(feature_mode))
        tags.append(f"mode={mode_alias}")

    scopes = _get_by_path(config, "feature_extraction.scopes")
    if scopes is not MISSING and scopes != _get_by_path(default_config, "feature_extraction.scopes"):
        tags.append(f"scope={_short_list(scopes)}")

    groups = _get_by_path(config, "feature_extraction.enabled_groups")
    if groups is not MISSING and groups != _get_by_path(default_config, "feature_extraction.enabled_groups"):
        tags.append(f"groups={_short_list(groups)}")

    if _get_by_path(config, "image.preprocessing.use_original_size") is True:
        tags.append("orig_size")

    if _get_by_path(config, "paths.feature_table") not in {MISSING, None, ""}:
        tags.append("cached_features")

    split_strategy = _get_by_path(config, "split.strategy")
    if split_strategy is not MISSING and split_strategy != _get_by_path(default_config, "split.strategy"):
        tags.append(f"split={split_strategy}")

    split_folds = _get_by_path(config, "split.n_splits")
    if split_folds is not MISSING and split_folds != _get_by_path(default_config, "split.n_splits"):
        tags.append(f"folds={split_folds}")

    age_bins = _get_by_path(config, "split.age_bins")
    if age_bins is not MISSING and age_bins != _get_by_path(default_config, "split.age_bins"):
        tags.append(f"age_bins={age_bins}")

    if _get_by_path(config, "selection.enabled") is True:
        method = _get_by_path(config, "selection.method")
        tags.append(f"selection={method if method is not MISSING else 'on'}")

    if _get_by_path(config, "preprocessing.pca.enabled") is True:
        tags.append("pca")

    if _get_by_path(config, "explain.shap.enabled") is True:
        tags.append("shap")

    search_enabled = _get_by_path(config, "search.enabled")
    if search_enabled is False:
        tags.append("no_search")
    elif search_enabled is True:
        search_method = _get_by_path(config, "search.method")
        default_search_method = _get_by_path(default_config, "search.method")
        if search_method is not MISSING and search_method != default_search_method:
            tags.append(f"search={search_method}")

    model_params = _get_by_path(config, "model.params")
    default_model_params = _get_by_path(default_config, "model.params")
    if isinstance(model_params, dict):
        for key in sorted(model_params):
            value = model_params[key]
            default_value = default_model_params.get(key, MISSING) if isinstance(default_model_params, dict) else MISSING
            if value != default_value:
                tags.append(f"{key}={_stringify_value(value)}")

    deduped: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        normalized = _normalize_text(tag)
        if not normalized or normalized in seen:
            continue
        deduped.append(tag)
        seen.add(normalized)
    return deduped[:8]


def _parse_run_timestamp(run_dir: Path) -> str:
    match = re.match(r"run_(\d{8})_(\d{6})_", run_dir.name)
    if match:
        return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    return datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def _short_run_name(run_dir: Path) -> str:
    match = re.match(r"(run_\d{8}_\d{6})", run_dir.name)
    if match:
        return match.group(1)
    return run_dir.name


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        row_count = sum(1 for _ in handle)
    return max(row_count - 1, 0)


def _count_raw_features(run_dir: Path) -> int:
    extracted_names = run_dir / "tables" / "extracted_feature_names.csv"
    count = _count_csv_rows(extracted_names)
    if count > 0:
        return count

    raw_table = run_dir / "tables" / "features_raw.csv"
    if not raw_table.exists():
        return 0
    with raw_table.open("r", encoding="utf-8") as handle:
        header = next(csv_reader(handle), [])
    return sum("__" in column for column in header)


def build_run_log_entry(
    run_dir: str | Path,
    config: dict[str, Any],
    mode: str,
    primary_metrics: dict[str, Any],
    raw_feature_count: int,
    selected_feature_count: int,
    validation_metrics: dict[str, Any] | None = None,
    fold_summary: dict[str, dict[str, Any]] | None = None,
    timestamp_text: str | None = None,
) -> str:
    """Build one concise one-line log entry for a run."""
    run_path = Path(run_dir)
    default_config = load_yaml(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else {}
    special_tags = _collect_special_tags(config, default_config)
    if mode == "holdout" and validation_metrics is not None:
        special_tags.append(f"val_mae={_format_number(validation_metrics.get('mae'))}")
    elif fold_summary:
        mean_metrics = fold_summary.get("mean", {}) or {}
        std_metrics = fold_summary.get("std", {}) or {}
        if mean_metrics.get("mae") is not None:
            special_tags.append(
                f"cv_mae={_format_number(mean_metrics.get('mae'))}±{_format_number(std_metrics.get('mae'))}"
            )
    special_segment = ", ".join(special_tags) if special_tags else "-"
    mae_value = _format_aligned_mae(primary_metrics.get("mae"))
    r2_value = _format_number(primary_metrics.get("r2"))
    return (
        f"{mae_value} | {_short_run_name(run_path)} | {r2_value} | "
        f"{selected_feature_count}/{raw_feature_count} | {special_segment}\n"
    )


def append_run_log(
    run_dir: str | Path,
    config: dict[str, Any],
    mode: str,
    primary_metrics: dict[str, Any],
    raw_feature_count: int,
    selected_feature_count: int,
    validation_metrics: dict[str, Any] | None = None,
    fold_summary: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Append one run entry to ``outputs/log.txt``."""
    run_path = Path(run_dir)
    log_path = run_path.parent / "log.txt"
    is_new_file = not log_path.exists() or log_path.stat().st_size == 0
    entry = build_run_log_entry(
        run_dir=run_path,
        config=config,
        mode=mode,
        primary_metrics=primary_metrics,
        raw_feature_count=raw_feature_count,
        selected_feature_count=selected_feature_count,
        validation_metrics=validation_metrics,
        fold_summary=fold_summary,
    )
    with log_path.open("a", encoding="utf-8") as handle:
        if is_new_file:
            handle.write(LOG_HEADER)
        handle.write(entry)
    return log_path


def _build_saved_run_entry(run_dir: Path) -> str | None:
    config_path = run_dir / "config_used.yaml"
    metrics_path = run_dir / "tables" / "metrics.json"
    if not config_path.exists() or not metrics_path.exists():
        return None

    config = load_yaml(config_path)
    metrics_payload = load_json(metrics_path)
    if "validation" in metrics_payload and "test" in metrics_payload:
        mode = "holdout"
        primary_metrics = metrics_payload.get("test", {})
        validation_metrics = metrics_payload.get("validation", {})
        fold_summary = None
    elif "outer_cv_pooled" in metrics_payload:
        mode = "nested_cv"
        primary_metrics = metrics_payload.get("outer_cv_pooled", {})
        validation_metrics = None
        fold_summary = {
            "mean": metrics_payload.get("fold_metrics_mean", {}) or {},
            "std": metrics_payload.get("fold_metrics_std", {}) or {},
        }
    else:
        return None

    raw_feature_count = _count_raw_features(run_dir)
    selected_feature_count = _count_csv_rows(run_dir / "tables" / "selected_features.csv")
    return build_run_log_entry(
        run_dir=run_dir,
        config=config,
        mode=mode,
        primary_metrics=primary_metrics,
        raw_feature_count=raw_feature_count,
        selected_feature_count=selected_feature_count,
        validation_metrics=validation_metrics,
        fold_summary=fold_summary,
        timestamp_text=_parse_run_timestamp(run_dir),
    )


def rebuild_run_log(outputs_root: str | Path) -> tuple[Path, int]:
    """Rebuild ``outputs/log.txt`` from all existing ``run_*`` directories."""
    outputs_path = Path(outputs_root)
    run_dirs = sorted(path for path in outputs_path.iterdir() if path.is_dir() and path.name.startswith("run_"))
    entries = [entry for path in run_dirs if (entry := _build_saved_run_entry(path)) is not None]
    log_path = outputs_path / "log.txt"
    content = LOG_HEADER + "".join(entries) if entries else LOG_HEADER
    log_path.write_text(content, encoding="utf-8")
    return log_path, len(entries)
