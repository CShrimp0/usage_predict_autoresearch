"""Construct the canonical sample-level dataframe."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from dataio.load_metadata import load_metadata_table

LOGGER = logging.getLogger("usage_predict_feature_engineering")

CORE_COLUMNS = {
    "sample_id",
    "subject_id",
    "image_path",
    "mask_path",
    "roi_path",
    "age",
    "split",
}


def _clean_optional_value(value: Any) -> str | None:
    if value in (None, "", "null", "None", "nan"):
        return None
    return str(value)


def _resolve_path(value: Any, root: str | Path | None) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.is_absolute():
        return str(candidate)
    if root is None:
        return str(candidate)
    return str((Path(root) / candidate).resolve())


def _rename_columns(df: pd.DataFrame, mappings: dict[str, Any]) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    required_keys = {"sample_id", "subject_id", "age"}
    for standard_name, source_name in mappings.items():
        if standard_name == "extra" or source_name in (None, "", "null", "None"):
            continue
        source = str(source_name)
        if source not in df.columns:
            if standard_name in required_keys:
                raise KeyError(f"Configured column '{source}' not found in metadata table.")
            continue
        rename_map[source] = standard_name

    extra_columns = mappings.get("extra", {}) or {}
    for standard_name, source_name in extra_columns.items():
        if source_name in (None, "", "null", "None"):
            continue
        source = str(source_name)
        if source not in df.columns:
            continue
        rename_map[source] = standard_name

    return df.rename(columns=rename_map)


def _resolve_image_columns(df: pd.DataFrame, data_config: dict[str, Any]) -> pd.DataFrame:
    columns = data_config["columns"]
    image_root = _clean_optional_value(data_config.get("image_root"))
    mask_root = _clean_optional_value(data_config.get("mask_root"))

    if "image_path" not in df.columns:
        if "image_filename" in df.columns:
            df["image_path"] = df["image_filename"].map(lambda x: _resolve_path(x, image_root))
        else:
            raise KeyError("No image path information available. Configure image_path or image_filename.")
    else:
        df["image_path"] = df["image_path"].map(lambda x: _resolve_path(x, image_root))

    if "mask_path" in df.columns:
        df["mask_path"] = df["mask_path"].map(lambda x: _resolve_path(x, mask_root))
    elif "roi_path" in df.columns:
        df["mask_path"] = df["roi_path"].map(lambda x: _resolve_path(x, mask_root))
    else:
        df["mask_path"] = None

    if "roi_path" not in df.columns:
        df["roi_path"] = df["mask_path"]

    return df


def _validate_dataframe(df: pd.DataFrame, data_config: dict[str, Any]) -> pd.DataFrame:
    required = ["sample_id", "subject_id", "age", "image_path"]
    for column in required:
        if column not in df.columns:
            raise KeyError(f"Required standardized column missing: {column}")

    df = df.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df["subject_id"] = df["subject_id"].astype(str)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    if data_config.get("drop_rows_missing_target", True):
        before = len(df)
        df = df.dropna(subset=["age"]).reset_index(drop=True)
        dropped = before - len(df)
        if dropped > 0:
            LOGGER.warning("Dropped %d rows with missing target age.", dropped)

    missing_paths = df["image_path"].isna().sum()
    if missing_paths > 0:
        raise ValueError(f"Found {missing_paths} rows with missing image_path.")

    if data_config.get("deduplicate_on"):
        key = data_config["deduplicate_on"]
        before = len(df)
        df = df.drop_duplicates(subset=[key]).reset_index(drop=True)
        dropped = before - len(df)
        if dropped > 0:
            LOGGER.warning("Dropped %d duplicate rows using key '%s'.", dropped, key)

    if data_config.get("strict_subject_consistency", True):
        counts = df.groupby("sample_id")["subject_id"].nunique()
        inconsistent = counts[counts > 1]
        if not inconsistent.empty:
            raise ValueError(
                "Some sample_id values map to multiple subject_id values: "
                f"{inconsistent.index.tolist()[:5]}"
            )

    return df


def summarize_missingness(df: pd.DataFrame) -> dict[str, int]:
    """Return a per-column missing-value summary."""
    return df.isna().sum().sort_values(ascending=False).astype(int).to_dict()


def build_dataframe(config: dict[str, Any]) -> pd.DataFrame:
    """Build the canonical dataframe used by downstream scripts."""
    data_config = config["data"]
    metadata = load_metadata_table(
        metadata_path=data_config["metadata_path"],
        file_type=data_config.get("metadata_format", "auto"),
        sheet_name=data_config.get("sheet_name"),
    )
    standardized = _rename_columns(metadata, data_config["columns"])
    standardized = _resolve_image_columns(standardized, data_config)
    standardized = _validate_dataframe(standardized, data_config)

    LOGGER.info("Loaded %d samples across %d subjects.", len(standardized), standardized["subject_id"].nunique())
    missing_summary = summarize_missingness(standardized)
    top_missing = {key: value for key, value in missing_summary.items() if value > 0}
    if top_missing:
        LOGGER.warning("Columns with missing values: %s", top_missing)

    return standardized
