"""Shared feature extraction workflow."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from dataio.build_dataframe import build_dataframe
from dataio.load_images import load_grayscale_image, load_mask, maybe_resize
from features.base import FeatureContext
from features.feature_registry import build_feature_extractor

LOGGER = logging.getLogger("usage_predict_feature_engineering")


def preprocess_image(image: np.ndarray, config: dict) -> np.ndarray:
    """Apply deterministic per-image preprocessing before feature extraction."""
    pre_cfg = config.get("image", {}).get("preprocessing", {}) or {}
    clip_percentiles = pre_cfg.get("clip_percentiles")
    if clip_percentiles:
        low, high = np.percentile(image, clip_percentiles)
        image = np.clip(image, low, high)
    normalize = pre_cfg.get("normalize", "minmax")
    if normalize == "minmax":
        denom = max(float(image.max() - image.min()), 1e-12)
        image = (image - image.min()) / denom
    elif normalize == "zscore":
        denom = max(float(image.std()), 1e-12)
        image = (image - image.mean()) / denom
    elif normalize in {"none", None, False}:
        pass
    else:
        raise ValueError(f"Unsupported image normalization mode: {normalize}")
    return image.astype(np.float32)


def extract_feature_table(config: dict, dataframe: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the canonical dataframe and append handcrafted features."""
    df = dataframe.copy() if dataframe is not None else build_dataframe(config)
    resize = maybe_resize(config.get("image", {}).get("preprocessing", {}))
    scopes = config["feature_extraction"].get("scopes", ["whole_image"])
    extractors = {scope: build_feature_extractor(config, scope=scope) for scope in scopes}

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        image = load_grayscale_image(row["image_path"], resize=resize)
        image = preprocess_image(image, config)
        mask = None
        mask_path = row.get("mask_path")
        if isinstance(mask_path, str) and mask_path.strip():
            mask_file = Path(mask_path)
            if mask_file.exists():
                mask = load_mask(mask_file, resize=resize)
            else:
                LOGGER.warning("Mask file not found for sample %s: %s", row["sample_id"], mask_file)

        combined = row.to_dict()
        for scope, extractor in extractors.items():
            current_mask = mask if scope == "roi" else None
            context = FeatureContext(
                scope=scope,
                sample_id=str(row["sample_id"]),
                subject_id=str(row["subject_id"]),
                image_path=str(row["image_path"]),
                mask_path=str(row.get("mask_path")) if pd.notna(row.get("mask_path")) else None,
            )
            combined.update(extractor.extract(image=image, mask=current_mask, context=context))
        rows.append(combined)

    feature_df = pd.DataFrame(rows)
    feature_columns = sorted([column for column in feature_df.columns if "__" in column])
    ordered_columns = [column for column in feature_df.columns if column not in feature_columns] + feature_columns
    return feature_df[ordered_columns]
