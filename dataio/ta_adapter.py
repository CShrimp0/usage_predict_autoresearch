"""Dataset adapter for the TA healthy ultrasound cohort."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("usage_predict_feature_engineering")


def load_ta_healthy_dataframe(config: dict) -> pd.DataFrame:
    """Parse the TA Excel sheet and expand it to one row per image."""
    data_config = config["data"]
    metadata_path = Path(data_config["metadata_path"])
    image_root = Path(data_config["image_root"])
    mask_root_value = data_config.get("mask_root")
    mask_root = Path(mask_root_value) if mask_root_value not in (None, "", "null", "None") else None

    if not metadata_path.exists():
        raise FileNotFoundError(f"TA metadata file not found: {metadata_path}")
    if not image_root.exists():
        raise FileNotFoundError(f"TA image directory not found: {image_root}")
    if mask_root is not None and not mask_root.exists():
        raise FileNotFoundError(f"TA mask directory not found: {mask_root}")

    raw = pd.read_excel(metadata_path, header=None)
    healthy = raw.iloc[2:, 0:5].copy()
    healthy.columns = ["number", "age", "height_cm", "weight_kg", "sex"]
    healthy = healthy.dropna(subset=["number"])
    healthy["number"] = pd.to_numeric(healthy["number"], errors="coerce")
    healthy["age"] = pd.to_numeric(healthy["age"], errors="coerce")
    healthy["height_cm"] = pd.to_numeric(healthy["height_cm"], errors="coerce")
    healthy["weight_kg"] = pd.to_numeric(healthy["weight_kg"], errors="coerce")
    healthy["sex"] = healthy["sex"].astype(str).str.strip()
    healthy = healthy.dropna(subset=["number", "age"]).copy()
    healthy["number"] = healthy["number"].astype(int)
    healthy["subject_id"] = healthy["number"].astype(str)
    healthy["height_m"] = healthy["height_cm"] / 100.0
    healthy["bmi"] = healthy["weight_kg"] / (healthy["height_m"] ** 2)

    rows: list[dict] = []
    missing_subjects: list[int] = []
    for _, subject in healthy.iterrows():
        subject_number = int(subject["number"])
        matched_images = sorted(image_root.glob(f"anon_{subject_number}_*.png"))
        if not matched_images:
            missing_subjects.append(subject_number)
            continue

        for image_path in matched_images:
            mask_path = None
            if mask_root is not None:
                candidate = mask_root / image_path.name
                if candidate.exists():
                    mask_path = str(candidate.resolve())
            rows.append(
                {
                    "sample_id": image_path.stem,
                    "subject_id": subject["subject_id"],
                    "image_path": str(image_path.resolve()),
                    "age": float(subject["age"]),
                    "sex": subject["sex"],
                    "height_cm": float(subject["height_cm"]) if pd.notna(subject["height_cm"]) else None,
                    "weight_kg": float(subject["weight_kg"]) if pd.notna(subject["weight_kg"]) else None,
                    "bmi": float(subject["bmi"]) if pd.notna(subject["bmi"]) else None,
                    "ta_number": subject_number,
                    "cohort": "healthy",
                    "split": None,
                    "mask_path": mask_path,
                    "roi_path": mask_path,
                }
            )

    if missing_subjects:
        LOGGER.warning(
            "TA adapter skipped %d healthy subjects with no matching images. Examples: %s",
            len(missing_subjects),
            missing_subjects[:10],
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("TA adapter produced an empty dataframe.")

    LOGGER.info(
        "TA healthy adapter built %d image rows from %d subjects.",
        len(frame),
        frame["subject_id"].nunique(),
    )
    return frame
