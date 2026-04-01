"""Build model input column groups from the feature table."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from dataio.build_dataframe import CORE_COLUMNS


@dataclass
class ColumnSpec:
    """Named column groups for the modeling pipeline."""

    image_features: list[str]
    numeric_metadata: list[str]
    categorical_metadata: list[str]
    boolean_metadata: list[str]
    target_column: str
    group_column: str

    @property
    def categorical_inputs(self) -> list[str]:
        return [*self.categorical_metadata, *self.boolean_metadata]

    @property
    def numeric_inputs(self) -> list[str]:
        return [*self.image_features, *self.numeric_metadata]

    @property
    def model_input_columns(self) -> list[str]:
        return [*self.numeric_inputs, *self.categorical_inputs]


def build_column_spec(df: pd.DataFrame, config: dict) -> ColumnSpec:
    """Infer model input columns from configuration and dataframe columns."""
    data_config = config["data"]
    target_column = "age"
    group_column = "subject_id"

    reserved = set(CORE_COLUMNS)
    image_features = sorted([column for column in df.columns if "__" in column and column not in reserved])

    metadata_config = data_config.get("model_metadata_columns", {}) or {}
    numeric_metadata = [col for col in metadata_config.get("numeric", []) if col in df.columns]
    categorical_metadata = [col for col in metadata_config.get("categorical", []) if col in df.columns]
    boolean_metadata = [col for col in metadata_config.get("boolean", []) if col in df.columns]

    feature_mode = data_config.get("feature_mode", "image_only")
    if feature_mode == "image_only":
        numeric_metadata = []
        categorical_metadata = []
        boolean_metadata = []
    elif feature_mode == "metadata_only":
        image_features = []
    elif feature_mode != "image_plus_metadata":
        raise ValueError(f"Unknown feature mode: {feature_mode}")

    if not image_features and feature_mode != "metadata_only":
        raise ValueError("No image features detected. Run feature extraction first or switch to metadata_only mode.")
    if not image_features and not any([numeric_metadata, categorical_metadata, boolean_metadata]):
        raise ValueError("No model input columns available after applying feature_mode.")

    return ColumnSpec(
        image_features=image_features,
        numeric_metadata=numeric_metadata,
        categorical_metadata=categorical_metadata,
        boolean_metadata=boolean_metadata,
        target_column=target_column,
        group_column=group_column,
    )
