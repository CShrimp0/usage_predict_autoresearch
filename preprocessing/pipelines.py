"""Pipeline construction for leakage-safe regression experiments."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from models.registry import build_model
from preprocessing.column_builder import ColumnSpec
from preprocessing.transformers import CorrelationFilter, DenseTransformer, NamedPCA, NamedVarianceThreshold
from selection.embedded_methods import build_embedded_selector
from selection.filter_methods import build_filter_selector


def _build_numeric_pipeline(config: dict[str, Any]) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy=config["preprocessing"].get("numeric_imputer", "median")))]
    scaler = config["preprocessing"].get("scaler", "standard")
    if scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler == "robust":
        steps.append(("scaler", RobustScaler()))
    elif scaler in {"none", None, False}:
        pass
    else:
        raise ValueError(f"Unsupported scaler: {scaler}")
    return Pipeline(steps)


def _build_categorical_pipeline(config: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy=config["preprocessing"].get("categorical_imputer", "most_frequent"))),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )


def build_preprocessor(column_spec: ColumnSpec, config: dict[str, Any]) -> ColumnTransformer:
    """Build the column transformer for image features and metadata."""
    transformers = []
    if column_spec.numeric_inputs:
        transformers.append(("numeric", _build_numeric_pipeline(config), column_spec.numeric_inputs))
    if column_spec.categorical_inputs:
        transformers.append(("categorical", _build_categorical_pipeline(config), column_spec.categorical_inputs))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def build_selector(config: dict[str, Any], random_state: int):
    """Build an optional feature selection step."""
    selection_config = config.get("selection", {})
    if not selection_config.get("enabled", False):
        return "passthrough"
    method = selection_config.get("method", "univariate_k_best")
    if method in {"univariate_k_best", "univariate_percentile"}:
        return build_filter_selector(selection_config)
    if method in {"lasso", "elasticnet", "tree"}:
        return build_embedded_selector(selection_config, random_state=random_state)
    raise ValueError(f"Unknown feature selection method: {method}")


def build_regression_pipeline(column_spec: ColumnSpec, config: dict[str, Any], random_state: int):
    """Create the full sklearn pipeline and the base model metadata."""
    preprocessing_config = config["preprocessing"]
    steps = [
        ("preprocessor", build_preprocessor(column_spec, config)),
        ("to_dense", DenseTransformer()),
    ]

    low_variance = preprocessing_config.get("low_variance", {}) or {}
    if low_variance.get("enabled", True):
        steps.append(("low_variance", NamedVarianceThreshold(threshold=float(low_variance.get("threshold", 0.0)))))

    correlation_config = preprocessing_config.get("correlation_filter", {}) or {}
    if correlation_config.get("enabled", True):
        steps.append(("correlation_filter", CorrelationFilter(threshold=float(correlation_config.get("threshold", 0.95)))))

    selector = build_selector(config, random_state=random_state)
    if selector != "passthrough":
        steps.append(("selector", selector))

    pca_config = preprocessing_config.get("pca", {}) or {}
    if pca_config.get("enabled", False):
        steps.append(("pca", NamedPCA(n_components=pca_config.get("n_components", 0.95), random_state=random_state)))

    model = build_model(config["model"]["name"], config["model"].get("params", {}), random_state=random_state)
    steps.append(("model", model))
    pipeline = Pipeline(steps=steps)
    return pipeline
