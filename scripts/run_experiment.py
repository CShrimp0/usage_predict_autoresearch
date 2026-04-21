#!/usr/bin/env python3
"""Run a hold-out white-box regression experiment."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone

import _bootstrap  # noqa: F401
from evaluation.metrics import compute_regression_metrics, residual_summary
from evaluation.plots import (
    save_age_bin_error,
    save_bland_altman,
    save_feature_importance_top20,
    save_predicted_vs_true,
    save_ranked_bar_plot,
    save_residual_hist,
    save_residual_vs_age,
)
from evaluation.reporting import build_run_summary, save_core_outputs, save_run_summary
from evaluation.subgroup_analysis import analyze_age_bins, analyze_subgroups
from explain.importance import compute_importance, extract_selected_feature_names
from explain.shap_utils import compute_shap_importance, save_shap_topk
from preprocessing.column_builder import build_column_spec
from preprocessing.split import assign_holdout_split
from _common import base_argument_parser, initialize_run, load_runtime_config
from utils.feature_extraction import extract_feature_table
from utils.modeling import build_pipeline_and_search, extract_best_params, unwrap_best_estimator
from utils.parallel import threading_backend_context
from utils.run_log import append_run_log
from utils.seeds import set_random_seed


def _load_or_extract_features(config: dict, logger, run_dir: Path) -> pd.DataFrame:
    cache_path = config.get("paths", {}).get("feature_table")
    if cache_path:
        logger.info("Loading cached feature table from %s", cache_path)
        return pd.read_csv(cache_path)
    feature_df = extract_feature_table(config)
    if config.get("reporting", {}).get("save_feature_table", True):
        tables_dir = run_dir / "tables"
        tables_dir.mkdir(exist_ok=True)
        feature_df.to_csv(tables_dir / "features_raw.csv", index=False)
    return feature_df


def main() -> None:
    parser = base_argument_parser("Run a hold-out classical ML age regression experiment.")
    args = parser.parse_args()

    config = load_runtime_config(args.config, args.override)
    if config.get("split", {}).get("strategy", "holdout") not in {"holdout", "predefined"}:
        raise ValueError("run_experiment.py currently supports split.strategy=holdout or predefined. Use run_nested_cv.py for CV workflows.")

    set_random_seed(int(config.get("seed", 42)))
    run_dir, logger = initialize_run(config)
    logger.info("Starting experiment '%s'.", config.get("experiment", {}).get("name", "experiment"))

    feature_df = _load_or_extract_features(config, logger, run_dir)
    raw_feature_columns = sorted([column for column in feature_df.columns if "__" in column])
    column_spec = build_column_spec(feature_df, config)
    split = assign_holdout_split(feature_df, config)
    feature_df = feature_df.copy()
    feature_df["split"] = split.sample_split_series

    train_df = feature_df.iloc[split.train_idx].reset_index(drop=True)
    val_df = feature_df.iloc[split.val_idx].reset_index(drop=True)
    test_df = feature_df.iloc[split.test_idx].reset_index(drop=True)
    logger.info(
        "Hold-out split: %d train / %d val / %d test samples.",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    estimator = build_pipeline_and_search(column_spec, config)
    x_train = train_df[column_spec.model_input_columns]
    y_train = train_df[column_spec.target_column]
    groups_train = train_df[column_spec.group_column]
    with threading_backend_context(config.get("search", {}).get("n_jobs")):
        estimator.fit(x_train, y_train, groups=groups_train)

    tuned_pipeline = unwrap_best_estimator(estimator)
    best_params = extract_best_params(estimator)

    val_predictions = tuned_pipeline.predict(val_df[column_spec.model_input_columns])
    val_frame = val_df[
        [column for column in ["sample_id", "subject_id", "age", "split"] + column_spec.categorical_inputs + column_spec.numeric_metadata if column in val_df.columns]
    ].copy()
    val_frame["prediction"] = val_predictions
    val_metrics = compute_regression_metrics(val_frame["age"], val_frame["prediction"])

    final_pipeline = clone(tuned_pipeline)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    with threading_backend_context(config.get("search", {}).get("n_jobs")):
        final_pipeline.fit(train_val_df[column_spec.model_input_columns], train_val_df[column_spec.target_column])
    test_predictions = final_pipeline.predict(test_df[column_spec.model_input_columns])

    prediction_columns = [
        column
        for column in ["sample_id", "subject_id", "age", "split"] + column_spec.categorical_inputs + column_spec.numeric_metadata
        if column in test_df.columns
    ]
    test_frame = test_df[prediction_columns].copy()
    test_frame["prediction"] = test_predictions
    predictions = pd.concat([val_frame, test_frame], ignore_index=True)

    test_metrics = compute_regression_metrics(test_frame["age"], test_frame["prediction"])
    diagnostics = residual_summary(test_frame["age"], test_frame["prediction"])
    metrics = {
        "validation": val_metrics,
        "test": test_metrics,
        "diagnostics": diagnostics,
    }

    feature_importance = compute_importance(
        final_pipeline,
        x_eval=test_df[column_spec.model_input_columns],
        y_eval=test_df[column_spec.target_column],
        input_columns=column_spec.model_input_columns,
        config=config,
    )
    selected_features = extract_selected_feature_names(final_pipeline, column_spec.model_input_columns)
    shap_frame = pd.DataFrame()
    if config.get("explain", {}).get("shap", {}).get("enabled", False):
        from utils.feature_names import transform_features

        transformed = transform_features(final_pipeline, test_df[column_spec.model_input_columns])
        shap_frame = compute_shap_importance(final_pipeline.named_steps["model"], transformed, selected_features)
    age_bin_metrics = analyze_age_bins(
        test_frame,
        bin_edges=config.get("evaluation", {}).get("age_bin_edges"),
        n_bins=int(config.get("evaluation", {}).get("age_bin_count", 5)),
    )
    subgroup_metrics = analyze_subgroups(test_frame, config.get("evaluation", {}).get("subgroup_columns", []))

    split_info = {
        **split.info,
        "best_params": best_params,
        "model_name": config["model"]["name"],
        "feature_mode": config["data"].get("feature_mode", "image_only"),
    }
    save_core_outputs(
        output_dir=run_dir,
        metrics=metrics,
        predictions=predictions,
        feature_importance=feature_importance,
        selected_features=selected_features,
        split_info=split_info,
        raw_feature_columns=raw_feature_columns,
        age_bin_metrics=age_bin_metrics,
        subgroup_metrics=subgroup_metrics,
    )

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    save_predicted_vs_true(test_frame, figures_dir / "predicted_vs_true.png")
    save_bland_altman(test_frame, figures_dir / "bland_altman.png")
    save_residual_hist(test_frame, figures_dir / "residual_hist.png")
    save_residual_vs_age(test_frame, figures_dir / "residual_vs_age.png")
    save_age_bin_error(age_bin_metrics, figures_dir / "age_bin_error.png")
    save_feature_importance_top20(
        feature_importance[feature_importance["importance_type"].isin(["permutation", "impurity", "coefficient_abs"])].drop_duplicates("feature"),
        figures_dir / "feature_importance_top20.png",
    )
    shap_top = pd.DataFrame()
    if not shap_frame.empty:
        shap_dir = run_dir / "shap"
        shap_dir.mkdir(exist_ok=True)
        top_k = int(config.get("explain", {}).get("shap", {}).get("top_k", 20))
        shap_frame.to_csv(shap_dir / "shap_importance_all.csv", index=False)
        shap_top = save_shap_topk(shap_frame, shap_dir, top_k=top_k)
        save_ranked_bar_plot(
            shap_top,
            shap_dir / f"shap_top{top_k}.png",
            title=f"Top {top_k} SHAP Importance",
            top_k=len(shap_top),
        )

    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(final_pipeline, models_dir / "model.joblib")
    summary = build_run_summary(
        run_dir=run_dir,
        config=config,
        mode="holdout",
        primary_metrics=test_metrics,
        split_info=split_info,
        diagnostics=diagnostics,
        predictions=test_frame,
        raw_feature_columns=raw_feature_columns,
        selected_features=selected_features,
        feature_importance=feature_importance.sort_values("importance", ascending=False),
        age_bin_metrics=age_bin_metrics,
        subgroup_metrics=subgroup_metrics,
        validation_metrics=val_metrics,
        shap_top=shap_top,
    )
    save_run_summary(summary, run_dir / "run_summary.md")
    append_run_log(
        run_dir=run_dir,
        config=config,
        mode="holdout",
        primary_metrics=test_metrics,
        raw_feature_count=len(raw_feature_columns),
        selected_feature_count=len(selected_features),
        validation_metrics=val_metrics,
    )
    logger.info("Experiment finished. Outputs written to %s", run_dir)


if __name__ == "__main__":
    main()
