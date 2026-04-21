#!/usr/bin/env python3
"""Run nested subject-level cross-validation for age regression."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

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
from preprocessing.split import iter_outer_cv
from _common import base_argument_parser, initialize_run, load_runtime_config
from selection.stability import aggregate_importances, summarize_selected_features
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
    parser = base_argument_parser("Run nested group cross-validation.")
    args = parser.parse_args()

    config = load_runtime_config(args.config, args.override)
    set_random_seed(int(config.get("seed", 42)))
    run_dir, logger = initialize_run(config)

    feature_df = _load_or_extract_features(config, logger, run_dir)
    raw_feature_columns = sorted([column for column in feature_df.columns if "__" in column])
    column_spec = build_column_spec(feature_df, config)

    predictions_by_fold = []
    fold_metrics_rows = []
    selected_feature_lists = []
    importance_frames = []
    shap_frames = []
    best_params_by_fold = {}
    last_pipeline = None

    for fold_idx, train_idx, test_idx in iter_outer_cv(feature_df, config):
        logger.info("Running outer fold %d.", fold_idx)
        train_df = feature_df.iloc[train_idx].reset_index(drop=True)
        test_df = feature_df.iloc[test_idx].reset_index(drop=True)

        estimator = build_pipeline_and_search(column_spec, config)
        with threading_backend_context(config.get("search", {}).get("n_jobs")):
            estimator.fit(
                train_df[column_spec.model_input_columns],
                train_df[column_spec.target_column],
                groups=train_df[column_spec.group_column],
            )
        pipeline = unwrap_best_estimator(estimator)
        last_pipeline = pipeline
        best_params_by_fold[f"fold_{fold_idx}"] = extract_best_params(estimator)

        preds = pipeline.predict(test_df[column_spec.model_input_columns])
        fold_frame = test_df[
            [column for column in ["sample_id", "subject_id", "age"] + column_spec.categorical_inputs + column_spec.numeric_metadata if column in test_df.columns]
        ].copy()
        fold_frame["prediction"] = preds
        fold_frame["fold"] = fold_idx
        predictions_by_fold.append(fold_frame)

        fold_metrics = compute_regression_metrics(fold_frame["age"], fold_frame["prediction"])
        fold_metrics["fold"] = fold_idx
        fold_metrics_rows.append(fold_metrics)

        selected_features = extract_selected_feature_names(pipeline, column_spec.model_input_columns)
        selected_feature_lists.append(selected_features)

        fold_importance = compute_importance(
            pipeline,
            x_eval=test_df[column_spec.model_input_columns],
            y_eval=test_df[column_spec.target_column],
            input_columns=column_spec.model_input_columns,
            config=config,
        )
        if not fold_importance.empty:
            fold_importance["fold"] = fold_idx
            importance_frames.append(fold_importance)

        if config.get("explain", {}).get("shap", {}).get("enabled", False):
            from utils.feature_names import transform_features

            transformed = transform_features(pipeline, test_df[column_spec.model_input_columns])
            shap_frame = compute_shap_importance(pipeline.named_steps["model"], transformed, selected_features)
            if not shap_frame.empty:
                shap_frame["fold"] = fold_idx
                shap_frames.append(shap_frame)

        models_dir = run_dir / "models"
        models_dir.mkdir(exist_ok=True)
        joblib.dump(pipeline, models_dir / f"model_fold_{fold_idx}.joblib")

    predictions = pd.concat(predictions_by_fold, ignore_index=True)
    pooled_metrics = compute_regression_metrics(predictions["age"], predictions["prediction"])
    diagnostics = residual_summary(predictions["age"], predictions["prediction"])
    fold_metrics_df = pd.DataFrame(fold_metrics_rows)
    age_bin_metrics = analyze_age_bins(
        predictions,
        bin_edges=config.get("evaluation", {}).get("age_bin_edges"),
        n_bins=int(config.get("evaluation", {}).get("age_bin_count", 5)),
    )
    subgroup_metrics = analyze_subgroups(predictions, config.get("evaluation", {}).get("subgroup_columns", []))

    stability_df = summarize_selected_features(selected_feature_lists)
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    stability_df.to_csv(tables_dir / "selection_stability.csv", index=False)
    feature_importance = aggregate_importances(importance_frames)
    if not feature_importance.empty:
        feature_importance["importance"] = feature_importance["mean_importance"]
    feature_importance.to_csv(tables_dir / "feature_importance.csv", index=False)
    fold_metrics_df.to_csv(tables_dir / "fold_metrics.csv", index=False)
    aggregated_shap = aggregate_importances(shap_frames)
    if not aggregated_shap.empty:
        aggregated_shap["importance"] = aggregated_shap["mean_importance"]

    split_info = {
        "strategy": "nested_group_cv",
        "outer_folds": int(config["split"].get("n_splits", 5)),
        "inner_folds": int(config.get("search", {}).get("cv_n_splits", 5)),
        "best_params_by_fold": best_params_by_fold,
        "model_name": config["model"]["name"],
    }
    metrics = {
        "outer_cv_pooled": pooled_metrics,
        "diagnostics": diagnostics,
        "fold_metrics_mean": fold_metrics_df.drop(columns=["fold"]).mean(numeric_only=True).to_dict(),
        "fold_metrics_std": fold_metrics_df.drop(columns=["fold"]).std(numeric_only=True).to_dict(),
    }
    save_core_outputs(
        output_dir=run_dir,
        metrics=metrics,
        predictions=predictions,
        feature_importance=feature_importance,
        selected_features=stability_df["feature"].tolist(),
        split_info=split_info,
        raw_feature_columns=raw_feature_columns,
        age_bin_metrics=age_bin_metrics,
        subgroup_metrics=subgroup_metrics,
    )
    stability_df.to_csv(run_dir / "selected_features.csv", index=False)

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    save_predicted_vs_true(predictions, figures_dir / "predicted_vs_true.png")
    save_bland_altman(predictions, figures_dir / "bland_altman.png")
    save_residual_hist(predictions, figures_dir / "residual_hist.png")
    save_residual_vs_age(predictions, figures_dir / "residual_vs_age.png")
    save_age_bin_error(age_bin_metrics, figures_dir / "age_bin_error.png")
    save_feature_importance_top20(feature_importance, figures_dir / "feature_importance_top20.png")
    if not aggregated_shap.empty:
        shap_dir = run_dir / "shap"
        shap_dir.mkdir(exist_ok=True)
        top_k = int(config.get("explain", {}).get("shap", {}).get("top_k", 20))
        shap_top = save_shap_topk(aggregated_shap, shap_dir, top_k=top_k)
        save_ranked_bar_plot(
            shap_top,
            shap_dir / f"shap_top{top_k}.png",
            title=f"Top {top_k} SHAP Importance",
            top_k=len(shap_top),
        )
        per_fold_top = []
        for frame in shap_frames:
            top_frame = frame.sort_values("importance", ascending=False).head(top_k).copy()
            per_fold_top.append(top_frame)
        if per_fold_top:
            pd.concat(per_fold_top, ignore_index=True).to_csv(shap_dir / f"shap_top{top_k}_by_fold.csv", index=False)
        aggregated_shap.to_csv(shap_dir / "shap_importance_all.csv", index=False)
    else:
        shap_top = pd.DataFrame()

    summary = build_run_summary(
        run_dir=run_dir,
        config=config,
        mode="nested_cv",
        primary_metrics=pooled_metrics,
        split_info=split_info,
        diagnostics=diagnostics,
        predictions=predictions,
        raw_feature_columns=raw_feature_columns,
        selected_features=stability_df["feature"].tolist(),
        feature_importance=feature_importance,
        age_bin_metrics=age_bin_metrics,
        subgroup_metrics=subgroup_metrics,
        fold_metrics_df=fold_metrics_df,
        shap_top=shap_top,
    )
    save_run_summary(summary, run_dir / "run_summary.md")
    append_run_log(
        run_dir=run_dir,
        config=config,
        mode="nested_cv",
        primary_metrics=pooled_metrics,
        raw_feature_count=len(raw_feature_columns),
        selected_feature_count=len(stability_df["feature"].tolist()),
        fold_summary={
            "mean": metrics["fold_metrics_mean"],
            "std": metrics["fold_metrics_std"],
        },
    )
    logger.info("Nested CV finished. Outputs written to %s", run_dir)


if __name__ == "__main__":
    main()
