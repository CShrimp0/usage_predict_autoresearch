"""Group-aware train/validation/test splitting and CV utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, GroupKFold, StratifiedGroupKFold, train_test_split


def _bin_regression_target(
    y: np.ndarray,
    strategy: str = "quantile",
    n_bins: int = 5,
    explicit_bins: list[float] | None = None,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if explicit_bins:
        bins = np.asarray(explicit_bins, dtype=float)
        return np.digitize(y, bins=bins, right=False)

    if strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, num=n_bins + 1)
        bins = np.quantile(y, quantiles)
        bins = np.unique(bins)
        if len(bins) <= 2:
            return np.zeros_like(y, dtype=int)
        return np.digitize(y, bins=bins[1:-1], right=False)
    if strategy == "uniform":
        bins = np.linspace(y.min(), y.max(), num=n_bins + 1)
        return np.digitize(y, bins=bins[1:-1], right=False)

    raise ValueError(f"Unknown age binning strategy: {strategy}")


class RegressionStratifiedGroupKFold(BaseCrossValidator):
    """Stratify regression targets by bins while respecting groups."""

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int | None = None,
        bin_strategy: str = "quantile",
        n_bins: int = 5,
        explicit_bins: list[float] | None = None,
    ) -> None:
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.bin_strategy = bin_strategy
        self.n_bins = n_bins
        self.explicit_bins = explicit_bins

    def split(self, x, y=None, groups=None):
        if y is None or groups is None:
            raise ValueError("RegressionStratifiedGroupKFold requires both y and groups.")
        y_binned = _bin_regression_target(
            y=np.asarray(y),
            strategy=self.bin_strategy,
            n_bins=self.n_bins,
            explicit_bins=self.explicit_bins,
        )
        splitter = StratifiedGroupKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )
        yield from splitter.split(X=x, y=y_binned, groups=groups)

    def get_n_splits(self, x=None, y=None, groups=None):
        return self.n_splits


def make_group_cv_splitter(config: dict, n_splits: int | None = None) -> BaseCrossValidator:
    """Create a group-aware CV splitter from config."""
    split_config = config["split"]
    n_folds = n_splits or int(split_config.get("n_splits", 5))
    stratify = bool(split_config.get("stratify_by_age", True))
    if stratify:
        return RegressionStratifiedGroupKFold(
            n_splits=n_folds,
            shuffle=bool(split_config.get("shuffle", True)),
            random_state=int(split_config.get("random_state", 42)),
            bin_strategy=split_config.get("age_bin_method", "quantile"),
            n_bins=int(split_config.get("age_bins", 5)),
            explicit_bins=split_config.get("age_bin_edges"),
        )
    return GroupKFold(n_splits=n_folds)


@dataclass
class HoldoutSplit:
    """Subject-level hold-out split indices and summary."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    sample_split_series: pd.Series
    info: dict


def _subject_level_frame(df: pd.DataFrame, target_col: str, group_col: str) -> pd.DataFrame:
    subject_df = (
        df.groupby(group_col)
        .agg(subject_age=(target_col, "mean"))
        .reset_index()
        .rename(columns={group_col: "subject_id"})
    )
    return subject_df


def validate_predefined_subject_split(df: pd.DataFrame, split_col: str = "split") -> None:
    """Ensure predefined splits are subject-consistent."""
    grouped = df.groupby("subject_id")[split_col].nunique()
    bad_subjects = grouped[grouped > 1]
    if not bad_subjects.empty:
        raise ValueError(
            "Predefined split column is inconsistent within subject_id: "
            f"{bad_subjects.index.tolist()[:10]}"
        )


def assign_holdout_split(df: pd.DataFrame, config: dict) -> HoldoutSplit:
    """Assign subject-level train/val/test splits."""
    split_config = config["split"]
    target_col = "age"
    group_col = "subject_id"
    strategy = split_config.get("strategy", "holdout")

    if strategy == "predefined":
        if "split" not in df.columns:
            raise KeyError("Split strategy is 'predefined' but standardized 'split' column is missing.")
        validate_predefined_subject_split(df)
        split_series = df["split"].astype(str).str.lower()
        train_idx = df.index[split_series == "train"].to_numpy()
        val_idx = df.index[split_series == "val"].to_numpy()
        test_idx = df.index[split_series == "test"].to_numpy()
        return HoldoutSplit(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            sample_split_series=split_series,
            info={
                "strategy": strategy,
                "n_train_samples": int(len(train_idx)),
                "n_val_samples": int(len(val_idx)),
                "n_test_samples": int(len(test_idx)),
            },
        )

    subjects = _subject_level_frame(df, target_col=target_col, group_col=group_col)
    stratify_labels = None
    if split_config.get("stratify_by_age", True):
        stratify_labels = _bin_regression_target(
            y=subjects["subject_age"].to_numpy(),
            strategy=split_config.get("age_bin_method", "quantile"),
            n_bins=int(split_config.get("age_bins", 5)),
            explicit_bins=split_config.get("age_bin_edges"),
        )

    train_size = float(split_config.get("train_size", 0.7))
    val_size = float(split_config.get("val_size", 0.15))
    test_size = float(split_config.get("test_size", 0.15))
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    try:
        train_val_subjects, test_subjects = train_test_split(
            subjects["subject_id"].to_numpy(),
            test_size=test_size,
            random_state=int(split_config.get("random_state", 42)),
            stratify=stratify_labels,
        )
    except ValueError:
        train_val_subjects, test_subjects = train_test_split(
            subjects["subject_id"].to_numpy(),
            test_size=test_size,
            random_state=int(split_config.get("random_state", 42)),
            stratify=None,
        )

    train_val_frame = subjects[subjects["subject_id"].isin(train_val_subjects)].reset_index(drop=True)
    train_val_stratify = None
    if split_config.get("stratify_by_age", True):
        train_val_stratify = _bin_regression_target(
            y=train_val_frame["subject_age"].to_numpy(),
            strategy=split_config.get("age_bin_method", "quantile"),
            n_bins=int(split_config.get("age_bins", 5)),
            explicit_bins=split_config.get("age_bin_edges"),
        )
    adjusted_val = val_size / (train_size + val_size)
    try:
        train_subjects, val_subjects = train_test_split(
            train_val_frame["subject_id"].to_numpy(),
            test_size=adjusted_val,
            random_state=int(split_config.get("random_state", 42)),
            stratify=train_val_stratify,
        )
    except ValueError:
        train_subjects, val_subjects = train_test_split(
            train_val_frame["subject_id"].to_numpy(),
            test_size=adjusted_val,
            random_state=int(split_config.get("random_state", 42)),
            stratify=None,
        )

    split_series = pd.Series(index=df.index, dtype=object)
    split_series.loc[df["subject_id"].isin(train_subjects)] = "train"
    split_series.loc[df["subject_id"].isin(val_subjects)] = "val"
    split_series.loc[df["subject_id"].isin(test_subjects)] = "test"

    train_idx = df.index[split_series == "train"].to_numpy()
    val_idx = df.index[split_series == "val"].to_numpy()
    test_idx = df.index[split_series == "test"].to_numpy()
    info = {
        "strategy": strategy,
        "n_subjects": int(subjects.shape[0]),
        "n_train_subjects": int(len(train_subjects)),
        "n_val_subjects": int(len(val_subjects)),
        "n_test_subjects": int(len(test_subjects)),
        "n_train_samples": int(len(train_idx)),
        "n_val_samples": int(len(val_idx)),
        "n_test_samples": int(len(test_idx)),
    }
    return HoldoutSplit(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        sample_split_series=split_series,
        info=info,
    )


def iter_outer_cv(df: pd.DataFrame, config: dict) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield fold indices for outer cross-validation."""
    splitter = make_group_cv_splitter(config=config, n_splits=int(config["split"].get("n_splits", 5)))
    x_dummy = np.zeros((len(df), 1))
    y = df["age"].to_numpy()
    groups = df["subject_id"].to_numpy()
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(x_dummy, y=y, groups=groups), start=1):
        yield fold_idx, train_idx, test_idx
