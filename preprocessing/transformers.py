"""Custom sklearn-compatible transformers with feature-name support."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold


class DenseTransformer(BaseEstimator, TransformerMixin):
    """Convert sparse matrices to dense arrays when downstream steps require it."""

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        if sparse.issparse(x):
            return x.toarray()
        return np.asarray(x)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None):
        return np.asarray(input_features, dtype=object)


class NamedVarianceThreshold(VarianceThreshold):
    """VarianceThreshold with stable feature-name propagation."""

    def get_feature_names_out(self, input_features: Sequence[str] | None = None):
        if input_features is None:
            raise ValueError("input_features must be provided for NamedVarianceThreshold.")
        features = np.asarray(input_features, dtype=object)
        return features[self.get_support()]


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop highly correlated features based on the training fold only."""

    def __init__(self, threshold: float = 0.95) -> None:
        self.threshold = threshold
        self.keep_indices_: np.ndarray | None = None
        self.input_features_: np.ndarray | None = None

    def fit(self, x, y=None):
        array = x.toarray() if sparse.issparse(x) else np.asarray(x)
        if array.ndim != 2:
            raise ValueError("CorrelationFilter expects a 2D input array.")
        n_features = array.shape[1]
        if n_features == 0:
            self.keep_indices_ = np.array([], dtype=int)
            return self
        corr = np.corrcoef(array, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        keep = np.ones(n_features, dtype=bool)
        for i in range(n_features):
            if not keep[i]:
                continue
            for j in range(i + 1, n_features):
                if keep[j] and abs(corr[i, j]) >= self.threshold:
                    keep[j] = False
        self.keep_indices_ = np.where(keep)[0]
        return self

    def transform(self, x):
        if self.keep_indices_ is None:
            raise RuntimeError("CorrelationFilter must be fitted before transform.")
        if sparse.issparse(x):
            return x[:, self.keep_indices_]
        return np.asarray(x)[:, self.keep_indices_]

    def get_feature_names_out(self, input_features: Sequence[str] | None = None):
        if self.keep_indices_ is None:
            raise RuntimeError("CorrelationFilter must be fitted before get_feature_names_out.")
        if input_features is None:
            if self.input_features_ is None:
                raise ValueError("input_features must be provided.")
            features = self.input_features_
        else:
            features = np.asarray(input_features, dtype=object)
        return features[self.keep_indices_]


class NamedPCA(PCA):
    """PCA wrapper that emits deterministic component names."""

    def get_feature_names_out(self, input_features=None):
        if not hasattr(self, "n_components_"):
            raise RuntimeError("NamedPCA must be fitted before get_feature_names_out.")
        return np.asarray([f"pca_component_{idx:03d}" for idx in range(self.n_components_)], dtype=object)
