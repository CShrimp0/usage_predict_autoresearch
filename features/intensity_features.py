"""Intensity, sharpness, and contrast features."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_gradient_magnitude, laplace

from dataio.load_images import apply_mask
from features.base import BaseFeatureExtractor, FeatureContext


def _histogram_entropy(values: np.ndarray, bins: int) -> float:
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0), density=False)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return np.nan
    probs = hist / total
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


class IntensityStatisticsExtractor(BaseFeatureExtractor):
    """Basic grayscale statistics."""

    group_name = "intensity"

    def __init__(self, config: dict[str, Any], scope: str) -> None:
        super().__init__(config, scope)
        self.bins = int(config.get("bins", 64))
        self.percentiles = [int(p) for p in config.get("percentiles", [5, 25, 50, 75, 95])]

    def feature_names(self) -> list[str]:
        base = [
            "mean",
            "std",
            "min",
            "max",
            "median",
            "skewness",
            "kurtosis",
            "entropy",
            "energy",
            "iqr",
            "mad",
        ]
        percentile_features = [f"p{p}" for p in self.percentiles]
        return [self.prefix(name) for name in base + percentile_features]

    def _extract_impl(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        context: FeatureContext,
    ) -> dict[str, float]:
        values = apply_mask(image, mask).astype(np.float64)
        result = {
            self.prefix("mean"): float(np.mean(values)),
            self.prefix("std"): float(np.std(values)),
            self.prefix("min"): float(np.min(values)),
            self.prefix("max"): float(np.max(values)),
            self.prefix("median"): float(np.median(values)),
            self.prefix("skewness"): float(stats.skew(values, bias=False)),
            self.prefix("kurtosis"): float(stats.kurtosis(values, fisher=True, bias=False)),
            self.prefix("entropy"): _histogram_entropy(values, self.bins),
            self.prefix("energy"): float(np.sum(np.square(values))),
            self.prefix("iqr"): float(np.percentile(values, 75) - np.percentile(values, 25)),
            self.prefix("mad"): float(np.mean(np.abs(values - np.mean(values)))),
        }
        for percentile in self.percentiles:
            result[self.prefix(f"p{percentile}")] = float(np.percentile(values, percentile))
        return result


class SharpnessContrastExtractor(BaseFeatureExtractor):
    """Sharpness, focus, and contrast descriptors."""

    group_name = "sharpness"

    def feature_names(self) -> list[str]:
        return [
            self.prefix("laplacian_variance"),
            self.prefix("tenengrad"),
            self.prefix("rms_contrast"),
            self.prefix("dynamic_range"),
            self.prefix("gradient_mean"),
            self.prefix("gradient_std"),
        ]

    def _extract_impl(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        context: FeatureContext,
    ) -> dict[str, float]:
        values = apply_mask(image, mask).astype(np.float64)
        if mask is not None:
            working = np.where(mask, image, np.median(values))
        else:
            working = image
        lap = laplace(working)
        grad = gaussian_gradient_magnitude(working, sigma=1.0)
        return {
            self.prefix("laplacian_variance"): float(np.var(lap[mask] if mask is not None else lap)),
            self.prefix("tenengrad"): float(np.mean(np.square(grad[mask] if mask is not None else grad))),
            self.prefix("rms_contrast"): float(np.sqrt(np.mean((values - np.mean(values)) ** 2))),
            self.prefix("dynamic_range"): float(np.max(values) - np.min(values)),
            self.prefix("gradient_mean"): float(np.mean(grad[mask] if mask is not None else grad)),
            self.prefix("gradient_std"): float(np.std(grad[mask] if mask is not None else grad)),
        }
