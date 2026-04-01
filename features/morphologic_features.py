"""Morphologic and geometric features for ROI masks."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, perimeter, regionprops

from features.base import BaseFeatureExtractor, FeatureContext


class MorphologicFeatureExtractor(BaseFeatureExtractor):
    """Mask-based geometric measurements."""

    group_name = "morphology"
    requires_mask = True

    def feature_names(self) -> list[str]:
        names = [
            "area",
            "perimeter",
            "eccentricity",
            "compactness",
            "solidity",
            "major_axis_length",
            "minor_axis_length",
            "extent",
            "orientation",
            "equivalent_diameter",
            "thickness_mean",
            "thickness_median",
            "thickness_max",
        ]
        return [self.prefix(name) for name in names]

    def _extract_impl(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        context: FeatureContext,
    ) -> dict[str, float]:
        assert mask is not None
        labeled = label(mask.astype(np.uint8))
        props = regionprops(labeled)
        if not props:
            return {name: np.nan for name in self.feature_names()}
        region = max(props, key=lambda item: item.area)
        area = float(region.area)
        peri = float(perimeter(mask.astype(np.uint8), neighborhood=8))
        compactness = float((peri ** 2) / (4.0 * np.pi * max(area, 1e-12)))

        distance = distance_transform_edt(mask)
        thickness = 2.0 * distance[mask]
        return {
            self.prefix("area"): area,
            self.prefix("perimeter"): peri,
            self.prefix("eccentricity"): float(region.eccentricity),
            self.prefix("compactness"): compactness,
            self.prefix("solidity"): float(region.solidity),
            self.prefix("major_axis_length"): float(region.major_axis_length),
            self.prefix("minor_axis_length"): float(region.minor_axis_length),
            self.prefix("extent"): float(region.extent),
            self.prefix("orientation"): float(region.orientation),
            self.prefix("equivalent_diameter"): float(region.equivalent_diameter_area),
            self.prefix("thickness_mean"): float(np.mean(thickness)),
            self.prefix("thickness_median"): float(np.median(thickness)),
            self.prefix("thickness_max"): float(np.max(thickness)),
        }
