"""Optional PyRadiomics integration plus global handcrafted fallback."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from features.base import BaseFeatureExtractor, CompositeFeatureExtractor, FeatureContext
from features.intensity_features import IntensityStatisticsExtractor
from features.texture_features import TextureFeatureExtractor

LOGGER = logging.getLogger("usage_predict_feature_engineering")


class GlobalRadiomicsStyleExtractor(BaseFeatureExtractor):
    """Radiomics-style fallback built from handcrafted global descriptors."""

    group_name = "radiomics_style"

    def __init__(self, config: dict[str, Any], scope: str) -> None:
        super().__init__(config, scope)
        self.composite = CompositeFeatureExtractor(
            [
                IntensityStatisticsExtractor(config.get("intensity", {}), scope=scope),
                TextureFeatureExtractor(config.get("texture", {}), scope=scope),
            ]
        )

    def feature_names(self) -> list[str]:
        return [
            name.replace(f"__intensity__", f"__{self.group_name}__intensity__").replace(
                f"__texture__", f"__{self.group_name}__texture__"
            )
            for name in self.composite.feature_names()
        ]

    def _extract_impl(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        context: FeatureContext,
    ) -> dict[str, float]:
        features = self.composite.extract(image=image, mask=mask, context=context)
        remapped: dict[str, float] = {}
        for key, value in features.items():
            remapped[key.replace(f"__intensity__", f"__{self.group_name}__intensity__").replace(
                f"__texture__", f"__{self.group_name}__texture__"
            )] = value
        return remapped


class PyRadiomicsROIExtractor(BaseFeatureExtractor):
    """Optional PyRadiomics-based ROI extractor."""

    group_name = "pyradiomics"
    requires_mask = True

    def __init__(self, config: dict[str, Any], scope: str) -> None:
        super().__init__(config, scope)
        self.enabled = False
        self._extractor = None
        self._feature_names: list[str] = []
        try:
            from radiomics import featureextractor
        except ImportError:
            LOGGER.warning("PyRadiomics is not installed; pyradiomics extractor will emit NaN features.")
            return

        settings = config.get("settings", {}) or {}
        self._extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        enabled_classes = config.get("feature_classes", ["firstorder", "shape2D", "glcm", "glrlm", "glszm", "ngtdm"])
        self._extractor.disableAllFeatures()
        for feature_class in enabled_classes:
            self._extractor.enableFeatureClassByName(feature_class)
        self.enabled = True
        # Names are not known until extraction. Use a conservative fixed subset for stable schema.
        self._feature_names = [
            self.prefix(f"feature_{name}")
            for name in [
                "firstorder_Mean",
                "firstorder_Entropy",
                "firstorder_Skewness",
                "shape2D_Perimeter",
                "shape2D_PixelSurface",
                "glcm_Contrast",
                "glcm_Correlation",
                "glrlm_ShortRunEmphasis",
                "glszm_SmallAreaEmphasis",
                "ngtdm_Coarseness",
            ]
        ]

    def feature_names(self) -> list[str]:
        return self._feature_names

    def _extract_impl(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        context: FeatureContext,
    ) -> dict[str, float]:
        if not self.enabled or self._extractor is None or not context.mask_path:
            return {name: np.nan for name in self.feature_names()}

        try:
            import SimpleITK as sitk
        except ImportError:
            LOGGER.warning("SimpleITK is not installed; pyradiomics extractor will emit NaN features.")
            return {name: np.nan for name in self.feature_names()}

        image_sitk = sitk.ReadImage(context.image_path)
        mask_sitk = sitk.ReadImage(context.mask_path)
        raw = self._extractor.execute(image_sitk, mask_sitk)
        output = {name: np.nan for name in self.feature_names()}
        mapping = {
            "original_firstorder_Mean": self.prefix("feature_firstorder_Mean"),
            "original_firstorder_Entropy": self.prefix("feature_firstorder_Entropy"),
            "original_firstorder_Skewness": self.prefix("feature_firstorder_Skewness"),
            "original_shape2D_Perimeter": self.prefix("feature_shape2D_Perimeter"),
            "original_shape2D_PixelSurface": self.prefix("feature_shape2D_PixelSurface"),
            "original_glcm_Contrast": self.prefix("feature_glcm_Contrast"),
            "original_glcm_Correlation": self.prefix("feature_glcm_Correlation"),
            "original_glrlm_ShortRunEmphasis": self.prefix("feature_glrlm_ShortRunEmphasis"),
            "original_glszm_SmallAreaEmphasis": self.prefix("feature_glszm_SmallAreaEmphasis"),
            "original_ngtdm_Coarseness": self.prefix("feature_ngtdm_Coarseness"),
        }
        for source_name, target_name in mapping.items():
            if source_name in raw:
                output[target_name] = float(raw[source_name])
        return output
