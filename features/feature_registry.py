"""Factory for feature extractor suites."""

from __future__ import annotations

from typing import Any

from features.base import CompositeFeatureExtractor
from features.intensity_features import IntensityStatisticsExtractor, SharpnessContrastExtractor
from features.morphologic_features import MorphologicFeatureExtractor
from features.radiomics_features import GlobalRadiomicsStyleExtractor, PyRadiomicsROIExtractor
from features.texture_features import TextureFeatureExtractor


def build_feature_extractor(config: dict[str, Any], scope: str) -> CompositeFeatureExtractor:
    """Create a composite feature extractor from configuration."""
    feature_config = config["feature_extraction"]
    enabled_groups = feature_config.get("enabled_groups", ["intensity", "sharpness", "texture"])
    extractors = []

    for group in enabled_groups:
        if group == "intensity":
            extractors.append(IntensityStatisticsExtractor(feature_config.get("intensity", {}), scope=scope))
        elif group == "sharpness":
            extractors.append(SharpnessContrastExtractor(feature_config.get("sharpness", {}), scope=scope))
        elif group == "texture":
            extractors.append(TextureFeatureExtractor(feature_config.get("texture", {}), scope=scope))
        elif group == "morphology":
            extractors.append(MorphologicFeatureExtractor(feature_config.get("morphology", {}), scope=scope))
        elif group == "radiomics":
            radiomics_config = feature_config.get("radiomics", {})
            mode = radiomics_config.get("mode", "global_handcrafted")
            if mode == "pyradiomics_roi":
                extractors.append(PyRadiomicsROIExtractor(radiomics_config, scope=scope))
            else:
                extractors.append(GlobalRadiomicsStyleExtractor(radiomics_config, scope=scope))
        else:
            raise KeyError(f"Unknown feature group: {group}")

    return CompositeFeatureExtractor(extractors)
