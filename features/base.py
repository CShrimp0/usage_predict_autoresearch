"""Base classes for handcrafted feature extraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FeatureContext:
    """Context passed to feature extractors."""

    scope: str
    sample_id: str
    subject_id: str
    image_path: str
    mask_path: str | None = None


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    group_name: str = "base"
    requires_mask: bool = False

    def __init__(self, config: dict[str, Any], scope: str) -> None:
        self.config = config
        self.scope = scope

    @abstractmethod
    def feature_names(self) -> list[str]:
        """Return the full list of feature names emitted by the extractor."""

    @abstractmethod
    def _extract_impl(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        context: FeatureContext,
    ) -> dict[str, float]:
        """Implement extractor logic."""

    def extract(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        context: FeatureContext,
    ) -> dict[str, float]:
        """Extract features and guarantee a complete output dictionary."""
        if self.requires_mask and mask is None:
            return {name: np.nan for name in self.feature_names()}

        extracted = self._extract_impl(image=image, mask=mask, context=context)
        output = {name: np.nan for name in self.feature_names()}
        output.update(extracted)
        return output

    def prefix(self, feature_name: str) -> str:
        """Create a fully qualified feature name."""
        return f"{self.scope}__{self.group_name}__{feature_name}"


class CompositeFeatureExtractor:
    """Sequentially execute multiple extractors."""

    def __init__(self, extractors: list[BaseFeatureExtractor]) -> None:
        self.extractors = extractors

    def feature_names(self) -> list[str]:
        names: list[str] = []
        for extractor in self.extractors:
            names.extend(extractor.feature_names())
        return names

    def extract(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        context: FeatureContext,
    ) -> dict[str, float]:
        features: dict[str, float] = {}
        for extractor in self.extractors:
            features.update(extractor.extract(image=image, mask=mask, context=context))
        return features
