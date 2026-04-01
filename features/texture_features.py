"""Texture and radiomics-style matrix features."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

from features.base import BaseFeatureExtractor, FeatureContext

GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
GLRLM_FEATURES = ("sre", "lre", "gln", "rln", "rp", "lgre", "hgre")
GLSZM_FEATURES = ("sze", "lze", "gln", "zsn", "zp", "lgze", "hgze")
NGTDM_FEATURES = ("coarseness", "contrast", "busyness", "complexity", "strength")


def _prepare_quantized(
    image: np.ndarray,
    mask: np.ndarray | None,
    levels: int,
) -> tuple[np.ndarray, np.ndarray]:
    roi = mask.astype(bool) if mask is not None else np.ones_like(image, dtype=bool)
    if roi.sum() == 0:
        raise ValueError("Mask contains no valid pixels.")
    values = image[roi]
    vmin = float(values.min())
    vmax = float(values.max())
    if np.isclose(vmax, vmin):
        quantized = np.zeros_like(image, dtype=np.uint8)
        return quantized, roi
    scaled = (image - vmin) / (vmax - vmin)
    quantized = np.clip(np.floor(scaled * (levels - 1)), 0, levels - 1).astype(np.uint8)
    return quantized, roi


def _crop_to_mask(image: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    if mask is None:
        return image, None
    rows, cols = np.where(mask)
    if rows.size == 0:
        raise ValueError("Mask contains no positive pixels.")
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1
    return image[r0:r1, c0:c1], mask[r0:r1, c0:c1]


def _line_starts(shape: tuple[int, int], direction: tuple[int, int]) -> Iterable[tuple[int, int]]:
    rows, cols = shape
    dr, dc = direction
    starts: list[tuple[int, int]] = []
    for row in range(rows):
        prev_r = row - dr
        prev_c = -dc
        if not (0 <= prev_r < rows and 0 <= prev_c < cols):
            starts.append((row, 0))
    for col in range(cols):
        prev_r = -dr
        prev_c = col - dc
        if not (0 <= prev_r < rows and 0 <= prev_c < cols):
            starts.append((0, col))
    deduped = []
    seen = set()
    for item in starts:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _build_glrlm(
    quantized: np.ndarray,
    roi: np.ndarray,
    levels: int,
    directions: list[tuple[int, int]],
) -> np.ndarray:
    runs: dict[tuple[int, int], int] = defaultdict(int)
    max_run = 1
    shape = quantized.shape
    for dr, dc in directions:
        for start_row, start_col in _line_starts(shape, (dr, dc)):
            row, col = start_row, start_col
            current_level = None
            current_run = 0
            while 0 <= row < shape[0] and 0 <= col < shape[1]:
                if roi[row, col]:
                    level = int(quantized[row, col])
                    if level == current_level:
                        current_run += 1
                    else:
                        if current_level is not None and current_run > 0:
                            runs[(current_level, current_run)] += 1
                            max_run = max(max_run, current_run)
                        current_level = level
                        current_run = 1
                else:
                    if current_level is not None and current_run > 0:
                        runs[(current_level, current_run)] += 1
                        max_run = max(max_run, current_run)
                    current_level = None
                    current_run = 0
                row += dr
                col += dc
            if current_level is not None and current_run > 0:
                runs[(current_level, current_run)] += 1
                max_run = max(max_run, current_run)

    matrix = np.zeros((levels, max_run), dtype=np.float64)
    for (level, run_length), count in runs.items():
        matrix[level, run_length - 1] += count
    return matrix


def _glrlm_statistics(matrix: np.ndarray, n_pixels: int) -> dict[str, float]:
    if matrix.sum() == 0:
        return {name: np.nan for name in GLRLM_FEATURES}
    i = np.arange(1, matrix.shape[0] + 1, dtype=np.float64)[:, None]
    j = np.arange(1, matrix.shape[1] + 1, dtype=np.float64)[None, :]
    total_runs = matrix.sum()
    pg = matrix.sum(axis=1, keepdims=True)
    pr = matrix.sum(axis=0, keepdims=True)
    return {
        "sre": float((matrix / (j ** 2)).sum() / total_runs),
        "lre": float((matrix * (j ** 2)).sum() / total_runs),
        "gln": float(((pg ** 2).sum()) / total_runs),
        "rln": float(((pr ** 2).sum()) / total_runs),
        "rp": float(total_runs / max(n_pixels, 1)),
        "lgre": float((matrix / (i ** 2)).sum() / total_runs),
        "hgre": float((matrix * (i ** 2)).sum() / total_runs),
    }


def _build_glszm(quantized: np.ndarray, roi: np.ndarray, levels: int) -> np.ndarray:
    zones: dict[tuple[int, int], int] = defaultdict(int)
    max_zone = 1
    structure = np.ones((3, 3), dtype=int)
    for level in range(levels):
        level_mask = (quantized == level) & roi
        labeled, n_components = ndimage.label(level_mask, structure=structure)
        if n_components == 0:
            continue
        sizes = np.bincount(labeled.ravel())[1:]
        for size in sizes:
            zones[(level, int(size))] += 1
            max_zone = max(max_zone, int(size))
    matrix = np.zeros((levels, max_zone), dtype=np.float64)
    for (level, size), count in zones.items():
        matrix[level, size - 1] += count
    return matrix


def _glszm_statistics(matrix: np.ndarray, n_pixels: int) -> dict[str, float]:
    if matrix.sum() == 0:
        return {name: np.nan for name in GLSZM_FEATURES}
    i = np.arange(1, matrix.shape[0] + 1, dtype=np.float64)[:, None]
    j = np.arange(1, matrix.shape[1] + 1, dtype=np.float64)[None, :]
    total_zones = matrix.sum()
    pg = matrix.sum(axis=1, keepdims=True)
    pz = matrix.sum(axis=0, keepdims=True)
    return {
        "sze": float((matrix / (j ** 2)).sum() / total_zones),
        "lze": float((matrix * (j ** 2)).sum() / total_zones),
        "gln": float(((pg ** 2).sum()) / total_zones),
        "zsn": float(((pz ** 2).sum()) / total_zones),
        "zp": float(total_zones / max(n_pixels, 1)),
        "lgze": float((matrix / (i ** 2)).sum() / total_zones),
        "hgze": float((matrix * (i ** 2)).sum() / total_zones),
    }


def _ngtdm_statistics(quantized: np.ndarray, roi: np.ndarray, levels: int) -> dict[str, float]:
    kernel = np.ones((3, 3), dtype=np.float64)
    kernel[1, 1] = 0.0
    roi_float = roi.astype(np.float64)
    neighborhood_count = ndimage.convolve(roi_float, kernel, mode="constant", cval=0.0)
    neighborhood_sum = ndimage.convolve(quantized.astype(np.float64) * roi_float, kernel, mode="constant", cval=0.0)
    valid = roi & (neighborhood_count > 0)
    if valid.sum() == 0:
        return {name: np.nan for name in NGTDM_FEATURES}

    neighbor_mean = np.zeros_like(quantized, dtype=np.float64)
    neighbor_mean[valid] = neighborhood_sum[valid] / neighborhood_count[valid]

    counts = np.zeros(levels, dtype=np.float64)
    s = np.zeros(levels, dtype=np.float64)
    for level in range(levels):
        level_mask = valid & (quantized == level)
        counts[level] = float(level_mask.sum())
        if counts[level] > 0:
            s[level] = float(np.abs(level - neighbor_mean[level_mask]).sum())

    present = counts > 0
    if present.sum() < 2:
        return {name: np.nan for name in NGTDM_FEATURES}
    p = counts / counts.sum()
    levels_idx = np.arange(levels, dtype=np.float64)
    sum_ps = float(np.sum(p * s))
    level_diff = np.abs(levels_idx[:, None] - levels_idx[None, :])
    weighted_diff = (p[:, None] * p[None, :] * (level_diff ** 2)).sum()

    denominator_busyness = np.sum(
        np.abs(levels_idx[:, None] * p[:, None] - levels_idx[None, :] * p[None, :])
    )
    complexity_numerator = 0.0
    strength_numerator = 0.0
    for i in range(levels):
        for j in range(levels):
            if i == j or not present[i] or not present[j]:
                continue
            complexity_numerator += (
                level_diff[i, j] * (p[i] * s[i] + p[j] * s[j]) / max(p[i] + p[j], 1e-12)
            )
            strength_numerator += (p[i] + p[j]) * (level_diff[i, j] ** 2)

    return {
        "coarseness": float(1.0 / max(sum_ps, 1e-12)),
        "contrast": float(weighted_diff * sum_ps / max(valid.sum(), 1)),
        "busyness": float(sum_ps / max(denominator_busyness, 1e-12)),
        "complexity": float(complexity_numerator / max(valid.sum(), 1)),
        "strength": float(strength_numerator / max(sum_ps, 1e-12)),
    }


class TextureFeatureExtractor(BaseFeatureExtractor):
    """GLCM, GLRLM, GLSZM, NGTDM, LBP, and optional HOG features."""

    group_name = "texture"

    def __init__(self, config: dict[str, Any], scope: str) -> None:
        super().__init__(config, scope)
        self.levels = int(config.get("quantization_levels", 16))
        self.glcm_distances = [int(v) for v in config.get("glcm_distances", [1, 2])]
        self.glcm_angles = [float(v) for v in config.get("glcm_angles", [0.0, 0.785398, 1.570796, 2.356194])]
        self.lbp_radius = int(config.get("lbp_radius", 1))
        self.lbp_points = int(config.get("lbp_points", 8))
        self.include_hog = bool(config.get("include_hog", False))
        self.include_glrlm = bool(config.get("include_glrlm", True))
        self.include_glszm = bool(config.get("include_glszm", True))
        self.include_ngtdm = bool(config.get("include_ngtdm", False))
        self.hog_bins = int(config.get("hog_bins", 9))
        self.hog_pixels_per_cell = tuple(config.get("hog_pixels_per_cell", [16, 16]))
        self.hog_cells_per_block = tuple(config.get("hog_cells_per_block", [2, 2]))

    def feature_names(self) -> list[str]:
        names: list[str] = []
        for prop in GLCM_PROPS:
            for distance in self.glcm_distances:
                for angle in self.glcm_angles:
                    names.append(self.prefix(f"glcm__{prop}__d{distance}__a{angle:.3f}"))
        lbp_hist_size = self.lbp_points + 2
        for idx in range(lbp_hist_size):
            names.append(self.prefix(f"lbp__hist_bin_{idx}"))
        if self.include_glrlm:
            names.extend(self.prefix(f"glrlm__{name}") for name in GLRLM_FEATURES)
        if self.include_glszm:
            names.extend(self.prefix(f"glszm__{name}") for name in GLSZM_FEATURES)
        if self.include_ngtdm:
            names.extend(self.prefix(f"ngtdm__{name}") for name in NGTDM_FEATURES)
        if self.include_hog:
            names.extend(
                self.prefix(
                    f"hog__bin_{idx}"
                )
                for idx in range(
                    self.hog_bins * self.hog_cells_per_block[0] * self.hog_cells_per_block[1]
                )
            )
        return names

    def _extract_impl(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        context: FeatureContext,
    ) -> dict[str, float]:
        quantized, roi = _prepare_quantized(image, mask, self.levels)
        cropped_quantized, cropped_mask = _crop_to_mask(quantized, roi)
        working_mask = cropped_mask if cropped_mask is not None else np.ones_like(cropped_quantized, dtype=bool)
        working_quantized = np.where(working_mask, cropped_quantized, 0)

        glcm = graycomatrix(
            working_quantized,
            distances=self.glcm_distances,
            angles=self.glcm_angles,
            levels=self.levels,
            symmetric=True,
            normed=True,
        )
        features: dict[str, float] = {}
        for prop in GLCM_PROPS:
            values = graycoprops(glcm, prop)
            for distance_index, distance in enumerate(self.glcm_distances):
                for angle_index, angle in enumerate(self.glcm_angles):
                    features[self.prefix(f"glcm__{prop}__d{distance}__a{angle:.3f}")] = float(
                        values[distance_index, angle_index]
                    )

        lbp = local_binary_pattern(
            working_quantized,
            P=self.lbp_points,
            R=self.lbp_radius,
            method="uniform",
        )
        lbp_hist, _ = np.histogram(
            lbp[working_mask],
            bins=self.lbp_points + 2,
            range=(0, self.lbp_points + 2),
            density=True,
        )
        for idx, value in enumerate(lbp_hist):
            features[self.prefix(f"lbp__hist_bin_{idx}")] = float(value)

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        if self.include_glrlm:
            glrlm = _build_glrlm(working_quantized, working_mask, self.levels, directions)
            for name, value in _glrlm_statistics(glrlm, int(working_mask.sum())).items():
                features[self.prefix(f"glrlm__{name}")] = value

        if self.include_glszm:
            glszm = _build_glszm(working_quantized, working_mask, self.levels)
            for name, value in _glszm_statistics(glszm, int(working_mask.sum())).items():
                features[self.prefix(f"glszm__{name}")] = value

        if self.include_ngtdm:
            for name, value in _ngtdm_statistics(working_quantized, working_mask, self.levels).items():
                features[self.prefix(f"ngtdm__{name}")] = value

        if self.include_hog:
            float_image = working_quantized.astype(np.float32) / max(self.levels - 1, 1)
            hog_vector = hog(
                float_image,
                orientations=self.hog_bins,
                pixels_per_cell=self.hog_pixels_per_cell,
                cells_per_block=self.hog_cells_per_block,
                feature_vector=True,
                visualize=False,
            )
            expected = self.hog_bins * self.hog_cells_per_block[0] * self.hog_cells_per_block[1]
            for idx in range(expected):
                value = float(hog_vector[idx]) if idx < len(hog_vector) else np.nan
                features[self.prefix(f"hog__bin_{idx}")] = value

        return features
