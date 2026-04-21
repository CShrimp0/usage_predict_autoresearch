"""Image and mask I/O utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def load_grayscale_image(path: str | Path, resize: tuple[int, int] | None = None) -> np.ndarray:
    """Load an image as float32 grayscale in the range [0, 1]."""
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = Image.open(image_path).convert("L")
    if resize is not None:
        image = image.resize(resize, resample=Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    if array.max() > 1.0:
        array = array / 255.0
    return array


def load_mask(path: str | Path, resize: tuple[int, int] | None = None) -> np.ndarray:
    """Load a mask as a boolean array."""
    mask_path = Path(path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    mask = Image.open(mask_path).convert("L")
    if resize is not None:
        mask = mask.resize(resize, resample=Image.Resampling.NEAREST)
    array = np.asarray(mask)
    return array > 0


def apply_mask(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Return masked pixel values or flattened image values if no mask is provided."""
    if mask is None:
        return image.reshape(-1)
    if mask.shape != image.shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match image shape {image.shape}."
        )
    pixels = image[mask]
    if pixels.size == 0:
        raise ValueError("Mask contains no positive pixels.")
    return pixels


def maybe_resize(config: dict[str, Any]) -> tuple[int, int] | None:
    """Read resize configuration."""
    if bool(config.get("use_original_size", False)):
        return None
    resize = config.get("resize")
    if resize in (None, False):
        return None
    if not isinstance(resize, (list, tuple)) or len(resize) != 2:
        raise ValueError("Image resize must be null or a 2-element sequence.")
    return int(resize[1]), int(resize[0])
