#!/usr/bin/env python3
"""Generate a compact image/mask/overlay figure for one ROI example."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_grayscale(path: str | Path) -> np.ndarray:
    array = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    if array.max() > 0:
        array = array / array.max()
    return array


def load_mask(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return None
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def add_bbox(ax, bbox: tuple[int, int, int, int] | None) -> None:
    if bbox is None:
        return
    import matplotlib.patches as patches

    r0, r1, c0, c1 = bbox
    rect = patches.Rectangle(
        (c0, r0),
        c1 - c0 + 1,
        r1 - r0 + 1,
        linewidth=2.0,
        edgecolor="#f59e0b",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)


def build_figure(image_path: Path, mask_path: Path, output_path: Path, title: str | None) -> None:
    image = load_grayscale(image_path)
    mask = load_mask(mask_path)
    bbox = mask_bbox(mask)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Image")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mask")

    axes[2].imshow(image, cmap="gray")
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 3] = mask.astype(np.float32) * 0.35
    axes[2].imshow(overlay)
    add_bbox(axes[2], bbox)
    axes[2].set_title("Overlay + ROI bbox")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an image/mask/overlay ROI demo figure.")
    parser.add_argument("--image", required=True, help="Path to the source image.")
    parser.add_argument("--mask", required=True, help="Path to the binary mask.")
    parser.add_argument("--output", required=True, help="Path to the output PNG.")
    parser.add_argument("--title", default=None, help="Optional figure title.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_figure(
        image_path=Path(args.image),
        mask_path=Path(args.mask),
        output_path=Path(args.output),
        title=args.title,
    )


if __name__ == "__main__":
    main()
