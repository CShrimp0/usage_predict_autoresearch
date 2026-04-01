"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def set_random_seed(seed: int) -> None:
    """Set the random seed for Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
