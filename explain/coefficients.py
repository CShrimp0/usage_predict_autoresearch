"""Linear-model coefficient summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd


def coefficient_importance(feature_names, coefficients) -> pd.DataFrame:
    """Convert coefficients into a sortable dataframe."""
    feature_names = np.asarray(feature_names, dtype=object)
    coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": np.abs(coefficients),
            "signed_value": coefficients,
            "importance_type": "coefficient_abs",
        }
    )
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)
