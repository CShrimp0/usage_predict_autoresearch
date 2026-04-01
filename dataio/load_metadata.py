"""Load metadata tables from CSV/Excel sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _resolve_sheet_name(sheet_name: Any) -> Any:
    if sheet_name in (None, "", "null", "None"):
        return 0
    return sheet_name


def load_metadata_table(
    metadata_path: str | Path,
    file_type: str = "auto",
    sheet_name: Any = None,
) -> pd.DataFrame:
    """Load a metadata table from CSV/TSV/Excel."""
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    effective_type = file_type.lower()
    if effective_type == "auto":
        suffix = path.suffix.lower()
        if suffix in {".xlsx", ".xls", ".xlsm"}:
            effective_type = "excel"
        elif suffix in {".tsv", ".txt"}:
            effective_type = "tsv"
        else:
            effective_type = "csv"

    if effective_type == "excel":
        return pd.read_excel(path, sheet_name=_resolve_sheet_name(sheet_name))
    if effective_type == "tsv":
        return pd.read_csv(path, sep="\t")
    if effective_type == "csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported metadata file type: {file_type}")
