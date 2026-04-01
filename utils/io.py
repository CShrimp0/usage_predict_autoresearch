"""Simple serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Write JSON with UTF-8 encoding."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    """Load JSON content."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
