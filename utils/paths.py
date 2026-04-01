"""Filesystem and artifact helpers."""

from __future__ import annotations

import re
import shlex
import sys
from datetime import datetime
from pathlib import Path


def slugify_run_part(value: str | None) -> str | None:
    """Normalize one run-name component into a filesystem-friendly slug."""
    if value is None:
        return None
    text = value.strip().lower()
    if not text:
        return None
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or None


def build_run_suffix(parts: list[str | None]) -> str:
    """Build a deduplicated run-name suffix from multiple components."""
    normalized = []
    seen = set()
    for part in parts:
        slug = slugify_run_part(part)
        if slug and slug not in seen:
            normalized.append(slug)
            seen.add(slug)
    return "_".join(normalized) if normalized else "experiment"


def create_run_dir(outputs_root: str | Path, run_suffix: str | None = None) -> Path:
    """Create a timestamped run directory under ``outputs_root``."""
    suffix = slugify_run_part(run_suffix) or "experiment"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(outputs_root) / f"run_{timestamp}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_command(path: str | Path) -> None:
    """Persist the current command line for reproducibility."""
    command = " ".join(shlex.quote(arg) for arg in sys.argv)
    Path(path).write_text(f"{command}\n", encoding="utf-8")
