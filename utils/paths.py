"""Filesystem and artifact helpers."""

from __future__ import annotations

import shlex
import sys
from datetime import datetime
from pathlib import Path


def create_run_dir(outputs_root: str | Path, experiment_name: str | None = None) -> Path:
    """Create a timestamped run directory under ``outputs_root``."""
    suffix = experiment_name.strip().replace(" ", "_") if experiment_name else "experiment"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(outputs_root) / f"run_{timestamp}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_command(path: str | Path) -> None:
    """Persist the current command line for reproducibility."""
    command = " ".join(shlex.quote(arg) for arg in sys.argv)
    Path(path).write_text(f"{command}\n", encoding="utf-8")
