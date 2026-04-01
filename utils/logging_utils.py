"""Logging setup helpers."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_path: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure the root logger for console and optional file logging."""
    logger = logging.getLogger("usage_predict_feature_engineering")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_path is not None:
        file_path = Path(log_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
