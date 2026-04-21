#!/usr/bin/env python3
"""Rebuild outputs/log.txt from existing run directories."""

from __future__ import annotations

import _bootstrap  # noqa: F401
from utils.run_log import rebuild_run_log


def main() -> None:
    log_path, count = rebuild_run_log("/home/szdx/LNX/usage_predict_feature_engineering/outputs")
    print(f"Rebuilt {log_path} with {count} run entries.")


if __name__ == "__main__":
    main()
