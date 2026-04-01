#!/usr/bin/env python3
"""Standalone feature extraction entry point."""

from __future__ import annotations

import pandas as pd

import _bootstrap  # noqa: F401
from _common import base_argument_parser, initialize_run, load_runtime_config
from utils.feature_extraction import extract_feature_table
from utils.seeds import set_random_seed


def main() -> None:
    parser = base_argument_parser("Extract handcrafted ultrasound features.")
    parser.add_argument("--output-csv", default=None, help="Optional explicit output CSV path.")
    args = parser.parse_args()

    config = load_runtime_config(args.config, args.override)
    set_random_seed(int(config.get("seed", 42)))
    run_dir, logger = initialize_run(config, experiment_name=f"{config.get('experiment', {}).get('name', 'experiment')}_features")

    feature_df = extract_feature_table(config)
    output_csv = args.output_csv or str(run_dir / "features_raw.csv")
    feature_df.to_csv(output_csv, index=False)
    logger.info("Saved feature table with shape %s to %s", feature_df.shape, output_csv)


if __name__ == "__main__":
    main()
