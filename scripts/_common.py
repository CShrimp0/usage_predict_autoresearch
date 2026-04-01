"""Shared CLI utilities for scripts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from utils.config import load_configs, save_yaml, set_by_dotted_path
from utils.logging_utils import configure_logging
from utils.paths import create_run_dir, write_command


def parse_key_value_overrides(overrides: list[str]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must use key=value syntax: {item}")
        key, raw_value = item.split("=", 1)
        parsed[key] = yaml.safe_load(raw_value)
    return parsed


def load_runtime_config(config_paths: list[str], overrides: list[str]) -> dict:
    config = load_configs(config_paths)
    for key, value in parse_key_value_overrides(overrides).items():
        config = set_by_dotted_path(config, key, value)
    return config


def initialize_run(config: dict, experiment_name: str | None = None):
    outputs_root = config.get("paths", {}).get("outputs_root", "outputs")
    name = experiment_name or config.get("experiment", {}).get("name", "experiment")
    run_dir = create_run_dir(outputs_root, name)
    logger = configure_logging(run_dir / "run.log", level=logging.INFO)
    save_yaml(config, run_dir / "config_used.yaml")
    write_command(run_dir / "command.sh")
    return run_dir, logger


def base_argument_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Path to a YAML config file. Pass multiple times to merge left-to-right.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override one config value using dotted.path=value syntax.",
    )
    return parser
