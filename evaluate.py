"""
Fixed evaluation script for usage_predict autoresearch.

Usage:
    conda activate us
    python evaluate.py --checkpoint outputs/autoresearch/run_xxx/best_model.pth

This script is intended for explicit candidate confirmation, not routine
search-time ranking.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from dataclasses import fields
from pathlib import Path

import torch

import prepare
import train


def _load_checkpoint_config(checkpoint: dict) -> train.ExperimentConfig:
    raw_config = checkpoint.get("config") or checkpoint.get("args") or {}
    valid_fields = {field.name for field in fields(train.ExperimentConfig)}
    filtered = {key: value for key, value in raw_config.items() if key in valid_fields}
    return train.ExperimentConfig(**filtered)


def _build_data_loader(cfg: train.ExperimentConfig, split: str, batch_size: int, num_workers: int):
    train_dataset, val_dataset, test_dataset, metadata = prepare.build_datasets(cfg)
    datasets = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
    dataset = datasets[split]
    data_loader = prepare.make_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=cfg.seed + {"train": 10, "val": 11, "test": 12}[split],
    )
    return data_loader, metadata, len(dataset)


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    split: str,
    batch_size: int | None,
    num_workers: int | None,
    device_override: str | None,
    output_json: str | Path | None,
) -> dict:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = _load_checkpoint_config(checkpoint)

    if batch_size is None:
        batch_size = cfg.batch_size
    if num_workers is None:
        num_workers = cfg.num_workers

    device = torch.device(device_override) if device_override else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader, metadata, sample_count = _build_data_loader(cfg, split, batch_size, num_workers)

    model_cfg = replace(cfg, pretrained_path=None)
    model = train.AgeRegressor(model_cfg, aux_input_dim=int(metadata["aux_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    metrics = prepare.evaluate_regression(
        model,
        data_loader,
        device,
        use_aux=bool(metadata["use_aux_features"]),
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "sample_count": sample_count,
        "prediction_mae": float(metrics["mae"]),
        "prediction_rmse": float(metrics["rmse"]),
        "mean_error": float(metrics["mean_error"]),
        "median_abs_error": float(metrics["median_abs_error"]),
        "best_val_mae": float(checkpoint.get("best_val_mae", checkpoint.get("val_mae", float("nan")))),
        "best_val_rmse": float(checkpoint.get("best_val_rmse", checkpoint.get("val_rmse", float("nan")))),
        "config": prepare.cfg_to_dict(cfg),
        "dataset": metadata,
    }

    if output_json is None:
        output_json = checkpoint_path.parent / f"{split}_eval.json"
    prepare.save_json(output_json, summary)
    summary["output_json"] = str(output_json)
    return summary


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a shortlisted usage_predict_autoresearch checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers for evaluation")
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g. cpu or cuda:0")
    parser.add_argument("--output-json", type=str, default=None, help="Where to save the evaluation JSON")
    return parser


def main() -> dict:
    parser = create_arg_parser()
    args = parser.parse_args()
    summary = evaluate_checkpoint(
        args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_override=args.device,
        output_json=args.output_json,
    )
    print(f"split:            {summary['split']}")
    print(f"sample_count:     {summary['sample_count']}")
    print(f"prediction_mae:   {summary['prediction_mae']:.6f}")
    print(f"prediction_rmse:  {summary['prediction_rmse']:.6f}")
    print(f"best_val_mae:     {summary['best_val_mae']:.6f}")
    print(f"output_json:      {summary['output_json']}")
    return summary


if __name__ == "__main__":
    main()
