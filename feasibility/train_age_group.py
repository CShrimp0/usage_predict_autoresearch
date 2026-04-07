"""粗粒度年龄三分类训练脚本。"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import stat
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from age_group_dataset import load_age_group_datasets
from age_group_model import get_age_group_model
from utils import (
    DEFAULT_CLASS_NAMES,
    DEFAULT_EXCEL_PATH,
    DEFAULT_IMAGE_DIR,
    ensure_dir,
    get_device,
    save_json,
    seed_everything,
)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for images, labels, _ in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size
        running_correct += (preds == labels).sum().item()
        running_total += batch_size

    avg_loss = running_loss / max(running_total, 1)
    avg_acc = running_correct / max(running_total, 1)
    return avg_loss, avg_acc


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    all_labels: List[int] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Val", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (preds == labels).sum().item()
            running_total += batch_size

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = running_loss / max(running_total, 1)
    avg_acc = running_correct / max(running_total, 1)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, avg_acc, float(macro_f1)


def compute_class_weights(
    labels: List[int],
    num_classes: int,
    mode: str = "auto",
    manual_weights: List[float] | None = None,
) -> Tuple[torch.Tensor | None, List[float] | None]:
    if mode == "none":
        return None, None

    if mode == "manual":
        if manual_weights is None or len(manual_weights) != num_classes:
            raise ValueError(
                f"manual class weights 需要提供 {num_classes} 个值，当前为: {manual_weights}"
            )
        weights = np.asarray(manual_weights, dtype=np.float32)
        return torch.tensor(weights, dtype=torch.float32), weights.tolist()

    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    total = float(counts.sum())

    weights = np.zeros(num_classes, dtype=np.float32)
    for i, cnt in enumerate(counts):
        if cnt > 0:
            weights[i] = total / (num_classes * cnt)
        else:
            weights[i] = 0.0

    return torch.tensor(weights, dtype=torch.float32), weights.tolist()


def generate_command_line(args: argparse.Namespace) -> str:
    cmd_args = [
        f"--model {args.model}",
        f"--batch-size {args.batch_size}",
        f"--dropout {args.dropout}",
        f"--lr {args.lr}",
        f"--weight-decay {args.weight_decay}",
        f"--epochs {args.epochs}",
        f"--patience {args.patience}",
        f"--image-size {args.image_size}",
        f"--seed {args.seed}",
        f"--test-size {args.test_size}",
        f"--val-size {args.val_size}",
        f"--min-age {args.min_age}",
        f"--max-age {args.max_age}",
        f"--young-max {args.young_max}",
        f"--middle-max {args.middle_max}",
        f"--stratify-mode {args.stratify_mode}",
        f"--age-bin-width {args.age_bin_width}",
        f"--class-weights {args.class_weights}",
        f"--num-workers {args.num_workers}",
    ]

    if args.class_weights == "manual" and args.class_weight_values is not None:
        manual_values = " ".join(str(x) for x in args.class_weight_values)
        cmd_args.append(f"--class-weight-values {manual_values}")

    if not args.pretrained:
        cmd_args.append("--no-pretrained")

    if args.image_dir != DEFAULT_IMAGE_DIR:
        cmd_args.append(f'--image-dir "{args.image_dir}"')

    if args.excel_path != DEFAULT_EXCEL_PATH:
        cmd_args.append(f'--excel-path "{args.excel_path}"')

    if args.output_dir != "./outputs/feasibility":
        cmd_args.append(f'--output-dir "{args.output_dir}"')

    if args.deterministic:
        cmd_args.append("--deterministic")

    lines = ["python feasibility/train_age_group.py \\"]
    for idx, item in enumerate(cmd_args):
        suffix = " \\" if idx < len(cmd_args) - 1 else ""
        lines.append(f"  {item}{suffix}")
    return "\n".join(lines)


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练粗粒度年龄三分类可行性实验")

    parser.add_argument("--image-dir", type=str, default=DEFAULT_IMAGE_DIR, help="图像文件夹路径")
    parser.add_argument("--excel-path", type=str, default=DEFAULT_EXCEL_PATH, help="Excel标签文件路径")
    parser.add_argument("--output-dir", type=str, default="./outputs/feasibility", help="输出目录")

    parser.add_argument("--test-size", type=float, default=0.15, help="测试集比例")
    parser.add_argument("--val-size", type=float, default=0.15, help="验证集比例（从train_val中划分）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--deterministic", action="store_true", help="启用确定性训练")

    parser.add_argument("--min-age", type=float, default=18, help="最小年龄（包含）")
    parser.add_argument("--max-age", type=float, default=100, help="最大年龄（包含）")
    parser.add_argument("--young-max", type=float, default=35, help="青年上界（不含）")
    parser.add_argument("--middle-max", type=float, default=60, help="中年上界（不含）")

    parser.add_argument(
        "--stratify-mode",
        type=str,
        default="coarse",
        choices=["coarse", "age_bin", "none"],
        help="受试者级分层方式",
    )
    parser.add_argument("--age-bin-width", type=int, default=10, help="按年龄分层时的bin宽度")

    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet50", "efficientnet_b0", "efficientnet_b1", "convnext", "mobilenet_v3", "regnet"],
        help="模型骨干网络",
    )
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True, help="使用预训练权重")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="不使用预训练权重")
    parser.add_argument("--dropout", type=float, default=0.6, help="Dropout比例")
    parser.add_argument("--image-size", type=int, default=224, choices=[224, 256], help="输入图像尺寸")

    parser.add_argument("--epochs", type=int, default=200, help="最大训练轮数")
    parser.add_argument("--patience", type=int, default=50, help="早停耐心值")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--num-workers", type=int, default=8, help="数据加载线程数")

    parser.add_argument(
        "--class-weights",
        type=str,
        default="auto",
        choices=["auto", "none", "manual"],
        help="类别权重策略",
    )
    parser.add_argument(
        "--class-weight-values",
        nargs=3,
        type=float,
        default=None,
        help="manual模式下类别权重，顺序: young middle old",
    )

    return parser


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed, deterministic=args.deterministic)
    device = get_device()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_dir(Path(args.output_dir) / f"run_{timestamp}")

    print(f"使用设备: {device}")
    print(f"输出目录: {output_dir}")

    train_dataset, val_dataset, _, split_info = load_age_group_datasets(
        image_dir=args.image_dir,
        excel_path=args.excel_path,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
        image_size=args.image_size,
        stratify_mode=args.stratify_mode,
        age_bin_width=args.age_bin_width,
        min_age=args.min_age,
        max_age=args.max_age,
        young_max=args.young_max,
        middle_max=args.middle_max,
        class_names=DEFAULT_CLASS_NAMES,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = get_age_group_model(
        model_name=args.model,
        pretrained=args.pretrained,
        dropout=args.dropout,
        num_classes=len(DEFAULT_CLASS_NAMES),
    ).to(device)

    class_weight_tensor, class_weight_list = compute_class_weights(
        train_dataset.labels,
        num_classes=len(DEFAULT_CLASS_NAMES),
        mode=args.class_weights,
        manual_weights=args.class_weight_values,
    )
    if class_weight_tensor is not None:
        class_weight_tensor = class_weight_tensor.to(device)
        print(f"使用类别权重: {class_weight_list}")
    else:
        print("类别权重: disabled")

    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
    }

    config = {
        "script_name": "feasibility/train_age_group.py",
        "script_version": "1.0",
        "timestamp": timestamp,
        "description": "Coarse age-group feasibility classification",
        "environment": {
            "device": str(device),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else [],
            "gpu_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "pytorch_version": torch.__version__,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        },
        "dataset": {
            "image_dir": args.image_dir,
            "excel_path": args.excel_path,
            "split_info": split_info,
        },
        "model": {
            "architecture": args.model,
            "pretrained": args.pretrained,
            "dropout": args.dropout,
            "num_classes": len(DEFAULT_CLASS_NAMES),
            "class_names": DEFAULT_CLASS_NAMES,
            "task": "age_group_classification",
        },
        "training": {
            "loss_function": "CrossEntropyLoss",
            "class_weights_mode": args.class_weights,
            "class_weights": class_weight_list,
            "optimizer": "AdamW",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "num_workers": args.num_workers,
            "early_stopping": {
                "enabled": True,
                "patience": args.patience,
                "monitor": "val_macro_f1",
                "mode": "max",
            },
            "tracked_metrics": [
                "train_loss",
                "val_loss",
                "train_acc",
                "val_acc",
                "val_macro_f1",
            ],
        },
        "output": {
            "output_dir": args.output_dir,
            "run_name": output_dir.name,
            "saved_files": [
                "config.json",
                "command.sh",
                "history.json",
                "best_model.pth",
            ],
        },
        "all_args": vars(args),
    }

    save_json(config, output_dir / "config.json", ensure_ascii=False)

    command_file = output_dir / "command.sh"
    with open(command_file, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write("# Reproduce this feasibility run\n")
        f.write(f"# Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(generate_command_line(args))
        f.write("\n")
    command_file.chmod(command_file.stat().st_mode | stat.S_IEXEC)

    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    train_start = time.time()

    print("\n开始训练...")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_macro_f1 = validate_one_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        history["val_macro_f1"].append(float(val_macro_f1))

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, val_macro_f1={val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = float(val_macro_f1)
            best_epoch = int(epoch)
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1": float(val_macro_f1),
                    "val_acc": float(val_acc),
                    "model": args.model,
                    "dropout": float(args.dropout),
                    "pretrained": bool(args.pretrained),
                    "num_classes": int(len(DEFAULT_CLASS_NAMES)),
                    "class_names": DEFAULT_CLASS_NAMES,
                    "age_boundaries": {
                        "young_max": float(args.young_max),
                        "middle_max": float(args.middle_max),
                        "min_age": float(args.min_age),
                        "max_age": float(args.max_age),
                    },
                    "split_info": split_info,
                    "args": vars(args),
                },
                output_dir / "best_model.pth",
            )
            print(f"  ✓ 保存最佳模型 (val_macro_f1={best_val_macro_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  ⚠ 无提升 ({patience_counter}/{args.patience})")

            if patience_counter >= args.patience:
                print(f"\n早停触发: 连续 {args.patience} 个epoch无提升。")
                break

        save_json(history, output_dir / "history.json")

    total_time = time.time() - train_start
    save_json(history, output_dir / "history.json")

    config["training"]["best_epoch"] = best_epoch
    config["training"]["best_val_macro_f1"] = best_val_macro_f1
    config["training"]["train_time_seconds"] = total_time
    save_json(config, output_dir / "config.json", ensure_ascii=False)

    print("\n训练完成")
    print(f"best_epoch: {best_epoch}")
    print(f"best_val_macro_f1: {best_val_macro_f1:.4f}")
    print(f"结果目录: {output_dir}")


if __name__ == "__main__":
    parser = create_arg_parser()
    main(parser.parse_args())
