"""粗粒度年龄三分类评估脚本。"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from age_group_dataset import load_age_group_datasets
from age_group_model import get_age_group_model
from utils import DEFAULT_CLASS_NAMES, DEFAULT_EXCEL_PATH, DEFAULT_IMAGE_DIR, ensure_dir, get_device, save_json


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyze_misclassification_patterns(
    prediction_records: List[dict],
    class_names: List[str],
) -> Dict:
    misclassified = [x for x in prediction_records if x["true_label"] != x["pred_label"]]

    transition_counts: Dict[str, int] = {}
    transition_age_lists: Dict[str, List[float]] = {}

    for row in misclassified:
        key = f"{class_names[row['true_label']]}->{class_names[row['pred_label']]}"
        transition_counts[key] = transition_counts.get(key, 0) + 1
        transition_age_lists.setdefault(key, []).append(float(row["true_age"]))

    sorted_transitions = sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)

    transition_summary = []
    for k, v in sorted_transitions:
        ages = transition_age_lists.get(k, [])
        transition_summary.append(
            {
                "transition": k,
                "count": int(v),
                "mean_true_age": float(np.mean(ages)) if ages else None,
                "std_true_age": float(np.std(ages)) if ages else None,
            }
        )

    return {
        "misclassified_count": int(len(misclassified)),
        "total_samples": int(len(prediction_records)),
        "misclassification_rate": float(len(misclassified) / max(len(prediction_records), 1)),
        "transition_counts": transition_counts,
        "transition_summary": transition_summary,
    }


def save_misclassified_samples(prediction_records: List[dict], class_names: List[str], output_path: Path) -> None:
    misclassified = [x for x in prediction_records if x["true_label"] != x["pred_label"]]
    misclassified = sorted(misclassified, key=lambda x: x.get("confidence", 0.0), reverse=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Misclassified Samples\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total: {len(misclassified)}\n\n")

        for i, row in enumerate(misclassified, start=1):
            probs_str = ", ".join(f"{p:.4f}" for p in row["pred_probs"])
            true_name = row.get("true_label_name", class_names[row["true_label"]])
            pred_name = row.get("pred_label_name", class_names[row["pred_label"]])
            f.write(
                f"[{i}] {row['filename']} | age={row['true_age']:.1f} | "
                f"true={true_name}({row['true_label']}) | "
                f"pred={pred_name}({row['pred_label']}) | "
                f"conf={row.get('confidence', 0.0):.4f} | probs=[{probs_str}]\n"
            )


def save_confusion_matrix_text(cm: np.ndarray, class_names: List[str], output_path: Path) -> None:
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_ratio = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=np.float64), where=row_sum != 0)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Confusion Matrix (Count + Row Normalized %)\n")
        f.write("=" * 80 + "\n\n")
        f.write("Rows: true class, Columns: predicted class\n\n")

        header = ["true\\pred"] + class_names
        f.write("\t".join(header) + "\n")
        for i, name in enumerate(class_names):
            row_items = [name]
            for j in range(len(class_names)):
                row_items.append(f"{cm[i, j]} ({cm_ratio[i, j] * 100:.1f}%)")
            f.write("\t".join(row_items) + "\n")


def save_predictions_csv(prediction_records: List[dict], class_names: List[str], output_path: Path) -> None:
    fieldnames = [
        "filename",
        "filepath",
        "true_age",
        "true_label",
        "true_label_name",
        "pred_label",
        "pred_label_name",
        "confidence",
        "confidence_margin",
        "nearest_boundary_distance",
        "is_correct",
    ] + [f"prob_{name}" for name in class_names]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in prediction_records:
            out_row = {k: row.get(k) for k in fieldnames if not k.startswith("prob_")}
            for name in class_names:
                out_row[f"prob_{name}"] = row["pred_probs_dict"].get(name, 0.0)
            writer.writerow(out_row)


def get_compact_split_info(split_info: Dict) -> Dict:
    compact = copy.deepcopy(split_info)
    subject_ids = compact.pop("subject_ids", None)
    if subject_ids is not None:
        compact["subject_id_counts"] = {k: len(v) for k, v in subject_ids.items()}
    return compact


def summarize_boundary_difficulty(
    prediction_records: List[dict], young_max: float, middle_max: float
) -> List[Dict]:
    boundaries = [young_max, middle_max]
    bins = [(0.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, float("inf"))]
    rows = []

    for low, high in bins:
        subset = [
            x
            for x in prediction_records
            if low <= min(abs(x["true_age"] - b) for b in boundaries) < high
        ]
        n = len(subset)
        if n == 0:
            rows.append(
                {
                    "distance_range": f"[{low:.0f}, {high if np.isfinite(high) else 'inf'})",
                    "count": 0,
                    "accuracy": None,
                }
            )
            continue
        acc = sum(x["is_correct"] for x in subset) / n
        rows.append(
            {
                "distance_range": f"[{low:.0f}, {high if np.isfinite(high) else 'inf'})",
                "count": n,
                "accuracy": float(acc),
            }
        )
    return rows


def summarize_confidence(prediction_records: List[dict]) -> List[Dict]:
    bins = [(0.0, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    rows = []
    for low, high in bins:
        subset = [x for x in prediction_records if low <= x["confidence"] < high]
        n = len(subset)
        if n == 0:
            rows.append({"confidence_range": f"[{low:.1f}, {high:.1f})", "count": 0, "accuracy": None})
            continue
        acc = sum(x["is_correct"] for x in subset) / n
        rows.append({"confidence_range": f"[{low:.1f}, {high:.1f})", "count": n, "accuracy": float(acc)})
    return rows


def save_inference_summary(
    metrics_payload: Dict,
    report_text: str,
    output_path: Path,
) -> None:
    overall = metrics_payload["overall_metrics"]
    error_analysis = metrics_payload["error_analysis"]
    per_class = metrics_payload["per_class_metrics"]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Inference Summary\n\n")
        f.write("## 1) 一句话结论\n")

        hardest = min(per_class, key=lambda x: x["f1"])
        easiest = max(per_class, key=lambda x: x["f1"])
        top_transition = error_analysis.get("transition_summary", [])
        top_transition_text = top_transition[0]["transition"] if top_transition else "N/A"
        top_transition_count = top_transition[0]["count"] if top_transition else 0

        f.write(
            f"- Macro F1 = **{overall['macro_f1']:.4f}**，"
            f"最难类别是 **{hardest['class_name']}** (F1={hardest['f1']:.4f})，"
            f"最稳类别是 **{easiest['class_name']}** (F1={easiest['f1']:.4f})。\n"
        )
        f.write(f"- 主要错分模式：**{top_transition_text}**（{top_transition_count} 例）。\n\n")

        f.write("## 2) Overall Metrics\n")
        f.write("| Metric | Value |\n")
        f.write("|---|---:|\n")
        f.write(f"| Accuracy | {overall['accuracy']:.4f} |\n")
        f.write(f"| Balanced Accuracy | {overall['balanced_accuracy']:.4f} |\n")
        f.write(f"| Macro Precision | {overall['macro_precision']:.4f} |\n")
        f.write(f"| Macro Recall | {overall['macro_recall']:.4f} |\n")
        f.write(f"| Macro F1 | {overall['macro_f1']:.4f} |\n")
        f.write(f"| Weighted F1 | {overall['weighted_f1']:.4f} |\n\n")

        f.write("## 3) Per-Class Metrics\n")
        f.write("| Class | Precision | Recall | F1 | Support |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in per_class:
            f.write(
                f"| {row['class_name']} | {row['precision']:.4f} | {row['recall']:.4f} | "
                f"{row['f1']:.4f} | {row['support']} |\n"
            )
        f.write("\n")

        f.write("## 4) Misclassification Patterns\n")
        f.write(
            f"- Misclassified: {error_analysis['misclassified_count']}/"
            f"{error_analysis['total_samples']} "
            f"({error_analysis['misclassification_rate'] * 100:.2f}%)\n"
        )
        f.write("| Transition | Count | Mean True Age | Std True Age |\n")
        f.write("|---|---:|---:|---:|\n")
        for row in error_analysis.get("transition_summary", []):
            f.write(
                f"| {row['transition']} | {row['count']} | "
                f"{row['mean_true_age']:.2f} | {row['std_true_age']:.2f} |\n"
            )
        f.write("\n")

        readability = metrics_payload.get("readability_analysis", {})
        boundary_rows = readability.get("boundary_difficulty", [])
        conf_rows = readability.get("confidence_buckets", [])

        if boundary_rows:
            f.write("## 5) 边界敏感性（距35/60岁边界）\n")
            f.write("| Distance Bin | Count | Accuracy |\n")
            f.write("|---|---:|---:|\n")
            for row in boundary_rows:
                acc_text = f"{row['accuracy']:.4f}" if row["accuracy"] is not None else "N/A"
                f.write(f"| {row['distance_range']} | {row['count']} | {acc_text} |\n")
            f.write("\n")

        if conf_rows:
            f.write("## 6) 置信度分桶\n")
            f.write("| Confidence Bin | Count | Accuracy |\n")
            f.write("|---|---:|---:|\n")
            for row in conf_rows:
                acc_text = f"{row['accuracy']:.4f}" if row["accuracy"] is not None else "N/A"
                f.write(f"| {row['confidence_range']} | {row['count']} | {acc_text} |\n")
            f.write("\n")

        f.write("## 7) Classification Report\n")
        f.write("```\n")
        f.write(report_text.strip() + "\n")
        f.write("```\n")


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评估粗粒度年龄三分类模型")

    parser.add_argument("--checkpoint", type=str, required=True, help="训练得到的best_model.pth路径")
    parser.add_argument("--image-dir", type=str, default=None, help="图像目录（默认读取checkpoint训练参数）")
    parser.add_argument("--excel-path", type=str, default=None, help="Excel标签（默认读取checkpoint训练参数）")
    parser.add_argument("--output-dir", type=str, default=None, help="评估输出目录（默认: checkpoint同级/age_group_eval）")

    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--num-workers", type=int, default=4, help="数据加载线程数")

    return parser


def main(args: argparse.Namespace) -> None:
    device = get_device()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = checkpoint.get("args", {})

    image_dir = args.image_dir or train_args.get("image_dir", DEFAULT_IMAGE_DIR)
    excel_path = args.excel_path or train_args.get("excel_path", DEFAULT_EXCEL_PATH)

    # 保持与训练一致的数据划分参数
    dataset_params = {
        "test_size": train_args.get("test_size", 0.15),
        "val_size": train_args.get("val_size", 0.15),
        "random_state": train_args.get("seed", 42),
        "image_size": train_args.get("image_size", 224),
        "stratify_mode": train_args.get("stratify_mode", "coarse"),
        "age_bin_width": train_args.get("age_bin_width", 10),
        "min_age": train_args.get("min_age", 18),
        "max_age": train_args.get("max_age", 100),
        "young_max": train_args.get("young_max", 35),
        "middle_max": train_args.get("middle_max", 60),
    }

    class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)

    print("使用与训练一致的数据划分参数:")
    for k, v in dataset_params.items():
        print(f"  {k}={v}")

    _, _, test_dataset, split_info = load_age_group_datasets(
        image_dir=image_dir,
        excel_path=excel_path,
        class_names=class_names,
        **dataset_params,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model_name = checkpoint.get("model", train_args.get("model", "resnet50"))
    dropout = checkpoint.get("dropout", train_args.get("dropout", 0.6))
    num_classes = int(checkpoint.get("num_classes", len(class_names)))

    model = get_age_group_model(
        model_name=model_name,
        pretrained=False,
        dropout=dropout,
        num_classes=num_classes,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args.output_dir is None:
        output_dir = ensure_dir(checkpoint_path.parent / "age_group_eval")
    else:
        output_dir = ensure_dir(Path(args.output_dir))

    all_true: List[int] = []
    all_pred: List[int] = []
    prediction_records: List[dict] = []

    sample_idx = 0
    with torch.no_grad():
        for images, labels, ages in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            labels_np = labels.cpu().numpy()
            preds_np = preds.cpu().numpy()
            probs_np = probs.cpu().numpy()
            ages_np = ages.cpu().numpy()

            for i in range(len(preds_np)):
                image_path = test_dataset.image_paths[sample_idx]
                pred_probs_list = [float(x) for x in probs_np[i].tolist()]
                top2 = sorted(pred_probs_list, reverse=True)[:2]
                true_label = int(labels_np[i])
                pred_label = int(preds_np[i])
                nearest_boundary_distance = float(
                    min(abs(float(ages_np[i]) - dataset_params["young_max"]), abs(float(ages_np[i]) - dataset_params["middle_max"]))
                )
                record = {
                    "filename": Path(image_path).name,
                    "filepath": image_path,
                    "true_age": float(ages_np[i]),
                    "true_label": true_label,
                    "true_label_name": class_names[true_label],
                    "pred_label": pred_label,
                    "pred_label_name": class_names[pred_label],
                    "pred_probs": pred_probs_list,
                    "pred_probs_dict": {class_names[j]: pred_probs_list[j] for j in range(num_classes)},
                    "confidence": float(pred_probs_list[pred_label]),
                    "confidence_margin": float(top2[0] - top2[1]) if len(top2) == 2 else 0.0,
                    "nearest_boundary_distance": nearest_boundary_distance,
                    "is_correct": bool(true_label == pred_label),
                }
                prediction_records.append(record)
                sample_idx += 1

            all_true.extend(labels_np.tolist())
            all_pred.extend(preds_np.tolist())
    cm = confusion_matrix(all_true, all_pred, labels=list(range(num_classes)))

    accuracy = accuracy_score(all_true, all_pred)
    balanced_acc = balanced_accuracy_score(all_true, all_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_true, all_pred, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        all_true, all_pred, average="weighted", zero_division=0
    )

    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        all_true,
        all_pred,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    )

    report_text = classification_report(
        all_true,
        all_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")
    save_confusion_matrix_text(cm, class_names, output_dir / "confusion_matrix.txt")

    predictions_payload = {
        "class_names": class_names,
        "age_boundaries": split_info.get("age_boundaries", {}),
        "predictions": prediction_records,
    }
    save_json(predictions_payload, output_dir / "predictions.json")
    save_predictions_csv(prediction_records, class_names, output_dir / "predictions_readable.csv")

    error_analysis = analyze_misclassification_patterns(prediction_records, class_names)
    save_misclassified_samples(prediction_records, class_names, output_dir / "misclassified_samples.txt")

    per_class_metrics = []
    for idx, name in enumerate(class_names):
        per_class_metrics.append(
            {
                "class_id": int(idx),
                "class_name": name,
                "precision": float(per_class_precision[idx]),
                "recall": float(per_class_recall[idx]),
                "f1": float(per_class_f1[idx]),
                "support": int(per_class_support[idx]),
            }
        )

    split_info_compact = get_compact_split_info(split_info)
    boundary_difficulty = summarize_boundary_difficulty(
        prediction_records, dataset_params["young_max"], dataset_params["middle_max"]
    )
    confidence_buckets = summarize_confidence(prediction_records)

    metrics_payload = {
        "evaluation_info": {
            "checkpoint_path": str(checkpoint_path),
            "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
        },
        "model_config": {
            "architecture": model_name,
            "dropout": float(dropout),
            "num_classes": int(num_classes),
            "best_epoch": int(checkpoint.get("epoch", -1)),
            "best_val_macro_f1": float(checkpoint.get("val_macro_f1", 0.0)),
        },
        "class_names": class_names,
        "age_boundaries": split_info.get("age_boundaries", {}),
        "dataset_split_info": split_info_compact,
        "overall_metrics": {
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_acc),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": {
            "labels": list(range(num_classes)),
            "class_names": class_names,
            "matrix": cm.astype(int).tolist(),
        },
        "error_analysis": error_analysis,
        "readability_analysis": {
            "boundary_difficulty": boundary_difficulty,
            "confidence_buckets": confidence_buckets,
        },
        "readability_outputs": [
            "inference_summary.md",
            "predictions_readable.csv",
            "confusion_matrix.txt",
            "test_metrics_full.json",
        ],
    }

    save_json(metrics_payload, output_dir / "test_metrics.json", ensure_ascii=False)
    metrics_full_payload = copy.deepcopy(metrics_payload)
    metrics_full_payload["dataset_split_info"] = split_info
    save_json(metrics_full_payload, output_dir / "test_metrics_full.json", ensure_ascii=False)
    save_inference_summary(metrics_payload, report_text, output_dir / "inference_summary.md")

    print("\n测试集评估结果")
    print("=" * 60)
    print(f"accuracy:          {accuracy:.4f}")
    print(f"balanced_accuracy: {balanced_acc:.4f}")
    print(f"macro_precision:   {macro_precision:.4f}")
    print(f"macro_recall:      {macro_recall:.4f}")
    print(f"macro_f1:          {macro_f1:.4f}")
    print(f"weighted_f1:       {weighted_f1:.4f}")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print("主报告: inference_summary.md")
    print("紧凑指标: test_metrics.json | 完整指标: test_metrics_full.json")


if __name__ == "__main__":
    parser = create_arg_parser()
    main(parser.parse_args())
