"""Feasibility 实验通用工具。"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


DEFAULT_IMAGE_DIR = "/home/szdx/LNX/data/TA/Healthy/Images"
DEFAULT_EXCEL_PATH = "/home/szdx/LNX/data/TA/characteristics.xlsx"
DEFAULT_CLASS_NAMES = ["young", "middle", "old"]


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """固定随机种子，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, output_path: Path | str, ensure_ascii: bool = False) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=ensure_ascii)


def load_age_dict(excel_path: str | Path) -> Dict[str, float]:
    """读取 Excel，构建 subject_id -> age。"""
    df = pd.read_excel(excel_path)

    age_dict: Dict[str, float] = {}

    healthy_df = df[["Healthy", "Unnamed: 1"]].copy()
    healthy_df.columns = ["Number", "Age"]
    healthy_df = healthy_df[1:].dropna()

    for _, row in healthy_df.iterrows():
        try:
            subject_id = str(int(float(row["Number"])))
            age = float(row["Age"])
            age_dict[subject_id] = age
        except (ValueError, TypeError):
            continue

    return age_dict


def extract_subject_id(image_path: str | Path) -> str | None:
    """从文件名中提取 subject_id，格式默认: TAXX_<subject_id>_X.png。"""
    stem = Path(image_path).stem
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    return parts[1]


def age_to_label(age: float, young_max: float = 35.0, middle_max: float = 60.0) -> int:
    """年龄映射到粗粒度类别: 0=young, 1=middle, 2=old。"""
    if age < young_max:
        return 0
    if age < middle_max:
        return 1
    return 2


def label_mapping(class_names: List[str] | None = None) -> Dict[str, str]:
    class_names = class_names or DEFAULT_CLASS_NAMES
    return {str(i): name for i, name in enumerate(class_names)}


def class_distribution(labels: List[int], class_names: List[str] | None = None) -> Dict[str, int]:
    class_names = class_names or DEFAULT_CLASS_NAMES
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=len(class_names))
    return {class_names[i]: int(counts[i]) for i in range(len(class_names))}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
