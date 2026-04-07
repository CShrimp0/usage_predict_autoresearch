"""年龄粗分类可行性实验数据集与划分逻辑。"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import warnings

from PIL import Image
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from utils import (
    DEFAULT_CLASS_NAMES,
    age_to_label,
    class_distribution,
    extract_subject_id,
    label_mapping,
    load_age_dict,
)


class AgeGroupDataset(Dataset):
    """年龄粗分类数据集。"""

    def __init__(
        self,
        image_paths: Sequence[str],
        ages: Sequence[float],
        transform=None,
        young_max: float = 35.0,
        middle_max: float = 60.0,
        class_names: Sequence[str] | None = None,
    ) -> None:
        self.image_paths = list(image_paths)
        self.ages = [float(x) for x in ages]
        self.transform = transform
        self.young_max = float(young_max)
        self.middle_max = float(middle_max)
        self.class_names = list(class_names or DEFAULT_CLASS_NAMES)

        self.labels = [age_to_label(age, self.young_max, self.middle_max) for age in self.ages]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        age = self.ages[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long), torch.tensor(age, dtype=torch.float32)


def get_age_bin_label(age: float, bin_width: int = 10) -> str:
    lower = int(age // bin_width) * bin_width
    upper = lower + int(bin_width)
    return f"{lower}-{upper}"


def _build_transform(image_size: int = 224, is_train: bool = True):
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _get_subject_stratify_labels(
    subject_ids: Sequence[str],
    age_dict: Dict[str, float],
    stratify_mode: str,
    young_max: float,
    middle_max: float,
    age_bin_width: int,
) -> List[str] | List[int] | None:
    if stratify_mode == "none":
        return None
    if stratify_mode == "coarse":
        return [age_to_label(age_dict[sid], young_max, middle_max) for sid in subject_ids]
    if stratify_mode == "age_bin":
        return [get_age_bin_label(age_dict[sid], age_bin_width) for sid in subject_ids]
    raise ValueError(f"Unknown stratify_mode: {stratify_mode}")


def _can_stratify(labels: Sequence) -> bool:
    if labels is None:
        return False
    if len(labels) == 0:
        return False
    counter = Counter(labels)
    if len(counter) <= 1:
        return False
    return min(counter.values()) >= 2


def _split_subjects(
    all_subjects: Sequence[str],
    age_dict: Dict[str, float],
    test_size: float,
    val_size: float,
    random_state: int,
    stratify_mode: str,
    young_max: float,
    middle_max: float,
    age_bin_width: int,
) -> Tuple[List[str], List[str], List[str]]:
    labels = _get_subject_stratify_labels(
        all_subjects, age_dict, stratify_mode, young_max, middle_max, age_bin_width
    )

    if _can_stratify(labels):
        train_val_subjects, test_subjects = train_test_split(
            list(all_subjects),
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
            shuffle=True,
        )
    else:
        if stratify_mode != "none":
            warnings.warn(
                "subject-level stratify 无法执行（某些分层样本过少），已回退到随机划分。"
            )
        train_val_subjects, test_subjects = train_test_split(
            list(all_subjects), test_size=test_size, random_state=random_state, shuffle=True
        )

    if val_size <= 0:
        return sorted(train_val_subjects), [], sorted(test_subjects)

    train_val_labels = _get_subject_stratify_labels(
        train_val_subjects, age_dict, stratify_mode, young_max, middle_max, age_bin_width
    )

    if _can_stratify(train_val_labels):
        train_subjects, val_subjects = train_test_split(
            train_val_subjects,
            test_size=val_size,
            random_state=random_state,
            stratify=train_val_labels,
            shuffle=True,
        )
    else:
        if stratify_mode != "none":
            warnings.warn("训练/验证划分 stratify 无法执行，已回退到随机划分。")
        train_subjects, val_subjects = train_test_split(
            train_val_subjects, test_size=val_size, random_state=random_state, shuffle=True
        )

    return sorted(train_subjects), sorted(val_subjects), sorted(test_subjects)


def _gather_split_samples(
    subjects: Sequence[str], subject_images: Dict[str, List[str]], age_dict: Dict[str, float]
) -> Tuple[List[str], List[float]]:
    image_paths: List[str] = []
    ages: List[float] = []
    for sid in subjects:
        for img_path in subject_images[sid]:
            image_paths.append(img_path)
            ages.append(float(age_dict[sid]))
    return image_paths, ages


def load_age_group_datasets(
    image_dir: str,
    excel_path: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    image_size: int = 224,
    stratify_mode: str = "coarse",
    age_bin_width: int = 10,
    min_age: float = 18.0,
    max_age: float = 100.0,
    young_max: float = 35.0,
    middle_max: float = 60.0,
    class_names: Sequence[str] | None = None,
):
    """加载并按 subject-level 划分粗年龄分类数据集。"""
    class_names = list(class_names or DEFAULT_CLASS_NAMES)
    age_dict = load_age_dict(excel_path)

    image_dir_path = Path(image_dir)
    all_image_paths = sorted(
        list(image_dir_path.glob("*.png"))
        + list(image_dir_path.glob("*.jpg"))
        + list(image_dir_path.glob("*.jpeg"))
    )

    subject_images: Dict[str, List[str]] = defaultdict(list)
    for img_path in all_image_paths:
        sid = extract_subject_id(img_path)
        if sid is None:
            continue
        if sid not in age_dict:
            continue
        age = float(age_dict[sid])
        if age < min_age or age > max_age:
            continue
        subject_images[sid].append(str(img_path))

    # 仅保留有图像数据的受试者
    all_subjects = sorted([sid for sid, imgs in subject_images.items() if len(imgs) > 0])
    if len(all_subjects) == 0:
        raise ValueError("未找到符合条件的受试者，请检查 image_dir/excel_path/年龄过滤范围。")

    # 划分
    train_subjects, val_subjects, test_subjects = _split_subjects(
        all_subjects=all_subjects,
        age_dict=age_dict,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        stratify_mode=stratify_mode,
        young_max=young_max,
        middle_max=middle_max,
        age_bin_width=age_bin_width,
    )

    # 收集图像与年龄
    train_paths, train_ages = _gather_split_samples(train_subjects, subject_images, age_dict)
    val_paths, val_ages = _gather_split_samples(val_subjects, subject_images, age_dict)
    test_paths, test_ages = _gather_split_samples(test_subjects, subject_images, age_dict)

    # 变换
    train_transform = _build_transform(image_size=image_size, is_train=True)
    eval_transform = _build_transform(image_size=image_size, is_train=False)

    train_dataset = AgeGroupDataset(
        train_paths,
        train_ages,
        transform=train_transform,
        young_max=young_max,
        middle_max=middle_max,
        class_names=class_names,
    )
    val_dataset = AgeGroupDataset(
        val_paths,
        val_ages,
        transform=eval_transform,
        young_max=young_max,
        middle_max=middle_max,
        class_names=class_names,
    )
    test_dataset = AgeGroupDataset(
        test_paths,
        test_ages,
        transform=eval_transform,
        young_max=young_max,
        middle_max=middle_max,
        class_names=class_names,
    )

    # 统计信息
    total_images = len(train_paths) + len(val_paths) + len(test_paths)
    total_subjects = len(train_subjects) + len(val_subjects) + len(test_subjects)

    split_info = {
        "split_method": "subject_level",
        "data_leakage_prevention": True,
        "stratify_mode": stratify_mode,
        "age_bin_width": int(age_bin_width),
        "random_seed": int(random_state),
        "test_size": float(test_size),
        "val_size": float(val_size),
        "age_filter": {
            "min_age": float(min_age),
            "max_age": float(max_age),
        },
        "age_boundaries": {
            "young_range": f"[{float(min_age):g}, {float(young_max):g})",
            "middle_range": f"[{float(young_max):g}, {float(middle_max):g})",
            "old_range": f"[{float(middle_max):g}, +inf)",
            "young_max": float(young_max),
            "middle_max": float(middle_max),
        },
        "class_names": class_names,
        "label_mapping": label_mapping(class_names),
        "overall": {
            "subjects": int(total_subjects),
            "images": int(total_images),
            "class_distribution": class_distribution(
                train_dataset.labels + val_dataset.labels + test_dataset.labels, class_names
            ),
        },
        "train": {
            "subjects": int(len(train_subjects)),
            "images": int(len(train_paths)),
            "class_distribution": class_distribution(train_dataset.labels, class_names),
        },
        "val": {
            "subjects": int(len(val_subjects)),
            "images": int(len(val_paths)),
            "class_distribution": class_distribution(val_dataset.labels, class_names),
        },
        "test": {
            "subjects": int(len(test_subjects)),
            "images": int(len(test_paths)),
            "class_distribution": class_distribution(test_dataset.labels, class_names),
        },
        "subject_ids": {
            "train": train_subjects,
            "val": val_subjects,
            "test": test_subjects,
        },
    }

    print("\n数据集划分完成（subject-level）:")
    print(f"  训练集: {len(train_paths)} images / {len(train_subjects)} subjects")
    print(f"  验证集: {len(val_paths)} images / {len(val_subjects)} subjects")
    print(f"  测试集: {len(test_paths)} images / {len(test_subjects)} subjects")
    print("  类别映射:", split_info["label_mapping"])
    print("  训练集类别分布:", split_info["train"]["class_distribution"])
    print("  验证集类别分布:", split_info["val"]["class_distribution"])
    print("  测试集类别分布:", split_info["test"]["class_distribution"])

    return train_dataset, val_dataset, test_dataset, split_info
