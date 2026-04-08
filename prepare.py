"""
Fixed data preparation and evaluation utilities for usage_predict autoresearch.

This file is the read-only harness for experiments. During autoresearch, the LLM
should modify `train.py` only.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# Fixed experiment constants
# ---------------------------------------------------------------------------

TIME_BUDGET_SECONDS = 5 * 60
IMAGE_DIR = "/home/szdx/LNX/data/TA/Healthy/Images"
EXCEL_PATH = "/home/szdx/LNX/data/TA/characteristics.xlsx"
OUTPUT_ROOT = "outputs/autoresearch"

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def cfg_to_dict(cfg: Any) -> dict[str, Any]:
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return dict(vars(cfg))
    raise TypeError(f"Unsupported config type: {type(cfg)!r}")


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def make_run_dir(output_root: str | Path = OUTPUT_ROOT) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def extract_subject_id(image_path: str | Path) -> str:
    parts = Path(image_path).stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot extract subject id from filename: {image_path}")
    return parts[1]


def ensure_data_exists(image_dir: str | Path, excel_path: str | Path) -> None:
    image_dir = Path(image_dir)
    excel_path = Path(excel_path)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")


# ---------------------------------------------------------------------------
# Subject metadata
# ---------------------------------------------------------------------------

def _normalize_subject_id(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text.split(".")[0]


def _to_float(value: Any) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return float(numeric)


def _read_subject_block(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    subset = df[columns].copy()
    subset.columns = ["subject_id", "age", "length_cm", "weight_kg", "sex"]
    subset = subset[1:].reset_index(drop=True)

    records: list[dict[str, Any]] = []
    for _, row in subset.iterrows():
        subject_id = _normalize_subject_id(row["subject_id"])
        age = _to_float(row["age"])
        if subject_id is None or age is None:
            continue

        length_cm = _to_float(row["length_cm"])
        weight_kg = _to_float(row["weight_kg"])
        sex = None if pd.isna(row["sex"]) else str(row["sex"]).strip().upper()

        bmi = None
        if length_cm and weight_kg and length_cm > 0:
            bmi = weight_kg / ((length_cm / 100.0) ** 2)
            if bmi < 10 or bmi > 60:
                bmi = None

        records.append(
            {
                "subject_id": subject_id,
                "age": age,
                "sex": sex,
                "bmi": bmi,
            }
        )
    return records


def load_subject_info(excel_path: str | Path) -> dict[str, dict[str, Any]]:
    df = pd.read_excel(excel_path)
    records = _read_subject_block(
        df,
        ["Healthy", "Unnamed: 1", "Unnamed: 2", "Unnamed: 3", "Unnamed: 4"],
    )
    records.extend(
        _read_subject_block(
            df,
            ["Pathological", "Unnamed: 7", "Unnamed: 8", "Unnamed: 9", "Unnamed: 10"],
        )
    )
    subject_info = {record["subject_id"]: record for record in records}
    if not subject_info:
        raise RuntimeError(f"No subject records loaded from {excel_path}")
    return subject_info


def collect_subject_images(
    image_dir: str | Path,
    subject_info: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    image_dir = Path(image_dir)
    image_paths = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
    subject_images: dict[str, list[str]] = defaultdict(list)
    for image_path in image_paths:
        try:
            subject_id = extract_subject_id(image_path)
        except ValueError:
            continue
        if subject_id in subject_info:
            subject_images[subject_id].append(str(image_path))
    if not subject_images:
        raise RuntimeError(f"No matching images found under {image_dir}")
    return dict(subject_images)


# ---------------------------------------------------------------------------
# Auxiliary features
# ---------------------------------------------------------------------------

def compute_image_sharpness(image_array: np.ndarray) -> float:
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if image_array.ndim == 3 else image_array
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_image_skewness(image_array: np.ndarray) -> float:
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if image_array.ndim == 3 else image_array
    return float(skew(gray.flatten()))


def compute_image_intensity(image_array: np.ndarray) -> float:
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if image_array.ndim == 3 else image_array
    return float(gray.mean())


class AuxiliaryFeatureExtractor:
    def __init__(
        self,
        subject_info: dict[str, dict[str, Any]],
        *,
        use_gender: bool,
        use_bmi: bool,
        use_skewness: bool,
        use_intensity: bool,
        use_clarity: bool,
    ) -> None:
        self.subject_info = subject_info
        self.use_gender = use_gender
        self.use_bmi = use_bmi
        self.use_skewness = use_skewness
        self.use_intensity = use_intensity
        self.use_clarity = use_clarity
        self.aux_dim = 0
        if self.use_gender:
            self.aux_dim += 2
        if self.use_bmi:
            self.aux_dim += 1
        if self.use_skewness:
            self.aux_dim += 1
        if self.use_intensity:
            self.aux_dim += 1
        if self.use_clarity:
            self.aux_dim += 1

        self.bmi_mean = 0.0
        self.bmi_std = 1.0
        self.skewness_mean = 0.0
        self.skewness_std = 1.0
        self.intensity_mean = 0.0
        self.intensity_std = 1.0
        self.clarity_mean = 0.0
        self.clarity_std = 1.0

    def get_valid_subject_ids(self) -> list[str]:
        valid_ids = []
        for subject_id, info in self.subject_info.items():
            if self.use_gender and info.get("sex") not in {"M", "F"}:
                continue
            if self.use_bmi and info.get("bmi") is None:
                continue
            valid_ids.append(subject_id)
        return valid_ids

    def set_normalization_params(self, train_subject_ids: list[str], train_image_paths: list[str]) -> None:
        if self.use_bmi:
            values = [self.subject_info[sid]["bmi"] for sid in train_subject_ids if self.subject_info[sid]["bmi"] is not None]
            if values:
                self.bmi_mean = float(np.mean(values))
                self.bmi_std = float(np.std(values) + 1e-8)

        if not any([self.use_skewness, self.use_intensity, self.use_clarity]):
            return

        sample_paths = train_image_paths[: min(1000, len(train_image_paths))]
        skewness_values: list[float] = []
        intensity_values: list[float] = []
        clarity_values: list[float] = []
        for image_path in sample_paths:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            if self.use_skewness:
                skewness_values.append(compute_image_skewness(image_rgb))
            if self.use_intensity:
                intensity_values.append(compute_image_intensity(image_rgb))
            if self.use_clarity:
                clarity_values.append(compute_image_sharpness(image_rgb))

        if self.use_skewness and skewness_values:
            self.skewness_mean = float(np.mean(skewness_values))
            self.skewness_std = float(np.std(skewness_values) + 1e-8)
        if self.use_intensity and intensity_values:
            self.intensity_mean = float(np.mean(intensity_values))
            self.intensity_std = float(np.std(intensity_values) + 1e-8)
        if self.use_clarity and clarity_values:
            self.clarity_mean = float(np.mean(clarity_values))
            self.clarity_std = float(np.std(clarity_values) + 1e-8)

    def extract(self, subject_id: str, image_path: str | Path) -> torch.Tensor | None:
        if self.aux_dim == 0:
            return None
        info = self.subject_info.get(subject_id)
        if info is None:
            return None

        features: list[float] = []
        if self.use_gender:
            sex = info.get("sex")
            if sex not in {"M", "F"}:
                return None
            features.extend([1.0, 0.0] if sex == "M" else [0.0, 1.0])

        if self.use_bmi:
            bmi = info.get("bmi")
            if bmi is None:
                return None
            features.append((float(bmi) - self.bmi_mean) / self.bmi_std)

        if any([self.use_skewness, self.use_intensity, self.use_clarity]):
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                return None
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            if self.use_skewness:
                value = compute_image_skewness(image_rgb)
                features.append((value - self.skewness_mean) / self.skewness_std)
            if self.use_intensity:
                value = compute_image_intensity(image_rgb)
                features.append((value - self.intensity_mean) / self.intensity_std)
            if self.use_clarity:
                value = compute_image_sharpness(image_rgb)
                features.append((value - self.clarity_mean) / self.clarity_std)

        return torch.tensor(features, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Dataset + split logic
# ---------------------------------------------------------------------------

def age_bucket(age: float, bin_width: int) -> str:
    lower = int(age // bin_width) * bin_width
    upper = lower + bin_width
    return f"{lower}-{upper}"


def _can_stratify(labels: list[str]) -> bool:
    counts = Counter(labels)
    return len(counts) > 1 and min(counts.values()) >= 2


def split_subject_ids(
    subject_ids: list[str],
    subject_info: dict[str, dict[str, Any]],
    *,
    test_size: float,
    val_size: float,
    seed: int,
    age_bin_width: int,
) -> tuple[list[str], list[str], list[str]]:
    labels = [age_bucket(subject_info[sid]["age"], age_bin_width) for sid in subject_ids]
    stratify_test = labels if _can_stratify(labels) else None

    train_val_ids, test_ids = train_test_split(
        subject_ids,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify_test,
    )

    train_val_labels = [age_bucket(subject_info[sid]["age"], age_bin_width) for sid in train_val_ids]
    stratify_val = train_val_labels if _can_stratify(train_val_labels) else None
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify_val,
    )
    return list(train_ids), list(val_ids), list(test_ids)


def build_transforms(cfg: Any) -> tuple[transforms.Compose, transforms.Compose]:
    train_ops: list[Any] = [transforms.Resize((cfg.image_size, cfg.image_size))]
    eval_ops: list[Any] = [transforms.Resize((cfg.image_size, cfg.image_size))]

    if getattr(cfg, "rotation_degrees", 0.0) > 0:
        train_ops.append(transforms.RandomRotation(degrees=cfg.rotation_degrees))
    if getattr(cfg, "horizontal_flip_prob", 0.0) > 0:
        train_ops.append(transforms.RandomHorizontalFlip(p=cfg.horizontal_flip_prob))

    brightness = float(getattr(cfg, "brightness_jitter", 0.0))
    contrast = float(getattr(cfg, "contrast_jitter", 0.0))
    saturation = float(getattr(cfg, "saturation_jitter", 0.0))
    hue = float(getattr(cfg, "hue_jitter", 0.0))
    if any(value > 0 for value in [brightness, contrast, saturation, hue]):
        train_ops.append(
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )
        )

    tensor_and_norm = [
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ]
    train_ops.extend(tensor_and_norm)
    eval_ops.extend(tensor_and_norm)
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


class UltrasoundAgeDataset(Dataset):
    def __init__(
        self,
        image_paths: list[str],
        ages: list[float],
        transform: transforms.Compose,
        aux_extractor: AuxiliaryFeatureExtractor | None = None,
    ) -> None:
        self.image_paths = image_paths
        self.ages = ages
        self.transform = transform
        self.aux_extractor = aux_extractor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        image_path = self.image_paths[index]
        age = float(self.ages[index])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        image_tensor = self.transform(image)
        age_tensor = torch.tensor(age, dtype=torch.float32)

        if self.aux_extractor is None or self.aux_extractor.aux_dim == 0:
            return image_tensor, age_tensor

        subject_id = extract_subject_id(image_path)
        aux_features = self.aux_extractor.extract(subject_id, image_path)
        if aux_features is None:
            aux_features = torch.zeros(self.aux_extractor.aux_dim, dtype=torch.float32)
        return image_tensor, aux_features, age_tensor


def build_datasets(cfg: Any) -> tuple[Dataset, Dataset, Dataset, dict[str, Any]]:
    ensure_data_exists(cfg.image_dir, cfg.excel_path)
    subject_info = load_subject_info(cfg.excel_path)
    subject_images = collect_subject_images(cfg.image_dir, subject_info)

    subject_ids = [
        sid
        for sid in sorted(subject_images)
        if cfg.min_age <= subject_info[sid]["age"] <= cfg.max_age
    ]
    if not subject_ids:
        raise RuntimeError("No subjects remain after age filtering")

    aux_extractor = None
    use_aux = bool(getattr(cfg, "use_aux_features", False))
    if use_aux:
        aux_extractor = AuxiliaryFeatureExtractor(
            subject_info,
            use_gender=bool(getattr(cfg, "aux_gender", False)),
            use_bmi=bool(getattr(cfg, "aux_bmi", False)),
            use_skewness=bool(getattr(cfg, "aux_skewness", False)),
            use_intensity=bool(getattr(cfg, "aux_intensity", False)),
            use_clarity=bool(getattr(cfg, "aux_clarity", False)),
        )
        if aux_extractor.aux_dim == 0:
            aux_extractor = None
        else:
            valid_ids = set(aux_extractor.get_valid_subject_ids())
            subject_ids = [sid for sid in subject_ids if sid in valid_ids]
            if not subject_ids:
                raise RuntimeError("No subjects remain after auxiliary-feature filtering")

    train_ids, val_ids, test_ids = split_subject_ids(
        subject_ids,
        subject_info,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        seed=cfg.seed,
        age_bin_width=cfg.age_bin_width,
    )

    def build_split(subject_subset: list[str]) -> tuple[list[str], list[float]]:
        split_paths: list[str] = []
        split_ages: list[float] = []
        for subject_id in subject_subset:
            for image_path in subject_images[subject_id]:
                split_paths.append(image_path)
                split_ages.append(float(subject_info[subject_id]["age"]))
        return split_paths, split_ages

    train_paths, train_ages = build_split(train_ids)
    val_paths, val_ages = build_split(val_ids)
    test_paths, test_ages = build_split(test_ids)

    if aux_extractor is not None:
        aux_extractor.set_normalization_params(train_ids, train_paths)

    train_transform, eval_transform = build_transforms(cfg)
    train_dataset = UltrasoundAgeDataset(train_paths, train_ages, train_transform, aux_extractor)
    val_dataset = UltrasoundAgeDataset(val_paths, val_ages, eval_transform, aux_extractor)
    test_dataset = UltrasoundAgeDataset(test_paths, test_ages, eval_transform, aux_extractor)

    metadata = {
        "aux_dim": 0 if aux_extractor is None else aux_extractor.aux_dim,
        "use_aux_features": aux_extractor is not None and aux_extractor.aux_dim > 0,
        "train_age_mean": float(np.mean(train_ages)),
        "train_age_std": float(np.std(train_ages) + 1e-8),
        "subject_counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "sample_counts": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "age_range": {
            "min": float(min(subject_info[sid]["age"] for sid in subject_ids)),
            "max": float(max(subject_info[sid]["age"] for sid in subject_ids)),
        },
    }
    return train_dataset, val_dataset, test_dataset, metadata


def make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )


# ---------------------------------------------------------------------------
# Fixed evaluation harness
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_regression(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    *,
    use_aux: bool,
) -> dict[str, Any]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for batch in data_loader:
        if use_aux:
            images, aux_features, ages = batch
            images = images.to(device, non_blocking=True)
            aux_features = aux_features.to(device, non_blocking=True)
            outputs = model(images, aux_features)
        else:
            images, ages = batch
            images = images.to(device, non_blocking=True)
            outputs = model(images)
        predictions.append(outputs.detach().cpu().numpy().reshape(-1))
        targets.append(ages.numpy().reshape(-1))

    preds = np.concatenate(predictions, axis=0)
    gold = np.concatenate(targets, axis=0)
    errors = preds - gold
    abs_errors = np.abs(errors)
    mae = float(abs_errors.mean())
    rmse = float(np.sqrt(np.mean(errors**2)))
    return {
        "mae": mae,
        "rmse": rmse,
        "mean_error": float(errors.mean()),
        "median_abs_error": float(np.median(abs_errors)),
        "predictions": preds.tolist(),
        "targets": gold.tolist(),
    }
