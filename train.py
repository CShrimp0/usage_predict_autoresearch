"""
Single-GPU autoresearch training script for ultrasound age prediction.

The intended workflow is:
    python train.py > run.log 2>&1
    RUN_FINAL_EVAL=1 python train.py --final-eval > run.log 2>&1

During autoresearch, this is the only file that should be edited.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tv_models
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

import prepare
from usfm_adapter import USFMEncoderAdapter


DEFAULT_USFM_PRETRAINED_PATH = Path("/home/szdx/LNX/usage_predict/pretrained/USFM_latest.pth")
if not DEFAULT_USFM_PRETRAINED_PATH.exists():
    DEFAULT_USFM_PRETRAINED_PATH = None


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    image_dir: str = prepare.IMAGE_DIR
    excel_path: str = prepare.EXCEL_PATH
    output_root: str = prepare.OUTPUT_ROOT

    seed: int = 42
    deterministic: bool = False
    image_size: int = 224
    min_age: float = 18.0
    max_age: float = 100.0
    test_size: float = 0.15
    val_size: float = 0.15
    age_bin_width: int = 10

    model: str = "resnet50"
    pretrained: bool = True
    pretrained_path: str | None = None if DEFAULT_USFM_PRETRAINED_PATH is None else str(DEFAULT_USFM_PRETRAINED_PATH)
    freeze_backbone: bool = False
    usfm_global_pool: str = "auto"
    dropout: float = 0.4125
    aux_hidden_dim: int = 32

    use_aux_features: bool = True
    aux_gender: bool = True
    aux_bmi: bool = True
    aux_skewness: bool = True
    aux_intensity: bool = True
    aux_clarity: bool = True

    rotation_degrees: float = 10.0
    horizontal_flip_prob: float = 0.0
    brightness_jitter: float = 0.2
    contrast_jitter: float = 0.2
    saturation_jitter: float = 0.0
    hue_jitter: float = 0.0

    batch_size: int = 8
    num_workers: int = 8
    epochs: int = 500
    patience: int = 100
    lr: float = 3.893e-05
    weight_decay: float = 4.914e-05
    optimizer: str = "adamw"
    momentum: float = 0.9
    scheduler: str = "plateau"
    eta_min: float = 1e-7
    step_size: int = 20
    gamma: float = 0.5
    lr_patience: int = 4
    lr_factor: float = 0.696
    lr_min: float = 3.36e-06
    warmup_epochs: int = 5
    max_grad_norm: float = 1.0

    loss: str = "mae"
    huber_delta: float = 1.0
    lambda_rtm: float = 0.1
    use_ema: bool = True
    ema_decay: float = 0.997


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int) -> None:
        self.sum += float(value) * n
        self.count += int(n)
        self.avg = self.sum / max(1, self.count)


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup: dict[str, torch.Tensor] = {}

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class RTMHuberLoss(nn.Module):
    def __init__(self, mu: float, std: float, lambda_rtm: float, delta: float) -> None:
        super().__init__()
        self.mu = float(mu)
        self.std = float(std)
        self.lambda_rtm = float(lambda_rtm)
        self.huber = nn.HuberLoss(delta=float(delta), reduction="mean")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float().view(-1)
        target = target.float().view(-1)
        huber_loss = self.huber(pred, target)
        mu = pred.new_tensor(self.mu)
        std = pred.new_tensor(self.std + 1e-8)
        d_true = torch.abs(target - mu)
        d_pred = torch.abs(pred - mu)
        shrink_penalty = torch.relu(d_true - d_pred)
        weight = d_true / std
        return huber_loss + self.lambda_rtm * (weight * shrink_penalty).mean()


def build_loss(cfg: ExperimentConfig, train_age_mean: float, train_age_std: float) -> nn.Module:
    if cfg.loss == "mae":
        return nn.L1Loss()
    if cfg.loss == "mse":
        return nn.MSELoss()
    if cfg.loss == "smoothl1":
        return nn.SmoothL1Loss()
    if cfg.loss == "huber":
        return nn.HuberLoss(delta=cfg.huber_delta)
    if cfg.loss == "rtm_huber":
        return RTMHuberLoss(
            mu=train_age_mean,
            std=train_age_std,
            lambda_rtm=cfg.lambda_rtm,
            delta=cfg.huber_delta,
        )
    raise ValueError(f"Unsupported loss: {cfg.loss}")


def build_optimizer(cfg: ExperimentConfig, model: nn.Module) -> optim.Optimizer:
    if cfg.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))
    if cfg.optimizer == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            momentum=cfg.momentum,
        )
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")


def build_scheduler(cfg: ExperimentConfig, optimizer: optim.Optimizer):
    if cfg.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.eta_min)
    if cfg.scheduler == "step":
        return StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    if cfg.scheduler == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=cfg.lr_patience,
            factor=cfg.lr_factor,
            min_lr=cfg.lr_min,
        )
    if cfg.scheduler == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {cfg.scheduler}")


def count_parameters(model: nn.Module) -> float:
    return sum(param.numel() for param in model.parameters()) / 1_000_000


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    cfg: ExperimentConfig,
    metadata: dict,
    epoch: int,
    val_mae: float,
    val_rmse: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(cfg),
            "metadata": metadata,
            "best_val_mae": float(val_mae),
            "best_val_rmse": float(val_rmse),
        },
        path,
    )


def _load_model_with_weights(factory, weights_enum_name: str, pretrained: bool, weight_member: str = "DEFAULT"):
    if hasattr(tv_models, weights_enum_name):
        weights_enum = getattr(tv_models, weights_enum_name)
        weights = getattr(weights_enum, weight_member) if pretrained else None
        return factory(weights=weights)
    return factory(pretrained=pretrained)


def _freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_backbone(cfg: ExperimentConfig) -> tuple[nn.Module, int]:
    model_name = cfg.model
    pretrained = cfg.pretrained
    if model_name == "resnet50":
        # Pin to V1 so the repo reuses the existing local cache instead of
        # downloading the newer torchvision default V2 checkpoint.
        model = _load_model_with_weights(
            tv_models.resnet50,
            "ResNet50_Weights",
            pretrained,
            weight_member="IMAGENET1K_V1",
        )
        features_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, features_dim
    if model_name == "efficientnet_b0":
        model = _load_model_with_weights(tv_models.efficientnet_b0, "EfficientNet_B0_Weights", pretrained)
        features_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return model, features_dim
    if model_name == "efficientnet_b1":
        model = _load_model_with_weights(tv_models.efficientnet_b1, "EfficientNet_B1_Weights", pretrained)
        features_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return model, features_dim
    if model_name == "convnext":
        factory = getattr(tv_models, "convnext_tiny")
        model = _load_model_with_weights(factory, "ConvNeXt_Tiny_Weights", pretrained)
        features_dim = model.classifier[2].in_features
        model.classifier = nn.Identity()
        return model, features_dim
    if model_name == "mobilenet_v3":
        model = _load_model_with_weights(
            tv_models.mobilenet_v3_large,
            "MobileNet_V3_Large_Weights",
            pretrained,
        )
        features_dim = model.classifier[3].in_features
        model.classifier = nn.Identity()
        return model, features_dim
    if model_name == "regnet":
        model = _load_model_with_weights(tv_models.regnet_y_400mf, "RegNet_Y_400MF_Weights", pretrained)
        features_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, features_dim
    if model_name == "usfm":
        pretrained_path = None
        if cfg.pretrained_path:
            pretrained_path = Path(cfg.pretrained_path)
            if not pretrained_path.exists():
                raise FileNotFoundError(f"USFM checkpoint not found: {pretrained_path}")
        model = USFMEncoderAdapter(
            image_size=cfg.image_size,
            pretrained_path=None if pretrained_path is None else str(pretrained_path),
            global_pool=cfg.usfm_global_pool,
        )
        if cfg.freeze_backbone:
            _freeze_module(model)
        return model, int(model.feature_dim)
    raise ValueError(f"Unsupported model: {model_name}")


class AgeRegressor(nn.Module):
    def __init__(self, cfg: ExperimentConfig, aux_input_dim: int) -> None:
        super().__init__()
        self.aux_input_dim = aux_input_dim
        self.backbone, image_feature_dim = build_backbone(cfg)

        if aux_input_dim > 0:
            self.aux_branch = nn.Sequential(
                nn.Linear(aux_input_dim, cfg.aux_hidden_dim),
                nn.BatchNorm1d(cfg.aux_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout * 0.6),
                nn.Linear(cfg.aux_hidden_dim, cfg.aux_hidden_dim),
                nn.BatchNorm1d(cfg.aux_hidden_dim),
                nn.ReLU(inplace=True),
            )
            fused_dim = image_feature_dim + cfg.aux_hidden_dim
        else:
            self.aux_branch = None
            fused_dim = image_feature_dim

        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout * 0.5),
            nn.Linear(128, 1),
        )

    def forward(self, images: torch.Tensor, aux_features: torch.Tensor | None = None) -> torch.Tensor:
        image_features = self.backbone(images)
        if self.aux_branch is not None and aux_features is not None:
            aux_repr = self.aux_branch(aux_features)
            fused = torch.cat([image_features, aux_repr], dim=1)
        else:
            fused = image_features
        return self.head(fused).squeeze(-1)


# ---------------------------------------------------------------------------
# Train / validate loops
# ---------------------------------------------------------------------------

def _forward(model: nn.Module, batch: tuple[torch.Tensor, ...], device: torch.device, use_aux: bool):
    if use_aux:
        images, aux_features, ages = batch
        images = images.to(device, non_blocking=True)
        aux_features = aux_features.to(device, non_blocking=True)
        ages = ages.to(device, non_blocking=True)
        outputs = model(images, aux_features)
    else:
        images, ages = batch
        images = images.to(device, non_blocking=True)
        ages = ages.to(device, non_blocking=True)
        outputs = model(images)
    return outputs, ages


def train_one_epoch(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    cfg: ExperimentConfig,
    *,
    use_aux: bool,
    ema: ExponentialMovingAverage | None,
) -> tuple[float, float]:
    model.train()
    loss_meter = AverageMeter()
    mae_meter = AverageMeter()

    for batch in data_loader:
        outputs, ages = _forward(model, batch, device, use_aux)
        loss = criterion(outputs, ages)
        mae = torch.abs(outputs - ages).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        if ema is not None:
            ema.update(model)

        batch_size = ages.shape[0]
        loss_meter.update(loss.item(), batch_size)
        mae_meter.update(mae.item(), batch_size)

    return loss_meter.avg, mae_meter.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    device: torch.device,
    *,
    use_aux: bool,
) -> tuple[float, float, float]:
    model.eval()
    loss_meter = AverageMeter()
    mae_meter = AverageMeter()
    predictions = []
    targets = []

    for batch in data_loader:
        outputs, ages = _forward(model, batch, device, use_aux)
        loss = criterion(outputs, ages)
        mae = torch.abs(outputs - ages).mean()

        batch_size = ages.shape[0]
        loss_meter.update(loss.item(), batch_size)
        mae_meter.update(mae.item(), batch_size)
        predictions.append(outputs.detach().cpu())
        targets.append(ages.detach().cpu())

    preds = torch.cat(predictions).numpy()
    gold = torch.cat(targets).numpy()
    rmse = float(math.sqrt(((preds - gold) ** 2).mean()))
    return loss_meter.avg, mae_meter.avg, rmse


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train usage_predict_autoresearch")
    parser.add_argument(
        "--final-eval",
        action="store_true",
        help="Run explicit final test evaluation after validation-driven training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the default seed for confirmation reruns.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["resnet50", "efficientnet_b0", "efficientnet_b1", "convnext", "mobilenet_v3", "regnet", "usfm"],
        help="Override the default backbone for this run.",
    )
    parser.add_argument(
        "--pretrained-path",
        type=str,
        default=None,
        help="USFM pretrained checkpoint path. Only used when --model usfm.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the image backbone parameters. Primarily useful for USFM.",
    )
    parser.add_argument(
        "--usfm-global-pool",
        type=str,
        default=None,
        choices=["auto", "avg", "token"],
        help="USFM feature pooling mode.",
    )
    parser.add_argument(
        "--disable-aux-features",
        action="store_true",
        help="Run the image-only path by disabling auxiliary features for this run.",
    )
    return parser


def env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def apply_arg_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    if args.model is not None:
        cfg.model = args.model
    if args.pretrained_path is not None:
        cfg.pretrained_path = args.pretrained_path
    if args.freeze_backbone:
        cfg.freeze_backbone = True
    if args.usfm_global_pool is not None:
        cfg.usfm_global_pool = args.usfm_global_pool
    if args.disable_aux_features:
        cfg.use_aux_features = False
    return cfg


def main() -> dict[str, object]:
    args = create_arg_parser().parse_args()
    cfg = apply_arg_overrides(ExperimentConfig(), args)
    train_seed = cfg.seed if args.seed is None else int(args.seed)
    final_eval_enabled = bool(args.final_eval or env_flag("RUN_FINAL_EVAL"))
    prepare.ensure_data_exists(cfg.image_dir, cfg.excel_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = prepare.make_run_dir(cfg.output_root)
    prepare.save_json(run_dir / "config.json", asdict(cfg))

    train_dataset, val_dataset, _test_dataset, metadata = prepare.build_datasets(cfg)
    prepare.set_seed(train_seed, deterministic=cfg.deterministic)
    metadata = dict(metadata)
    metadata["split_seed"] = int(cfg.seed)
    metadata["train_seed"] = int(train_seed)
    use_aux = bool(metadata["use_aux_features"])
    aux_dim = int(metadata["aux_dim"])

    train_loader = prepare.make_dataloader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        seed=train_seed,
    )
    val_loader = prepare.make_dataloader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        seed=train_seed + 1,
    )
    model = AgeRegressor(cfg, aux_input_dim=aux_dim).to(device)
    criterion = build_loss(cfg, metadata["train_age_mean"], metadata["train_age_std"])
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    ema = ExponentialMovingAverage(model, cfg.ema_decay) if cfg.use_ema else None

    history = {
        "train_loss": [],
        "train_mae": [],
        "val_loss": [],
        "val_mae": [],
        "val_rmse": [],
        "lr": [],
    }

    best_val_mae = float("inf")
    best_epoch = 0
    best_checkpoint_path = run_dir / "best_model.pth"
    patience_counter = 0

    print(f"device:            {device}")
    print(f"output_dir:        {run_dir}")
    print(f"run_mode:          {'final_eval' if final_eval_enabled else 'validation_only'}")
    print(f"model:             {cfg.model}")
    if cfg.model == "usfm":
        print(f"pretrained_path:   {cfg.pretrained_path}")
        print(f"freeze_backbone:   {cfg.freeze_backbone}")
        print(f"usfm_global_pool:  {cfg.usfm_global_pool}")
        if cfg.pretrained_path is None:
            print("usfm_init:         checkpoint-only or random-init backbone")
    print(f"split_seed:        {cfg.seed}")
    print(f"train_seed:        {train_seed}")
    print(f"use_aux_features:  {use_aux}")
    print(f"aux_dim:           {aux_dim}")
    print(f"train_samples:     {metadata['sample_counts']['train']}")
    print(f"val_samples:       {metadata['sample_counts']['val']}")
    print(f"test_samples:      {metadata['sample_counts']['test']}")
    print(f"num_params_M:      {count_parameters(model):.2f}")
    print(f"time_budget_sec:   {prepare.TIME_BUDGET_SECONDS}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    overall_start = time.time()
    training_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        if cfg.warmup_epochs > 0 and epoch <= cfg.warmup_epochs:
            warmup_lr = cfg.lr * epoch / cfg.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

        train_loss, train_mae = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            cfg,
            use_aux=use_aux,
            ema=ema,
        )

        if ema is not None:
            ema.apply_to(model)
        val_loss, val_mae, val_rmse = validate(
            model,
            val_loader,
            criterion,
            device,
            use_aux=use_aux,
        )
        if ema is not None:
            ema.restore(model)

        if scheduler is not None and epoch > cfg.warmup_epochs:
            if cfg.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(float(train_loss))
        history["train_mae"].append(float(train_mae))
        history["val_loss"].append(float(val_loss))
        history["val_mae"].append(float(val_mae))
        history["val_rmse"].append(float(val_rmse))
        history["lr"].append(float(current_lr))

        print(
            f"epoch {epoch:03d} | "
            f"train_mae {train_mae:.4f} | "
            f"val_mae {val_mae:.4f} | "
            f"val_rmse {val_rmse:.4f} | "
            f"lr {current_lr:.2e}"
        )

        if val_mae < best_val_mae:
            best_val_mae = float(val_mae)
            best_epoch = int(epoch)
            patience_counter = 0
            if ema is not None:
                ema.apply_to(model)
            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                cfg,
                metadata,
                epoch,
                val_mae,
                val_rmse,
            )
            if ema is not None:
                ema.restore(model)
        else:
            patience_counter += 1

        if patience_counter >= cfg.patience:
            print(f"early_stop:        patience reached at epoch {epoch}")
            break

        if time.time() - training_start >= prepare.TIME_BUDGET_SECONDS:
            print(f"time_stop:         reached {prepare.TIME_BUDGET_SECONDS} seconds")
            break

    training_seconds = time.time() - training_start
    prepare.save_json(run_dir / "history.json", history)

    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    total_seconds = time.time() - overall_start
    metrics = {
        "run_mode": "final_eval" if final_eval_enabled else "validation_only",
        "split_seed": int(cfg.seed),
        "train_seed": int(train_seed),
        "best_val_mae": float(best_val_mae),
        "best_val_rmse": float(checkpoint["best_val_rmse"]),
        "training_seconds": float(training_seconds),
        "total_seconds": float(total_seconds),
        "peak_vram_mb": float(peak_vram_mb),
        "best_epoch": int(best_epoch),
        "num_params_M": float(count_parameters(model)),
        "output_dir": str(run_dir),
        "final_test_mae": None,
        "final_test_rmse": None,
    }

    final_test_summary = None
    if final_eval_enabled:
        import evaluate

        final_test_summary = evaluate.evaluate_checkpoint(
            best_checkpoint_path,
            split="test",
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            device_override=str(device),
            output_json=run_dir / "test_eval.json",
        )
        metrics["final_test_mae"] = float(final_test_summary["prediction_mae"])
        metrics["final_test_rmse"] = float(final_test_summary["prediction_rmse"])

    metrics_payload = {
        "summary": metrics,
        "dataset": metadata,
        "config": asdict(cfg),
    }
    if final_test_summary is not None:
        metrics_payload["final_test"] = final_test_summary
    prepare.save_json(run_dir / "metrics.json", metrics_payload)

    print("---")
    print(f"run_mode:         {metrics['run_mode']}")
    print(f"split_seed:       {metrics['split_seed']}")
    print(f"train_seed:       {metrics['train_seed']}")
    print(f"best_val_mae:     {metrics['best_val_mae']:.6f}")
    print(f"best_val_rmse:    {metrics['best_val_rmse']:.6f}")
    print(f"training_seconds: {metrics['training_seconds']:.1f}")
    print(f"total_seconds:    {metrics['total_seconds']:.1f}")
    print(f"peak_vram_mb:     {metrics['peak_vram_mb']:.1f}")
    print(f"best_epoch:       {metrics['best_epoch']}")
    print(f"num_params_M:     {metrics['num_params_M']:.2f}")
    print(f"output_dir:       {metrics['output_dir']}")
    if final_test_summary is not None:
        print(f"prediction_mae:   {metrics['final_test_mae']:.6f}")
        print(f"prediction_rmse:  {metrics['final_test_rmse']:.6f}")
    else:
        print("final_eval:       disabled")

    return metrics


if __name__ == "__main__":
    main()
