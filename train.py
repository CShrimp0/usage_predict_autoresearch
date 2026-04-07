"""
统一训练脚本 - 支持单模型/集成训练、多种损失函数、DDP
使用方法:
    # 单GPU训练（MAE损失）
    python train_mae.py
    
    # 多GPU训练（DDP）
    torchrun --nproc_per_node=6 train_mae.py --batch-size 32
    
    # 使用MSE损失
    python train_mae.py --loss mse
    
    # 256x256分辨率
    python train_mae.py --image-size 256
    
    # 集成训练（6个模型并行）
    python train_mae.py --ensemble --ensemble-models resnet50 efficientnet_b0 efficientnet_b1 convnext mobilenet_v3 regnet
"""
import os
import argparse
import random
import time
from sympy import false
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR, ReduceLROnPlateau
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

from model import get_model


def seed_everything(seed, deterministic=False):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ExponentialMovingAverage:
    """Exponential Moving Average (EMA) for model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    def apply_to(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class RTMHuberLoss(nn.Module):
    """Huber + anti-regression-to-the-mean penalty."""

    def __init__(self, mu, std, lambda_rtm=0.1, delta=1.0, eps=1e-6):
        super().__init__()
        self.mu = float(mu)
        self.std = float(std)
        self.lambda_rtm = float(lambda_rtm)
        self.delta = float(delta)
        self.eps = float(eps)
        self.huber = nn.HuberLoss(delta=self.delta, reduction='mean')

    def forward(self, pred, target):
        pred = pred.float().view(-1)
        target = target.float().view(-1)

        huber_loss = self.huber(pred, target)

        mu_t = pred.new_tensor(self.mu)
        std_t = pred.new_tensor(self.std)
        eps_t = pred.new_tensor(self.eps)

        d_true = torch.abs(target - mu_t)
        d_pred = torch.abs(pred - mu_t)
        shrink_penalty = torch.relu(d_true - d_pred)
        weight = d_true / (std_t + eps_t)
        rtm_penalty = (weight * shrink_penalty).mean()

        return huber_loss + self.lambda_rtm * rtm_penalty


def get_loss_function(loss_type='mae', lambda_rtm=0.1, train_age_mean=None, train_age_std=None, huber_delta=1.0):
    """获取损失函数
    
    Args:
        loss_type: 损失函数类型 ('mae', 'mse', 'smoothl1', 'huber', 'rtm_huber')
    
    Returns:
        损失函数实例
    """
    loss_type = loss_type.lower()
    if loss_type == 'mae' or loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'mse' or loss_type == 'l2':
        return nn.MSELoss()
    elif loss_type == 'smoothl1':
        return nn.SmoothL1Loss()
    elif loss_type == 'huber':
        return nn.HuberLoss(delta=huber_delta)
    elif loss_type == 'rtm_huber':
        if train_age_mean is None or train_age_std is None:
            raise ValueError("rtm_huber 需要提供 train_age_mean 和 train_age_std")
        return RTMHuberLoss(
            mu=train_age_mean,
            std=train_age_std,
            lambda_rtm=lambda_rtm,
            delta=huber_delta
        )
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}。支持: mae, mse, smoothl1, huber, rtm_huber")


def load_dataset_module(image_size=224, use_age_stratify=True, age_bin_width=10,
                       use_multimodal=False, use_gender=False, use_bmi=False,
                       use_skewness=False, use_intensity=False, use_clarity=False):
    """动态加载数据集，支持图像尺寸、年龄分层和多模态配置
    
    Args:
        image_size: 图像尺寸 (224 或 256)
        use_age_stratify: 是否使用年龄分层抽样（默认True）
        age_bin_width: 年龄分组宽度（默认10岁）
        use_multimodal: 是否使用多模态数据集
        use_gender: 是否使用性别特征
        use_bmi: 是否使用BMI特征
        use_skewness: 是否使用偏度特征
        use_intensity: 是否使用平均灰度特征
        use_clarity: 是否使用清晰度特征
    
    Returns:
        配置好的 load_dataset 函数 和 辅助特征维度
    """
    if use_multimodal:
        from dataset import load_multimodal_dataset
        
        def configured_load_dataset(image_dir, excel_path, test_size=0.2, val_size=0.1, random_state=42,
                                   min_age=0, max_age=100):
            train_ds, val_ds, test_ds, aux_dim = load_multimodal_dataset(
                image_dir=image_dir,
                excel_path=excel_path,
                test_size=test_size,
                val_size=val_size,
                random_state=random_state,
                image_size=image_size,
                use_age_stratify=use_age_stratify,
                age_bin_width=age_bin_width,
                use_gender=use_gender,
                use_bmi=use_bmi,
                use_skewness=use_skewness,
                use_intensity=use_intensity,
                use_clarity=use_clarity,
                min_age=min_age,
                max_age=max_age
            )
            return train_ds, val_ds, test_ds, aux_dim
        
        print(f"使用 {image_size}×{image_size} 分辨率" +
              (f"，年龄分层抽样（每{age_bin_width}岁）" if use_age_stratify else "") +
              "，多模态特征")
        
        return configured_load_dataset
    else:
        from dataset import load_dataset as load_dataset_func
        
        def configured_load_dataset(image_dir, excel_path, test_size=0.2, val_size=0.1, random_state=42,
                                   min_age=0, max_age=100):
            train_ds, val_ds, test_ds = load_dataset_func(
                image_dir=image_dir,
                excel_path=excel_path,
                test_size=test_size,
                val_size=val_size,
                random_state=random_state,
                image_size=image_size,
                use_age_stratify=use_age_stratify,
                age_bin_width=age_bin_width,
                min_age=min_age,
                max_age=max_age
            )
            return train_ds, val_ds, test_ds, 0
        
        print(f"使用 {image_size}×{image_size} 分辨率" +
              (f"，年龄分层抽样（每{age_bin_width}岁）" if use_age_stratify else ""))
        
        return configured_load_dataset


def setup_ddp():
    """初始化DDP环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_ddp():
    """清理DDP"""
    if dist.is_initialized():
        dist.destroy_process_group()


def generate_command_line(args):
    """生成完整的命令行用于复现实验（包含所有参数）
    
    Args:
        args: argparse解析的参数对象
    
    Returns:
        str: 格式化的命令行字符串
    """
    # 参数描述映射
    param_descriptions = {
        'image_dir': '图像文件夹路径',
        'excel_path': 'Excel标签文件路径',
        'output_dir': '输出目录',
        'test_size': '测试集比例',
        'val_size': '验证集比例',
        'seed': '随机种子',
        'use_age_stratify': '启用年龄分层抽样',
        'age_bin_width': '年龄分组宽度（岁）',
        'model': '模型架构',
        'loss': '损失函数类型',
        'lambda_rtm': 'RTM惩罚系数（仅rtm_huber）',
        'huber_delta': 'Huber损失delta参数',
        'image_size': '输入图像尺寸',
        'pretrained': '使用ImageNet预训练权重',
        'dropout': 'Dropout比例',
        'epochs': '训练轮数',
        'time_budget_minutes': '固定墙钟训练时间预算（分钟）',
        'batch_size': '批次大小',
        'lr': '学习率',
        'weight_decay': '权重衰减',
        'optimizer': '优化器类型',
        'momentum': 'SGD动量',
        'eta_min': '学习率最小值',
        'scheduler': '学习率调度器',
        'step_size': 'StepLR步长',
        'gamma': 'StepLR衰减系数',
        'lr_patience': 'Plateau调度器耐心值',
        'lr_factor': 'Plateau调度器衰减系数',
        'lr_min_lr': 'Plateau最小学习率',
        'warmup_epochs': '学习率预热轮数',
        'use_ema': '启用EMA',
        'ema_decay': 'EMA衰减系数',
        'max_grad_norm': '梯度裁剪阈值',
        'patience': '早停耐心值',
        'num_workers': '数据加载线程数',
        'no_save': '禁用训练输出保存',
        'no_horizontal_flip': '禁用水平翻转',
        'use_ddp': '使用DDP分布式训练',
        'ensemble': '启用集成训练模式',
        'ensemble_models': '集成训练的模型列表',
        'ensemble_gpus': '集成训练使用的GPU列表',
    }
    
    # 构建命令行（始终包含所有参数）
    cmd_parts = ['python train.py \\']
    
    # 模型参数
    cmd_parts.append(f'    --model {args.model} \\')
    cmd_parts.append(f'        # {param_descriptions.get("model", "")}')
    
    # 训练核心参数
    cmd_parts.append(f'    --batch-size {args.batch_size} \\')
    cmd_parts.append(f'        # {param_descriptions.get("batch_size", "")}')
    cmd_parts.append(f'    --dropout {args.dropout} \\')
    cmd_parts.append(f'        # {param_descriptions.get("dropout", "")}')
    cmd_parts.append(f'    --epochs {args.epochs} \\')
    cmd_parts.append(f'        # {param_descriptions.get("epochs", "")}')
    if hasattr(args, 'time_budget_minutes') and args.time_budget_minutes and args.time_budget_minutes > 0:
        cmd_parts.append(f'    --time-budget-minutes {args.time_budget_minutes} \\')
        cmd_parts.append(f'        # {param_descriptions.get("time_budget_minutes", "")}')
    cmd_parts.append(f'    --lr {args.lr} \\')
    cmd_parts.append(f'        # {param_descriptions.get("lr", "")}')
    cmd_parts.append(f'    --weight-decay {args.weight_decay} \\')
    cmd_parts.append(f'        # {param_descriptions.get("weight_decay", "")}')
    cmd_parts.append(f'    --optimizer {args.optimizer} \\')
    cmd_parts.append(f'        # {param_descriptions.get("optimizer", "")}')
    cmd_parts.append(f'    --momentum {args.momentum} \\')
    cmd_parts.append(f'        # {param_descriptions.get("momentum", "")}')
    cmd_parts.append(f'    --patience {args.patience} \\')
    cmd_parts.append(f'        # {param_descriptions.get("patience", "")}')
    
    # 模型配置
    cmd_parts.append(f'    --image-size {args.image_size} \\')
    cmd_parts.append(f'        # {param_descriptions.get("image_size", "")}')
    cmd_parts.append(f'    --loss {args.loss} \\')
    cmd_parts.append(f'        # {param_descriptions.get("loss", "")}')
    cmd_parts.append(f'    --lambda-rtm {args.lambda_rtm} \\')
    cmd_parts.append(f'        # {param_descriptions.get("lambda_rtm", "")}')
    cmd_parts.append(f'    --huber-delta {args.huber_delta} \\')
    cmd_parts.append(f'        # {param_descriptions.get("huber_delta", "")}')
    
    # 数据增强和分层（年龄分层始终启用）
    cmd_parts.append(f'    --age-bin-width {args.age_bin_width} \\')
    cmd_parts.append(f'        # {param_descriptions.get("age_bin_width", "")}')
    
    # 其他训练参数
    cmd_parts.append(f'    --num-workers {args.num_workers} \\')
    cmd_parts.append(f'        # {param_descriptions.get("num_workers", "")}')
    cmd_parts.append(f'    --seed {args.seed} \\')
    cmd_parts.append(f'        # {param_descriptions.get("seed", "")}')
    
    # 高级参数（始终显示）
    cmd_parts.append(f'    --eta-min {args.eta_min} \\')
    cmd_parts.append(f'        # {param_descriptions.get("eta_min", "")}')
    cmd_parts.append(f'    --warmup-epochs {args.warmup_epochs} \\')
    cmd_parts.append(f'        # {param_descriptions.get("warmup_epochs", "")}')
    cmd_parts.append(f'    --scheduler {args.scheduler} \\')
    cmd_parts.append(f'        # {param_descriptions.get("scheduler", "")}')
    cmd_parts.append(f'    --step-size {args.step_size} \\')
    cmd_parts.append(f'        # {param_descriptions.get("step_size", "")}')
    cmd_parts.append(f'    --gamma {args.gamma} \\')
    cmd_parts.append(f'        # {param_descriptions.get("gamma", "")}')
    cmd_parts.append(f'    --lr-patience {args.lr_patience} \\')
    cmd_parts.append(f'        # {param_descriptions.get("lr_patience", "")}')
    cmd_parts.append(f'    --lr-factor {args.lr_factor} \\')
    cmd_parts.append(f'        # {param_descriptions.get("lr_factor", "")}')
    cmd_parts.append(f'    --lr-min-lr {args.lr_min_lr} \\')
    cmd_parts.append(f'        # {param_descriptions.get("lr_min_lr", "")}')
    cmd_parts.append(f'    --ema-decay {args.ema_decay} \\')
    cmd_parts.append(f'        # {param_descriptions.get("ema_decay", "")}')
    cmd_parts.append(f'    --max-grad-norm {args.max_grad_norm} \\')
    cmd_parts.append(f'        # {param_descriptions.get("max_grad_norm", "")}')
    
    # 数据集参数（始终显示）
    cmd_parts.append(f'    --test-size {args.test_size} \\')
    cmd_parts.append(f'        # {param_descriptions.get("test_size", "")}')
    cmd_parts.append(f'    --val-size {args.val_size} \\')
    cmd_parts.append(f'        # {param_descriptions.get("val_size", "")}')
    
    # 路径参数（非默认才显示）
    if args.image_dir != '/home/szdx/LNX/data/TA/Healthy/Images':
        cmd_parts.append(f'    --image-dir "{args.image_dir}" \\')
        cmd_parts.append(f'        # {param_descriptions.get("image_dir", "")}')
    if args.excel_path != '/home/szdx/LNX/data/TA/characteristics.xlsx':
        cmd_parts.append(f'    --excel-path "{args.excel_path}" \\')
        cmd_parts.append(f'        # {param_descriptions.get("excel_path", "")}')
    if args.output_dir != './outputs':
        cmd_parts.append(f'    --output-dir "{args.output_dir}" \\')
        cmd_parts.append(f'        # {param_descriptions.get("output_dir", "")}')
    
    # Pretrained权重
    if args.pretrained:
        cmd_parts.append(f'    --pretrained \\')
        cmd_parts.append(f'        # {param_descriptions.get("pretrained", "")}')
    
    # 可选的布尔参数
    if hasattr(args, 'no_horizontal_flip') and args.no_horizontal_flip:
        cmd_parts.append(f'    --no-horizontal-flip \\')
        cmd_parts.append(f'        # {param_descriptions.get("no_horizontal_flip", "")}')
    if hasattr(args, 'no_save') and args.no_save:
        cmd_parts.append(f'    --no-save \\')
        cmd_parts.append(f'        # {param_descriptions.get("no_save", "")}')
    if hasattr(args, 'use_ema') and args.use_ema:
        cmd_parts.append(f'    --use-ema \\')
        cmd_parts.append(f'        # {param_descriptions.get("use_ema", "")}')
    
    # DDP参数
    if hasattr(args, 'use_ddp') and args.use_ddp:
        cmd_parts.append(f'    --use-ddp \\')
        cmd_parts.append(f'        # {param_descriptions.get("use_ddp", "")}')
    
    # 集成训练参数
    if hasattr(args, 'ensemble') and args.ensemble:
        cmd_parts.append(f'    --ensemble \\')
        cmd_parts.append(f'        # {param_descriptions.get("ensemble", "")}')
        if args.ensemble_models:
            models_str = ' '.join(args.ensemble_models)
            cmd_parts.append(f'    --ensemble-models {models_str} \\')
            cmd_parts.append(f'        # {param_descriptions.get("ensemble_models", "")}')
        if args.ensemble_gpus:
            gpus_str = ' '.join(map(str, args.ensemble_gpus))
            cmd_parts.append(f'    --ensemble-gpus {gpus_str} \\')
            cmd_parts.append(f'        # {param_descriptions.get("ensemble_gpus", "")}')
    
    # 移除最后一个反斜杠
    if cmd_parts[-1].endswith(' \\'):
        cmd_parts[-1] = cmd_parts[-1][:-2]
    
    return '\n'.join(cmd_parts)


def plot_training_curves(history, output_dir, epoch=None):
    """绘制训练曲线
    
    Args:
        history: 训练历史字典
        output_dir: 输出目录
        epoch: 当前epoch（可选，用于标题显示）
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 设置标题
    if epoch is not None:
        fig.suptitle(f'Training Progress (Epoch {epoch})', fontsize=16, fontweight='bold')
    else:
        fig.suptitle('Training Progress - Final', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Loss曲线
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss (MSE)', fontsize=11)
    axes[0, 0].set_title('Loss Curves', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. MAE曲线
    axes[0, 1].plot(epochs, history['train_mae'], 'b-', label='Train MAE', linewidth=2)
    axes[0, 1].plot(epochs, history['val_mae'], 'r-', label='Val MAE', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('MAE (years)', fontsize=11)
    axes[0, 1].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. RMSE曲线
    axes[0, 2].plot(epochs, history['val_rmse'], 'g-', label='Val RMSE', linewidth=2)
    axes[0, 2].set_xlabel('Epoch', fontsize=11)
    axes[0, 2].set_ylabel('RMSE (years)', fontsize=11)
    axes[0, 2].set_title('Root Mean Square Error', fontsize=12, fontweight='bold')
    axes[0, 2].legend(fontsize=10)
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 学习率曲线
    if 'lr' in history and len(history['lr']) > 0:
        axes[1, 0].plot(epochs, history['lr'], 'purple', label='Learning Rate', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=11)
        axes[1, 0].set_ylabel('Learning Rate', fontsize=11)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        axes[1, 0].set_yscale('log')  # 使用对数刻度
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].axis('off')
        axes[1, 0].text(0.5, 0.5, 'Learning rate\nnot recorded', 
                       ha='center', va='center', fontsize=12, color='gray')
    
    # 5. 最佳指标信息（文本显示）
    axes[1, 1].axis('off')
    best_val_mae = min(history['val_mae'])
    best_epoch_idx = history['val_mae'].index(best_val_mae) + 1
    best_val_rmse = history['val_rmse'][best_epoch_idx - 1]
    
    current_epoch = len(history['train_loss'])
    current_val_mae = history['val_mae'][-1]
    current_val_rmse = history['val_rmse'][-1]
    
    metrics_text = f"""Training Summary:

Current Epoch: {current_epoch}
Current Val MAE: {current_val_mae:.2f} years
Current Val RMSE: {current_val_rmse:.2f} years

━━━━━━━━━━━━━━━━━━━━━━━

Best Performance:
Best Epoch: {best_epoch_idx}
Best Val MAE: {best_val_mae:.2f} years
Best Val RMSE: {best_val_rmse:.2f} years

━━━━━━━━━━━━━━━━━━━━━━━

Improvement:
MAE Reduction: {history['val_mae'][0] - best_val_mae:.2f} years
({(1 - best_val_mae/history['val_mae'][0]) * 100:.1f}% improvement)
"""
    
    axes[1, 1].text(0.1, 0.5, metrics_text, 
                    fontsize=11, 
                    verticalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 6. 训练进度统计（第三列第二行）
    axes[1, 2].axis('off')
    
    # 计算训练进度统计
    if 'lr' in history and len(history['lr']) > 0:
        current_lr = history['lr'][-1]
        initial_lr = history['lr'][0]
        lr_info = f"Current LR: {current_lr:.2e}\nInitial LR: {initial_lr:.2e}"
    else:
        lr_info = "LR: Not recorded"
    
    progress_text = f"""Training Details:

Total Epochs: {len(history['train_loss'])}
{lr_info}

━━━━━━━━━━━━━━━━━━━━━━━

Latest Metrics:
Train Loss: {history['train_loss'][-1]:.4f}
Val Loss: {history['val_loss'][-1]:.4f}
Train MAE: {history['train_mae'][-1]:.2f}
Val MAE: {history['val_mae'][-1]:.2f}
"""
    
    axes[1, 2].text(0.1, 0.5, progress_text,
                    fontsize=10,
                    verticalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, is_main_process,
                max_grad_norm=1.0, use_aux=False, ema=None):
    """训练一个epoch"""
    model.train()
    
    losses = AverageMeter()
    maes = AverageMeter()
    grad_norms = AverageMeter()  # 记录梯度范数
    
    if is_main_process:
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    else:
        pbar = train_loader
    
    for batch in pbar:
        if use_aux:
            images, aux_features, ages = batch
            images = images.to(device)
            aux_features = aux_features.to(device)
            ages = ages.to(device)
            # 前向传播（多模态）
            outputs = model(images, aux_features)
        else:
            images, ages = batch
            images = images.to(device)
            ages = ages.to(device)
            # 前向传播（单模态）
            outputs = model(images)
        
        loss = criterion(outputs, ages)
        
        # 计算MAE
        mae = torch.abs(outputs - ages).mean()
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        grad_norms.update(grad_norm.item())
        
        optimizer.step()
        if ema is not None:
            ema.update(model)
        
        # 更新统计
        batch_size = images.size(0)
        losses.update(loss.item(), batch_size)
        maes.update(mae.item(), batch_size)
        
        # 更新进度条
        if is_main_process:
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'MAE': f'{maes.avg:.2f}',
                'grad': f'{grad_norms.avg:.2f}'
            })
    
    return losses.avg, maes.avg, grad_norms.avg


def validate(model, val_loader, criterion, device, epoch, is_main_process, use_aux=False):
    """验证模型"""
    model.eval()
    
    losses = AverageMeter()
    maes = AverageMeter()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        if is_main_process:
            pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        else:
            pbar = val_loader
        
        for batch in pbar:
            if use_aux:
                images, aux_features, ages = batch
                images = images.to(device)
                aux_features = aux_features.to(device)
                ages = ages.to(device)
                # 前向传播（多模态）
                outputs = model(images, aux_features)
            else:
                images, ages = batch
                images = images.to(device)
                ages = ages.to(device)
                # 前向传播（单模态）
                outputs = model(images)
            
            loss = criterion(outputs, ages)
            
            # 计算MAE
            mae = torch.abs(outputs - ages).mean()
            
            # 更新统计
            batch_size = images.size(0)
            losses.update(loss.item(), batch_size)
            maes.update(mae.item(), batch_size)
            
            # 保存预测结果
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(ages.cpu().numpy())
            
            if is_main_process:
                pbar.set_postfix({
                    'loss': f'{losses.avg:.4f}',
                    'MAE': f'{maes.avg:.2f}'
                })
    
    # 计算RMSE
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    
    return losses.avg, maes.avg, rmse


def train(args, model_name=None, gpu_id=None, is_ensemble=False, reporter=None):
    """主训练函数
    
    Args:
        args: 命令行参数
        model_name: 模型名称（集成训练时使用）
        gpu_id: GPU ID（集成训练时使用）
        is_ensemble: 是否为集成训练模式
        reporter: 可选回调函数 reporter(epoch, val_mae) 用于外部汇报/剪枝
    """
    # 初始化DDP
    rank, world_size, local_rank = setup_ddp()
    is_main_process = (rank == 0)
    
    if not hasattr(args, 'optimizer'):
        args.optimizer = 'adamw'
    if not hasattr(args, 'momentum'):
        args.momentum = 0.9
    if not hasattr(args, 'scheduler'):
        args.scheduler = 'cosine'
    if not hasattr(args, 'step_size'):
        args.step_size = 10
    if not hasattr(args, 'gamma'):
        args.gamma = 0.1
    if not hasattr(args, 'lr_patience'):
        args.lr_patience = 5
    if not hasattr(args, 'lr_factor'):
        args.lr_factor = 0.1
    if not hasattr(args, 'lr_min_lr'):
        args.lr_min_lr = 1e-7
    if not hasattr(args, 'use_ema'):
        args.use_ema = False
    if not hasattr(args, 'ema_decay'):
        args.ema_decay = 0.999
    if not hasattr(args, 'no_save'):
        args.no_save = False
    if not hasattr(args, 'deterministic'):
        args.deterministic = False
    if not hasattr(args, 'time_budget_minutes'):
        args.time_budget_minutes = 0.0
    if not hasattr(args, 'lambda_rtm'):
        args.lambda_rtm = 0.1
    if not hasattr(args, 'huber_delta'):
        args.huber_delta = 1.0

    # 固定随机种子（每个进程使用不同但可复现的seed）
    base_seed = int(args.seed) + int(rank)
    seed_everything(base_seed, deterministic=getattr(args, 'deterministic', False))
    
    # 设置设备
    if gpu_id is not None:
        device = torch.device(f'cuda:{gpu_id}')
    elif world_size > 1:
        device = torch.device(f'cuda:{local_rank}')
        if is_main_process:
            print(f'使用DDP训练，World Size: {world_size}')
            print(f'GPU列表: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if is_main_process:
            print(f'使用设备: {device}')
    
    # 创建带时间戳的输出目录（仅主进程）
    if is_main_process and not getattr(args, 'no_save', False):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(args.output_dir) / f'run_{timestamp}'
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f'输出目录: {output_dir}')
    else:
        output_dir = None
    
    # 同步所有进程
    if world_size > 1:
        dist.barrier()
    
    # 加载数据
    if is_main_process:
        print('加载数据集...')
    
    # 检查是否启用多模态
    use_multimodal = hasattr(args, 'use_aux_features') and args.use_aux_features
    use_gender = hasattr(args, 'aux_gender') and args.aux_gender
    use_bmi = hasattr(args, 'aux_bmi') and args.aux_bmi
    use_skewness = hasattr(args, 'aux_skewness') and args.aux_skewness
    use_intensity = hasattr(args, 'aux_intensity') and args.aux_intensity
    use_clarity = hasattr(args, 'aux_clarity') and args.aux_clarity
    
    # 动态加载数据集模块
    # 年龄分层抽样始终启用（已成为默认最佳实践）
    load_dataset = load_dataset_module(
        args.image_size, True, args.age_bin_width,
        use_multimodal, use_gender, use_bmi, use_skewness, use_intensity, use_clarity
    )
    
    train_dataset, val_dataset, test_dataset, aux_dim = load_dataset(
        args.image_dir, 
        args.excel_path,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
        min_age=args.min_age,
        max_age=args.max_age
    )

    # 使用训练集年龄统计值作为RTM惩罚的固定参考（训练/验证共用）
    train_age_values = np.asarray(getattr(train_dataset, 'ages', []), dtype=np.float32)
    if train_age_values.size == 0:
        raise ValueError('无法从train_dataset提取年龄标签，无法计算RTM统计量')
    train_age_mean = float(np.mean(train_age_values))
    train_age_std = float(np.std(train_age_values))

    if is_main_process:
        print(f'训练集年龄统计: mu={train_age_mean:.4f}, s={train_age_std:.4f}')
    
    # 保存完整配置文件（包含数据集信息）
    if is_main_process and output_dir is not None:
        # 统计数据集中的受试者信息
        from collections import defaultdict
        train_subjects = defaultdict(int)
        val_subjects = defaultdict(int)
        test_subjects = defaultdict(int)
        
        for img_path in train_dataset.image_paths:
            subject_id = Path(img_path).stem.split('_')[1]
            train_subjects[subject_id] += 1
        for img_path in val_dataset.image_paths:
            subject_id = Path(img_path).stem.split('_')[1]
            val_subjects[subject_id] += 1
        for img_path in test_dataset.image_paths:
            subject_id = Path(img_path).stem.split('_')[1]
            test_subjects[subject_id] += 1
        
        optimizer_params = {}
        if args.optimizer == 'adamw':
            optimizer_params = {
                'betas': (0.9, 0.999),
                'eps': 1e-8,
                'amsgrad': False,
            }
        elif args.optimizer == 'sgd':
            optimizer_params = {
                'momentum': args.momentum,
            }
        
        scheduler_params = {}
        if args.scheduler == 'cosine':
            scheduler_params = {
                'T_max': args.epochs,
                'eta_min': args.eta_min,
            }
        elif args.scheduler == 'step':
            scheduler_params = {
                'step_size': args.step_size,
                'gamma': args.gamma,
            }
        elif args.scheduler == 'plateau':
            scheduler_params = {
                'patience': args.lr_patience,
                'factor': args.lr_factor,
                'min_lr': args.lr_min_lr,
            }
        
        config = {
            # ==================== 脚本信息 ====================
            'script_name': 'train.py',
            'script_version': '4.0',
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'description': 'Unified training script with age stratification support',
            
            # ==================== 运行环境 ====================
            'environment': {
                'device': str(device),
                'world_size': world_size,
                'use_ddp': world_size > 1,
                'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'pytorch_version': torch.__version__,
                'python_version': f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}",
            },
            
            # ==================== 数据集配置 ====================
            'dataset': {
                # 数据路径
                'image_dir': args.image_dir,
                'excel_path': args.excel_path,
                
                # 数据统计
                'total_samples': len(train_dataset) + len(val_dataset) + len(test_dataset),
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
                'test_samples': len(test_dataset),
                'train_subjects': len(train_subjects),
                'val_subjects': len(val_subjects),
                'test_subjects': len(test_subjects),
                'total_subjects': len(train_subjects) + len(val_subjects) + len(test_subjects),
                
                # 数据划分策略
                'test_size': args.test_size,
                'val_size': args.val_size,
                'random_seed': args.seed,
                'split_method': 'by_subject_id',
                'data_leakage_prevention': True,
                
                # 分层抽样配置（始终启用，已成为默认最佳实践）
                'use_age_stratify': True,
                'age_bin_width': args.age_bin_width,
                'stratify_description': 'Age-based stratified sampling to ensure balanced age distribution (always enabled)',
            },
            
            # ==================== 数据预处理与增强 ====================
            'preprocessing': {
                # 图像预处理
                'image_size': args.image_size,
                # 数据增强
                'rotation_degrees': 10,  # 训练时使用的随机旋转角度
                'horizontal_flip': True,  # 默认启用水平翻转
                'color_jitter': {
                    'brightness': 0.2,
                    'contrast': 0.2,
                    'saturation': 0.1,
                    'hue': 0.05,
                },
                'normalization': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225],
                    'description': 'ImageNet pretrained statistics',
                },
            },
            
            # ==================== 模型配置 ====================
            'model': {
                'architecture': args.model,
                'pretrained': args.pretrained,
                'dropout': args.dropout,
                'output_dim': 1,
                'task': 'age_regression',
                'description': f'{args.model} with dropout={args.dropout}, pretrained={args.pretrained}',
            },
            
            # ==================== 训练超参数 ====================
            'training': {
                # 损失函数
                'loss_function': args.loss.upper(),
                'loss_name': args.loss.lower(),
                'loss_description': {
                    'MAE': 'Mean Absolute Error (L1 Loss)',
                    'MSE': 'Mean Squared Error (L2 Loss)',
                    'SMOOTHL1': 'Smooth L1 Loss (Huber-like)',
                    'HUBER': 'Huber Loss',
                    'RTM_HUBER': 'Huber + Anti-RTM Shrink Penalty',
                }.get(args.loss.upper(), args.loss.upper()),
                'lambda_rtm': float(args.lambda_rtm),
                'huber_delta': float(args.huber_delta),
                'train_age_mean': float(train_age_mean),
                'train_age_std': float(train_age_std),
                
                # 优化器
                'optimizer': args.optimizer,
                'optimizer_params': optimizer_params,
                
                # 学习率调度
                'lr_scheduler': args.scheduler,
                'scheduler_params': scheduler_params,
                'warmup_epochs': args.warmup_epochs,
                'warmup_description': f'Linear warmup for {args.warmup_epochs} epochs',
                'ema': {
                    'enabled': args.use_ema,
                    'decay': args.ema_decay,
                },
                
                # 训练参数
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'effective_batch_size': args.batch_size * world_size,
                'learning_rate': args.lr,
                'weight_decay': args.weight_decay,
                'num_workers': args.num_workers,
                
                # 正则化与优化技巧
                'max_grad_norm': args.max_grad_norm,
                'gradient_clipping': True,
                'early_stopping': {
                    'enabled': True,
                    'patience': args.patience,
                    'monitor': 'val_mae',
                    'mode': 'min',
                },
                
                # 其他训练设置
                'plot_interval': 10,
                'save_checkpoint_interval': 10,
            },
            
            # ==================== 优化技巧总结 ====================
            'optimizations_summary': {
                'gradient_clipping': f'Enabled (max_norm={args.max_grad_norm})',
                'lr_warmup': f'Enabled ({args.warmup_epochs} epochs)',
                'lr_scheduler': (
                    f'CosineAnnealingLR (eta_min={args.eta_min})' if args.scheduler == 'cosine'
                    else f'StepLR (step_size={args.step_size}, gamma={args.gamma})' if args.scheduler == 'step'
                    else f'ReduceLROnPlateau (patience={args.lr_patience}, factor={args.lr_factor}, min_lr={args.lr_min_lr})'
                    if args.scheduler == 'plateau' else 'None'
                ),
                'early_stopping': f'Enabled (patience={args.patience})',
                'data_augmentation': 'Rotation + ColorJitter + (optional) HorizontalFlip',
                'age_stratification': 'Enabled (always)',
                'regularization': f'Dropout={args.dropout}, WeightDecay={args.weight_decay}, EMA={args.use_ema}',
            },
            
            # ==================== 输出配置 ====================
            'output': {
                'output_dir': args.output_dir,
                'run_name': output_dir.name,
                'saved_files': [
                    'config.json - Full configuration',
                    'command.sh - Executable command to reproduce this run',
                    'history.json - Training history (loss, MAE, RMSE, LR per epoch)',
                    'training_curves.png - Visualization of training progress',
                    'best_model.pth - Best model checkpoint (lowest val_mae)',
                    'top3_epoch_*_mae_*.pth - Top 3 best model checkpoints',
                    'test_results.json - Test set performance and error analysis',
                    'test_predictions.csv - Detailed predictions for each test sample',
                ],
            },
            
            # ==================== 完整参数记录 ====================
            'all_args': vars(args),  # 保存所有命令行参数，确保完全可复现
        }
        
        with open(output_dir / 'config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f'配置已保存: {output_dir / "config.json"}')
        
        # 生成并保存完整的命令行
        command_line = generate_command_line(args)
        command_file = output_dir / 'command.sh'
        with open(command_file, 'w', encoding='utf-8') as f:
            f.write('#!/bin/bash\n')
            f.write('# 此命令可用于复现本次训练\n')
            f.write(f'# 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write('# 使用方法: bash command.sh 或直接复制命令到终端\n\n')
            f.write(command_line)
            f.write('\n')
        # 使脚本可执行
        import stat
        command_file.chmod(command_file.stat().st_mode | stat.S_IEXEC)
        print(f'命令行已保存: {command_file}')
    
    # DataLoader worker种子
    def seed_worker(worker_id):
        worker_seed = base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    generator = torch.Generator()
    generator.manual_seed(base_seed)
    
    # 使用DistributedSampler for DDP
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, 
                             shuffle=(train_sampler is None), num_workers=args.num_workers, pin_memory=True,
                             worker_init_fn=seed_worker, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, 
                           shuffle=False, num_workers=args.num_workers, pin_memory=True,
                           worker_init_fn=seed_worker, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True,
                            worker_init_fn=seed_worker, generator=generator)
    
    # 创建模型
    current_model = model_name if model_name else args.model
    if is_main_process:
        print(f'创建模型: {current_model}')
    
    # 计算辅助特征维度
    aux_dim = 0
    if hasattr(args, 'use_aux_features') and args.use_aux_features:
        if args.aux_gender: aux_dim += 2
        if args.aux_bmi: aux_dim += 1
        if args.aux_skewness: aux_dim += 1
        if args.aux_intensity: aux_dim += 1
        if args.aux_clarity: aux_dim += 1
        if is_main_process:
            print(f'使用辅助特征，总维度: {aux_dim}')
    
    if is_main_process:
        print('准备将模型加载到设备...')
    model = get_model(current_model, pretrained=args.pretrained, dropout=args.dropout,
                     aux_input_dim=aux_dim, aux_hidden_dim=args.aux_hidden_dim if hasattr(args, 'aux_hidden_dim') else 32)
    model = model.to(device)
    if is_main_process:
        print('模型已加载到设备')

    # EMA初始化（仅参数）
    base_model = model.module if isinstance(model, DDP) else model
    ema = None
    if args.use_ema:
        ema = ExponentialMovingAverage(base_model, decay=args.ema_decay)
    
    # 使用DDP进行多GPU训练
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if is_main_process:
            print(f'模型已包装为DDP，将在 {world_size} 个GPU上训练')
    
    # 定义损失函数和优化器
    criterion = get_loss_function(
        args.loss,
        lambda_rtm=args.lambda_rtm,
        train_age_mean=train_age_mean,
        train_age_std=train_age_std,
        huber_delta=args.huber_delta
    )
    if is_main_process:
        loss_name_map = {'mae': 'MAE (L1)', 'mse': 'MSE (L2)', 'smoothl1': 'Smooth L1', 'huber': 'Huber', 'rtm_huber': 'RTM-Huber'}
        print(f'使用损失函数: {loss_name_map.get(args.loss.lower(), args.loss)}')
        rtm_enabled = args.loss.lower() == 'rtm_huber'
        print(f'RTM惩罚启用: {rtm_enabled}')
        print(f'RTM参数: mu={train_age_mean:.4f}, s={train_age_std:.4f}, lambda_rtm={args.lambda_rtm}, huber_delta={args.huber_delta}')
    
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum
        )
    else:
        raise ValueError(f"不支持的优化器: {args.optimizer}")
    
    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,  # 整个训练周期
            eta_min=args.eta_min  # 最小学习率
        )
    elif args.scheduler == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=args.lr_patience,
            factor=args.lr_factor,
            min_lr=args.lr_min_lr
        )
    elif args.scheduler == 'none':
        scheduler = None
    else:
        raise ValueError(f"不支持的学习率调度器: {args.scheduler}")
    
    if is_main_process:
        print(f'\n学习率配置:')
        print(f'  初始学习率: {args.lr:.2e}')
        if scheduler is None:
            print('  学习率调度: None')
        elif args.scheduler == 'cosine':
            print(f'  使用CosineAnnealingLR: T_max={args.epochs}, eta_min={args.eta_min:.2e}')
        elif args.scheduler == 'step':
            print(f'  使用StepLR: step_size={args.step_size}, gamma={args.gamma}')
        elif args.scheduler == 'plateau':
            print(f'  使用ReduceLROnPlateau: patience={args.lr_patience}, factor={args.lr_factor}, min_lr={args.lr_min_lr}')
        print(f'  梯度裁剪: max_norm={args.max_grad_norm}')
        print(f'  Warmup epochs: {args.warmup_epochs}\n')
    
    time_budget_minutes = float(getattr(args, 'time_budget_minutes', 0.0) or 0.0)
    time_budget_seconds = max(0.0, time_budget_minutes * 60.0)
    if is_main_process and time_budget_seconds > 0:
        print(f'固定时间预算: {time_budget_minutes:.1f} 分钟')
    
    # 训练循环
    best_val_mae = float('inf')
    best_epoch = 0
    patience_counter = 0  # 早停计数器
    top3_models = []  # 保存top3模型: [(val_mae, epoch, file_path), ...]
    history = {
        'train_loss': [],
        'train_mae': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': [],
        'lr': []  # 学习率历史
    }
    
    training_start_time = time.time()
    if is_main_process and device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    result = None
    try:
        if is_main_process:
            print('\n开始训练...')
        for epoch in range(1, args.epochs + 1):
            # 设置sampler的epoch（DDP需要）
            if world_size > 1:
                train_sampler.set_epoch(epoch)
            
            # Warmup阶段：线性增加学习率
            if epoch <= args.warmup_epochs:
                warmup_lr = args.lr * epoch / args.warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            # 训练
            use_aux = hasattr(args, 'use_aux_features') and args.use_aux_features and aux_dim > 0
            train_loss, train_mae, avg_grad_norm = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch, is_main_process,
                args.max_grad_norm, use_aux, ema
            )
            
            # 验证
            if ema is not None:
                ema.apply_to(base_model)
            val_loss, val_mae, val_rmse = validate(
                model, val_loader, criterion, device, epoch, is_main_process, use_aux
            )
            if ema is not None:
                ema.restore(base_model)
            
            if reporter is not None:
                reporter(epoch, float(val_mae))
            
            # 更新学习率（warmup后才使用余弦退火）
            if scheduler is not None and epoch > args.warmup_epochs:
                if args.scheduler == 'plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # 记录当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录历史（转换为Python原生float类型）
            history['train_loss'].append(float(train_loss))
            history['train_mae'].append(float(train_mae))
            history['val_loss'].append(float(val_loss))
            history['val_mae'].append(float(val_mae))
            history['val_rmse'].append(float(val_rmse))
            history['lr'].append(float(current_lr))  # 记录学习率
            
            # 打印统计（仅主进程）
            if is_main_process:
                warmup_status = ' [Warmup]' if epoch <= args.warmup_epochs else ''
                print(f'\nEpoch {epoch}/{args.epochs} (lr={current_lr:.2e}, grad={avg_grad_norm:.2f}){warmup_status}:')
                print(f'  Train - Loss: {train_loss:.4f}, MAE: {train_mae:.2f} years')
                print(f'  Val   - Loss: {val_loss:.4f}, MAE: {val_mae:.2f} years, RMSE: {val_rmse:.2f} years')
            
            # 保存最佳模型（仅主进程）
            if is_main_process:
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_epoch = epoch
                    patience_counter = 0  # 重置早停计数器
                    if output_dir is not None:
                        # 保存模型时处理DDP包装
                        model_to_save = model.module if isinstance(model, DDP) else model
                        if ema is not None:
                            ema.apply_to(model_to_save)
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model_to_save.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_mae': val_mae,
                            'val_rmse': val_rmse,
                            'loss_name': args.loss.lower(),
                            'lambda_rtm': float(args.lambda_rtm),
                            'train_age_mean': float(train_age_mean),
                            'train_age_std': float(train_age_std),
                            'args': vars(args)
                        }, output_dir / 'best_model.pth')
                        if ema is not None:
                            ema.restore(model_to_save)
                        print(f'  ✓ 保存最佳模型 (MAE: {val_mae:.2f} years)')
                        
                        # 管理top3模型保存
                        checkpoint_path = output_dir / f'top3_epoch_{epoch}_mae_{val_mae:.3f}.pth'
                        model_to_save = model.module if isinstance(model, DDP) else model
                        if ema is not None:
                            ema.apply_to(model_to_save)
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model_to_save.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_mae': val_mae,
                            'val_rmse': val_rmse,
                            'loss_name': args.loss.lower(),
                            'lambda_rtm': float(args.lambda_rtm),
                            'train_age_mean': float(train_age_mean),
                            'train_age_std': float(train_age_std)
                        }, checkpoint_path)
                        if ema is not None:
                            ema.restore(model_to_save)
                        
                        # 添加到top3列表
                        top3_models.append((val_mae, epoch, checkpoint_path))
                        top3_models.sort(key=lambda x: x[0])  # 按MAE排序
                        
                        # 如果超过3个，删除最差的
                        if len(top3_models) > 3:
                            _, _, path_to_remove = top3_models.pop()
                            if path_to_remove.exists():
                                path_to_remove.unlink()
                        
                        top3_info = ", ".join([f"Epoch{e}={m:.3f}" for m, e, _ in top3_models])
                        print(f'  ✓ 保存到Top3模型 (当前Top3: {top3_info})')
                else:
                    patience_counter += 1
                    print(f'  ⚠ 没有改善 ({patience_counter}/{args.patience})')
                    
                    # 早停检查
                    if patience_counter >= args.patience:
                        print(f'\n早停触发！连续 {args.patience} 个epoch没有改善，停止训练。')
                        print(f'最佳模型: Epoch {best_epoch}, MAE: {best_val_mae:.2f} years')
                        break
                
                # 每10轮绘制训练曲线（不保存模型）
                if output_dir is not None and epoch % 10 == 0:
                    print(f'  ✓ 更新训练曲线图 (Epoch {epoch})')
                    plot_training_curves(history, output_dir, epoch=epoch)
                
                if time_budget_seconds > 0:
                    elapsed = time.time() - training_start_time
                    if elapsed >= time_budget_seconds:
                        print(f'\n达到时间预算（{time_budget_minutes:.1f} min），停止训练。')
                        print(f'最佳模型: Epoch {best_epoch}, MAE: {best_val_mae:.2f} years')
                        break
    
        # 保存训练历史（仅主进程）
        if is_main_process:
            if output_dir is not None:
                with open(output_dir / 'history.json', 'w') as f:
                    json.dump(history, f, indent=2)
                print(f'训练历史已保存: {output_dir / "history.json"}')
                
                # 绘制最终训练曲线
                print('\n绘制最终训练曲线...')
                plot_training_curves(history, output_dir)
                
                print(f'\n训练完成!')
                print(f'最佳模型: Epoch {best_epoch}, Val MAE: {best_val_mae:.2f} years')
                print(f'\n所有结果保存在: {output_dir}')
                print(f'  - 最佳模型: best_model.pth')
                print(f'  - 训练曲线: training_curves.png')
                print(f'  - 训练历史: history.json')
                print(f'  - 配置文件: config.json')
                print(f'  - 命令脚本: command.sh')
            else:
                print(f'\n训练完成!')
                print(f'最佳模型: Epoch {best_epoch}, Val MAE: {best_val_mae:.2f} years')
            
            # ==================== 测试集评估 (已禁用) ====================
            # 注：训练完成后不再自动进行测试集评估，避免测试集过度使用
            # 如需测试，请使用独立的evaluate.py脚本
            """
            print('\n' + '='*60)
            print('在测试集上评估最佳模型...')
            print('='*60)
            
            # 加载最佳模型
            checkpoint = torch.load(output_dir / 'best_model.pth', map_location=device, weights_only=False)
            model_to_load = model.module if isinstance(model, DDP) else model
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # 在测试集上评估
            test_losses = AverageMeter()
            test_maes = AverageMeter()
            all_test_preds = []
            all_test_targets = []
            
            with torch.no_grad():
                for images, ages in tqdm(test_loader, desc='Testing'):
                    images = images.to(device)
                    ages = ages.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, ages)
                    mae = torch.abs(outputs - ages).mean()
                    
                    batch_size = images.size(0)
                    test_losses.update(loss.item(), batch_size)
                    test_maes.update(mae.item(), batch_size)
                    
                    all_test_preds.extend(outputs.cpu().numpy())
                    all_test_targets.extend(ages.cpu().numpy())
            
            # 计算详细指标
            all_test_preds = np.array(all_test_preds)
            all_test_targets = np.array(all_test_targets)
            test_rmse = np.sqrt(np.mean((all_test_preds - all_test_targets) ** 2))
            test_r2 = 1 - np.sum((all_test_targets - all_test_preds)**2) / np.sum((all_test_targets - np.mean(all_test_targets))**2)
            
            # 计算误差分布
            errors = all_test_preds - all_test_targets
            abs_errors = np.abs(errors)
            
            # 保存测试结果
            test_results = {
                'test_performance': {
                    'loss': float(test_losses.avg),
                    'mae': float(test_maes.avg),
                    'rmse': float(test_rmse),
                    'r2_score': float(test_r2),
                },
                'error_distribution': {
                    'mean_error': float(np.mean(errors)),
                    'std_error': float(np.std(errors)),
                    'median_error': float(np.median(errors)),
                    'mean_abs_error': float(np.mean(abs_errors)),
                    'median_abs_error': float(np.median(abs_errors)),
                    'max_error': float(np.max(abs_errors)),
                    'min_error': float(np.min(abs_errors)),
                },
                'error_percentiles': {
                    '25th': float(np.percentile(abs_errors, 25)),
                    '50th': float(np.percentile(abs_errors, 50)),
                    '75th': float(np.percentile(abs_errors, 75)),
                    '90th': float(np.percentile(abs_errors, 90)),
                    '95th': float(np.percentile(abs_errors, 95)),
                    '99th': float(np.percentile(abs_errors, 99)),
                },
                'accuracy_ranges': {
                    'within_3_years': float(np.sum(abs_errors <= 3) / len(abs_errors) * 100),
                    'within_5_years': float(np.sum(abs_errors <= 5) / len(abs_errors) * 100),
                    'within_7_years': float(np.sum(abs_errors <= 7) / len(abs_errors) * 100),
                    'within_10_years': float(np.sum(abs_errors <= 10) / len(abs_errors) * 100),
                },
                'best_model_info': {
                    'epoch': int(checkpoint['epoch']),
                    'val_mae': float(checkpoint['val_mae']),
                    'val_rmse': float(checkpoint['val_rmse']),
                },
            }
            
            with open(output_dir / 'test_results.json', 'w', encoding='utf-8') as f:
                json.dump(test_results, f, indent=2, ensure_ascii=False)
            print(f'\n测试结果已保存: {output_dir / "test_results.json"}')
            
            # 保存预测详情到CSV
            import pandas as pd
            predictions_df = pd.DataFrame({
                'true_age': all_test_targets.flatten(),
                'predicted_age': all_test_preds.flatten(),
                'error': errors.flatten(),
                'abs_error': abs_errors.flatten(),
            })
            predictions_df.to_csv(output_dir / 'test_predictions.csv', index=False)
            print(f'预测详情已保存: {output_dir / "test_predictions.csv"}')
            
            # 打印测试结果摘要
            print('\n' + '='*60)
            print('测试集评估结果:')
            print('='*60)
            print(f'Loss: {test_losses.avg:.4f}')
            print(f'MAE:  {test_maes.avg:.2f} years')
            print(f'RMSE: {test_rmse:.2f} years')
            print(f'R²:   {test_r2:.4f}')
            print('\n误差分布:')
            print(f'  平均误差: {np.mean(errors):.2f} ± {np.std(errors):.2f} years')
            print(f'  中位数绝对误差: {np.median(abs_errors):.2f} years')
            print(f'  最大绝对误差: {np.max(abs_errors):.2f} years')
            print('\n准确度范围:')
            print(f'  ±3年内:  {test_results["accuracy_ranges"]["within_3_years"]:.1f}%')
            print(f'  ±5年内:  {test_results["accuracy_ranges"]["within_5_years"]:.1f}%')
            print(f'  ±7年内:  {test_results["accuracy_ranges"]["within_7_years"]:.1f}%')
            print(f'  ±10年内: {test_results["accuracy_ranges"]["within_10_years"]:.1f}%')
            print('='*60)
            
            print(f'\n训练完成!')
            print(f'最佳模型: Epoch {best_epoch}, Val MAE: {best_val_mae:.2f} years')
            print(f'测试集性能: MAE {test_maes.avg:.2f} years, RMSE {test_rmse:.2f} years')
            print(f'\n所有结果保存在: {output_dir}')
            print(f'  - 最佳模型: best_model.pth')
            print(f'  - 训练曲线: training_curves.png')
            print(f'  - 训练历史: history.json')
            print(f'  - 配置文件: config.json')
            print(f'  - 命令脚本: command.sh')
            """
        
        train_time = time.time() - training_start_time
        peak_vram_mb = 0.0
        if device.type == 'cuda':
            peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if is_main_process:
            print(f'best_val_mae: {best_val_mae:.6f}')
            print(f'training_seconds: {train_time:.1f}')
            print(f'peak_vram_mb: {peak_vram_mb:.1f}')
            print(f'best_epoch: {best_epoch}')
        result = {
            'best_val_mae': float(best_val_mae),
            'best_epoch': int(best_epoch),
            'train_time': float(train_time),
            'peak_vram_mb': float(peak_vram_mb),
            'output_dir': str(output_dir) if output_dir is not None else None
        }
    finally:
        # 清理DDP
        cleanup_ddp()
    
    return result


def create_arg_parser():
    parser = argparse.ArgumentParser(description='训练TA超声图像年龄预测模型（支持DDP）')
    
    # 数据参数
    parser.add_argument('--image-dir', type=str, 
                       default='/home/szdx/LNX/data/TA/Healthy/Images',
                       help='图像文件夹路径')
    parser.add_argument('--excel-path', type=str,
                       default='/home/szdx/LNX/data/TA/characteristics.xlsx',
                       help='Excel标签文件路径')
    parser.add_argument('--output-dir', type=str, 
                       default='./outputs',
                       help='输出目录')
    
    # 数据集划分
    parser.add_argument('--test-size', type=float, default=0.15, help='测试集比例')
    parser.add_argument('--val-size', type=float, default=0.15, help='验证集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--age-bin-width', type=int, default=10,
                       help='年龄分组宽度（岁）- 年龄分层抽样始终启用')
    parser.add_argument('--min-age', type=float, default=18,
                       help='最小年龄（包含），默认18岁')
    parser.add_argument('--max-age', type=float, default=100,
                       help='最大年龄（包含），默认100岁')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='resnet50', 
                       choices=['resnet50', 'efficientnet_b0', 'efficientnet_b1', 'convnext', 'mobilenet_v3', 'regnet'],
                       help='模型架构')
    parser.add_argument('--loss', type=str, default='mae',
                       choices=['mae', 'mse', 'smoothl1', 'huber', 'rtm_huber'],
                       help='损失函数类型')
    parser.add_argument('--lambda-rtm', type=float, default=0.1,
                       help='RTM惩罚系数（仅rtm_huber使用）')
    parser.add_argument('--huber-delta', type=float, default=1.0,
                       help='Huber损失delta参数（huber/rtm_huber）')
    parser.add_argument('--image-size', type=int, default=224,
                       choices=[224, 256],
                       help='输入图像尺寸')
    # parser.add_argument('--pretrained', action='store_true', default=False,
    #                    help='不使用ImageNet预训练权重')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='使用ImageNet预训练权重')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout比例')
    
    # 多模态特征参数
    parser.add_argument('--use-aux-features', action='store_true', 
                       help='启用辅助特征（性别、BMI、图像统计）')
    parser.add_argument('--aux-gender', action='store_true', help='使用性别特征（2-dim）')
    parser.add_argument('--aux-bmi', action='store_true', help='使用BMI特征（1-dim）')
    parser.add_argument('--aux-skewness', action='store_true', help='使用偏度特征（1-dim）')
    parser.add_argument('--aux-intensity', action='store_true', help='使用平均灰度特征（1-dim）')
    parser.add_argument('--aux-clarity', action='store_true', help='使用清晰度特征（1-dim）')
    parser.add_argument('--aux-hidden-dim', type=int, default=32, 
                       help='辅助特征隐藏层维度（默认32）')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=500, help='最大训练轮数')
    parser.add_argument('--patience', type=int, default=100, help='早停耐心值（连续多少轮没有改善就停止）')
    parser.add_argument('--time-budget-minutes', type=float, default=0.0,
                       help='固定墙钟训练时间预算（分钟，<=0表示禁用）')
    parser.add_argument('--batch-size', type=int, default=32, help='每个GPU的批次大尋')
    parser.add_argument('--lr', type=float, default=1e-4, help='初始学习率（默认1e-4，更稳定）')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'], help='优化器类型')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD动量')
    parser.add_argument('--num-workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--no-save', action='store_true', help='禁用训练输出保存')
    parser.add_argument('--deterministic', action='store_true', help='启用确定性训练设置')
    
    # 学习率调度参数
    parser.add_argument('--eta-min', type=float, default=1e-7, help='最小学习率')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup轮数（线性增加lr）')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'plateau', 'none'], help='学习率调度器')
    parser.add_argument('--step-size', type=int, default=10, help='StepLR步长')
    parser.add_argument('--gamma', type=float, default=0.1, help='StepLR衰减系数')
    parser.add_argument('--lr-patience', type=int, default=5, help='Plateau调度器耐心值')
    parser.add_argument('--lr-factor', type=float, default=0.1, help='Plateau调度器衰减系数')
    parser.add_argument('--lr-min-lr', type=float, default=1e-7, help='Plateau最小学习率')
    parser.add_argument('--use-ema', action='store_true', help='启用EMA')
    parser.add_argument('--ema-decay', type=float, default=0.999, help='EMA衰减系数')
    
    # 集成训练参数
    parser.add_argument('--ensemble', action='store_true', help='启用集成训练模式')
    parser.add_argument('--ensemble-models', nargs='+', 
                       default=['resnet50', 'efficientnet_b0', 'efficientnet_b1', 'convnext', 'mobilenet_v3', 'regnet'],
                       help='集成训练的模型列表')
    parser.add_argument('--ensemble-gpus', nargs='+', type=int,
                       default=[0, 1, 2, 3, 4, 5],
                       help='集成训练使用的GPU列表')
    
    return parser


if __name__ == '__main__':
    parser = create_arg_parser()
    args = parser.parse_args()
    
    # 设置随机种子
    seed_everything(args.seed, deterministic=args.deterministic)
    
    # 开始训练
    if args.ensemble:
        # 集成训练模式
        import multiprocessing as mp
        
        print(f"\n{'='*80}")
        print(f"启动集成训练: {len(args.ensemble_models)} 个模型")
        print(f"模型列表: {', '.join(args.ensemble_models)}")
        print(f"GPU分配: {args.ensemble_gpus}")
        print(f"{'='*80}\n")
        
        # 创建进程池
        processes = []
        for i, model_name in enumerate(args.ensemble_models):
            gpu_id = args.ensemble_gpus[i % len(args.ensemble_gpus)]
            p = mp.Process(target=train, args=(args, model_name, gpu_id, True))
            p.start()
            processes.append(p)
        
        # 等待所有进程完成
        for p in processes:
            p.join()
        
        print(f"\n{'='*80}")
        print(f"集成训练完成！")
        print(f"{'='*80}")
    else:
        # 单模型训练
        train(args)
