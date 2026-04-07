"""
绘制真实年龄 vs 预测误差的散点图

使用方法:
    python plot_age_error.py --model-path outputs/run_20251226_182738/best_model.pth
"""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from dataset import load_dataset
from model import get_model


def setup_chinese_font():
    """设置中文字体"""
    # 尝试使用 WenQuanYi 字体
    font_candidates = [
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei',
        'SimHei',
        'Microsoft YaHei',
        'DejaVu Sans'
    ]
    
    available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    
    for font in font_candidates:
        if any(font.lower() in f.lower() for f in available_fonts):
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            print(f'使用字体: {font}')
            return
    
    print('警告: 未找到中文字体，可能无法正确显示中文')


def evaluate_model(model, data_loader, device):
    """评估模型并返回预测结果"""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, ages in tqdm(data_loader, desc='评估模型'):
            images = images.to(device)
            ages = ages.to(device)
            
            outputs = model(images)
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(ages.cpu().numpy())
    
    return np.array(all_preds).flatten(), np.array(all_targets).flatten()


def plot_age_vs_error(true_ages, pred_ages, output_path, dataset_name='验证集'):
    """绘制真实年龄 vs 预测误差的散点图
    
    Args:
        true_ages: 真实年龄数组
        pred_ages: 预测年龄数组
        output_path: 输出路径
        dataset_name: 数据集名称
    """
    # 创建 predict_result 文件夹
    result_dir = output_path.parent / 'predict_result'
    result_dir.mkdir(exist_ok=True)
    
    # 计算预测误差
    errors = pred_ages - true_ages
    
    # 计算统计指标
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 真实年龄 vs 预测误差散点图
    ax1 = axes[0, 0]
    scatter = ax1.scatter(true_ages, errors, alpha=0.5, s=30, c=np.abs(errors), 
                         cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=2, label='零误差线')
    ax1.axhline(y=mean_error, color='blue', linestyle='-', linewidth=1.5, label=f'平均误差: {mean_error:.2f}岁')
    ax1.axhline(y=mean_error + std_error, color='red', linestyle=':', linewidth=1, alpha=0.7, label=f'+1 STD: {std_error:.2f}岁')
    ax1.axhline(y=mean_error - std_error, color='red', linestyle=':', linewidth=1, alpha=0.7, label=f'-1 STD')
    
    ax1.set_xlabel('真实年龄 [岁]', fontsize=13, fontweight='bold')
    ax1.set_ylabel('预测误差 [岁]', fontsize=13, fontweight='bold')
    ax1.set_title(f'{dataset_name} - 真实年龄 vs 预测误差', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('绝对误差 [岁]', fontsize=11)
    
    # 2. 误差分布直方图
    ax2 = axes[0, 1]
    n, bins, patches = ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=2, label='零误差')
    ax2.axvline(x=mean_error, color='blue', linestyle='-', linewidth=1.5, label=f'平均: {mean_error:.2f}岁')
    
    ax2.set_xlabel('预测误差 [岁]', fontsize=13, fontweight='bold')
    ax2.set_ylabel('样本数', fontsize=13, fontweight='bold')
    ax2.set_title('预测误差分布', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 真实年龄 vs 预测年龄散点图（对角线）
    ax3 = axes[1, 0]
    ax3.scatter(true_ages, pred_ages, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
    
    # 绘制完美预测线（y=x）
    min_age = min(true_ages.min(), pred_ages.min())
    max_age = max(true_ages.max(), pred_ages.max())
    ax3.plot([min_age, max_age], [min_age, max_age], 'r--', linewidth=2, label='完美预测 (y=x)')
    
    ax3.set_xlabel('真实年龄 [岁]', fontsize=13, fontweight='bold')
    ax3.set_ylabel('预测年龄 [岁]', fontsize=13, fontweight='bold')
    ax3.set_title('真实年龄 vs 预测年龄', fontsize=14, fontweight='bold', pad=15)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')
    
    # 4. 统计信息文本框
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""
评估统计信息

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

数据集: {dataset_name}
样本数: {len(true_ages)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

误差统计:
  平均绝对误差 (MAE): {mae:.3f} 岁
  均方根误差 (RMSE): {rmse:.3f} 岁
  平均误差 (Bias): {mean_error:.3f} 岁
  误差标准差 (STD): {std_error:.3f} 岁

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

年龄范围:
  真实年龄: [{true_ages.min():.1f}, {true_ages.max():.1f}] 岁
  预测年龄: [{pred_ages.min():.1f}, {pred_ages.max():.1f}] 岁

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

误差分位数:
  25%: {np.percentile(np.abs(errors), 25):.2f} 岁
  50% (中位数): {np.percentile(np.abs(errors), 50):.2f} 岁
  75%: {np.percentile(np.abs(errors), 75):.2f} 岁
  95%: {np.percentile(np.abs(errors), 95):.2f} 岁

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

相关系数:
  Pearson R: {np.corrcoef(true_ages, pred_ages)[0, 1]:.4f}
"""
    
    ax4.text(0.1, 0.5, stats_text, 
             fontsize=11, 
             verticalalignment='center',
             fontfamily='sans-serif',  # 使用sans-serif而不是monospace
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\n综合图表已保存: {output_path}')
    
    # === 保存独立的子图到 predict_result 文件夹 ===
    dataset_prefix = 'val' if '验证' in dataset_name else 'test'
    
    # 1. 独立保存：真实年龄 vs 预测误差散点图
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    scatter = ax1.scatter(true_ages, errors, alpha=0.5, s=30, c=np.abs(errors), 
                         cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=2, label='零误差线')
    ax1.axhline(y=mean_error, color='blue', linestyle='-', linewidth=1.5, label=f'平均误差: {mean_error:.2f}岁')
    ax1.axhline(y=mean_error + std_error, color='red', linestyle=':', linewidth=1, alpha=0.7, label=f'+1 STD: {std_error:.2f}岁')
    ax1.axhline(y=mean_error - std_error, color='red', linestyle=':', linewidth=1, alpha=0.7, label=f'-1 STD')
    ax1.set_xlabel('真实年龄 [岁]', fontsize=13, fontweight='bold')
    ax1.set_ylabel('预测误差 [岁]', fontsize=13, fontweight='bold')
    ax1.set_title(f'{dataset_name} - 真实年龄 vs 预测误差', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('绝对误差 [岁]', fontsize=11)
    plt.tight_layout()
    plt.savefig(result_dir / f'{dataset_prefix}_age_vs_error.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 独立保存：误差分布直方图
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    n, bins, patches = ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=2, label='零误差')
    ax2.axvline(x=mean_error, color='blue', linestyle='-', linewidth=1.5, label=f'平均: {mean_error:.2f}岁')
    ax2.set_xlabel('预测误差 [岁]', fontsize=13, fontweight='bold')
    ax2.set_ylabel('样本数', fontsize=13, fontweight='bold')
    ax2.set_title(f'{dataset_name} - 预测误差分布', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(result_dir / f'{dataset_prefix}_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 独立保存：真实年龄 vs 预测年龄散点图
    fig3, ax3 = plt.subplots(figsize=(9, 9))
    ax3.scatter(true_ages, pred_ages, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
    min_age = min(true_ages.min(), pred_ages.min())
    max_age = max(true_ages.max(), pred_ages.max())
    ax3.plot([min_age, max_age], [min_age, max_age], 'r--', linewidth=2, label='完美预测 (y=x)')
    ax3.set_xlabel('真实年龄 [岁]', fontsize=13, fontweight='bold')
    ax3.set_ylabel('预测年龄 [岁]', fontsize=13, fontweight='bold')
    ax3.set_title(f'{dataset_name} - 真实年龄 vs 预测年龄', fontsize=14, fontweight='bold', pad=15)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(result_dir / f'{dataset_prefix}_true_vs_pred.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 独立保存：统计信息文本图
    fig4, ax4 = plt.subplots(figsize=(10, 10))
    ax4.axis('off')
    stats_text = f"""
评估统计信息

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

数据集: {dataset_name}
样本数: {len(true_ages)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

误差统计:
  平均绝对误差 (MAE): {mae:.3f} 岁
  均方根误差 (RMSE): {rmse:.3f} 岁
  平均误差 (Bias): {mean_error:.3f} 岁
  误差标准差 (STD): {std_error:.3f} 岁

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

年龄范围:
  真实年龄: [{true_ages.min():.1f}, {true_ages.max():.1f}] 岁
  预测年龄: [{pred_ages.min():.1f}, {pred_ages.max():.1f}] 岁

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

误差分位数:
  25%: {np.percentile(np.abs(errors), 25):.2f} 岁
  50% (中位数): {np.percentile(np.abs(errors), 50):.2f} 岁
  75%: {np.percentile(np.abs(errors), 75):.2f} 岁
  95%: {np.percentile(np.abs(errors), 95):.2f} 岁

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

相关系数:
  Pearson R: {np.corrcoef(true_ages, pred_ages)[0, 1]:.4f}
"""
    ax4.text(0.1, 0.5, stats_text, 
             fontsize=13, 
             verticalalignment='center',
             fontfamily='sans-serif',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    plt.tight_layout()
    plt.savefig(result_dir / f'{dataset_prefix}_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'独立子图已保存到: {result_dir}/')
    print(f'  - {dataset_prefix}_age_vs_error.png')
    print(f'  - {dataset_prefix}_error_distribution.png')
    print(f'  - {dataset_prefix}_true_vs_pred.png')
    print(f'  - {dataset_prefix}_statistics.png')
    
    plt.close()
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mean_error': float(mean_error),
        'std_error': float(std_error),
        'min_true_age': float(true_ages.min()),
        'max_true_age': float(true_ages.max()),
        'min_pred_age': float(pred_ages.min()),
        'max_pred_age': float(pred_ages.max()),
        'pearson_r': float(np.corrcoef(true_ages, pred_ages)[0, 1])
    }


def main(args):
    """主函数"""
    # 设置中文字体
    setup_chinese_font()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载模型检查点
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f'模型文件不存在: {model_path}')
    
    print(f'\n加载模型: {model_path}')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 从检查点获取参数
    model_args = checkpoint.get('args', {})
    model_name = model_args.get('model', 'resnet50')
    dropout = model_args.get('dropout', 0.5)
    
    print(f'模型架构: {model_name}')
    print(f'训练轮数: {checkpoint["epoch"]}')
    print(f'验证MAE: {checkpoint["val_mae"]:.2f} 岁')
    
    # 创建模型
    model = get_model(model_name, pretrained=False, dropout=dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # 加载数据集
    print('\n加载数据集...')
    train_dataset, val_dataset, test_dataset = load_dataset(
        args.image_dir,
        args.excel_path,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed
    )
    
    # 创建数据加载器
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # 创建输出目录
    output_dir = model_path.parent
    
    # 评估验证集
    print('\n评估验证集...')
    val_preds, val_targets = evaluate_model(model, val_loader, device)
    val_stats = plot_age_vs_error(val_targets, val_preds, 
                                   output_dir / 'age_error_validation.png', 
                                   dataset_name='验证集')
    
    # 评估测试集
    print('\n评估测试集...')
    test_preds, test_targets = evaluate_model(model, test_loader, device)
    test_stats = plot_age_vs_error(test_targets, test_preds, 
                                   output_dir / 'age_error_test.png', 
                                   dataset_name='测试集')
    
    # 保存统计结果
    result_dir = model_path.parent / 'predict_result'
    stats_result = {
        'validation': val_stats,
        'test': test_stats
    }
    
    with open(result_dir / 'statistics_summary.json', 'w', encoding='utf-8') as f:
        json.dump(stats_result, f, indent=2, ensure_ascii=False)
    
    print(f'\n统计汇总已保存: {result_dir / "statistics_summary.json"}')
    print('\n验证集结果:')
    print(f'  MAE: {val_stats["mae"]:.3f} 岁')
    print(f'  RMSE: {val_stats["rmse"]:.3f} 岁')
    print(f'  Pearson R: {val_stats["pearson_r"]:.4f}')
    
    print('\n测试集结果:')
    print(f'  MAE: {test_stats["mae"]:.3f} 岁')
    print(f'  RMSE: {test_stats["rmse"]:.3f} 岁')
    print(f'  Pearson R: {test_stats["pearson_r"]:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='绘制真实年龄 vs 预测误差散点图')
    
    # 模型参数
    parser.add_argument('--model-path', type=str,
                       default='outputs/run_20251226_182738/best_model.pth',
                       help='训练好的模型路径')
    
    # 数据参数
    parser.add_argument('--image-dir', type=str,
                       default='/home/szdx/LNX/data/TA/Healthy/Images',
                       help='图像文件夹路径')
    parser.add_argument('--excel-path', type=str,
                       default='/home/szdx/LNX/data/TA/characteristics.xlsx',
                       help='Excel标签文件路径')
    
    # 数据集划分
    parser.add_argument('--test-size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--val-size', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 评估参数
    parser.add_argument('--batch-size', type=int, default=64, help='批次大小')
    parser.add_argument('--num-workers', type=int, default=8, help='数据加载线程数')
    
    args = parser.parse_args()
    main(args)
