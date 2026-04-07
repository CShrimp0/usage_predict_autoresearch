"""
可视化原始图像与resize后的图像对比
展示训练时的尺寸变换效果
"""

import os
import sys
import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# 设置中文字体支持
def setup_chinese_font():
    """设置支持中文的字体"""
    import matplotlib.font_manager as fm
    
    # 尝试查找系统中可用的中文字体
    chinese_fonts = [
        'WenQuanYi Micro Hei',  # 文泉驿微米黑
        'WenQuanYi Zen Hei',    # 文泉驿正黑
        'Noto Sans CJK SC',     # 思源黑体
        'Noto Sans CJK TC',
        'Source Han Sans CN',   # 思源黑体
        'Microsoft YaHei',      # 微软雅黑
        'SimHei',               # 黑体
        'STHeiti',              # 华文黑体
        'Arial Unicode MS',
    ]
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 查找第一个可用的中文字体
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            plt.rcParams['font.monospace'] = [font, 'DejaVu Sans Mono']  # 等宽字体也用中文
            plt.rcParams['axes.unicode_minus'] = False
            print(f"[OK] 使用中文字体: {font}")
            return font
    
    # 如果没有找到中文字体，尝试使用DejaVu Sans（英文）
    print("[WARNING] 未找到中文字体，使用英文显示")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return None

chinese_font = setup_chinese_font()
plt.rcParams['figure.dpi'] = 150


def get_image_paths(data_dir, num_samples=6):
    """随机获取图像路径"""
    image_paths = []
    
    # 遍历Healthy和Pathological文件夹
    for category in ['Healthy', 'Pathological']:
        image_folder = Path(data_dir) / 'TA' / category / 'Images'
        if image_folder.exists():
            images = list(image_folder.glob('*.png'))
            image_paths.extend(images)
    
    # 随机选择
    if len(image_paths) > num_samples:
        image_paths = random.sample(image_paths, num_samples)
    
    return image_paths


def load_and_resize_image(image_path, target_size=224):
    """加载图像并resize"""
    # 加载原始图像
    original_img = Image.open(image_path).convert('RGB')
    original_size = original_img.size  # (width, height)
    
    # 训练时使用的transform
    resize_transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
    ])
    
    resized_img = resize_transform(original_img)
    
    return original_img, resized_img, original_size


def calculate_info_loss(original_size, target_size):
    """计算信息损失比例"""
    original_pixels = original_size[0] * original_size[1]
    target_pixels = target_size * target_size
    compression_ratio = original_pixels / target_pixels
    info_loss = (1 - 1/compression_ratio) * 100
    return compression_ratio, info_loss


def create_comparison_plot(image_paths, target_size=224, save_path=None):
    """创建美观的对比图"""
    
    num_samples = len(image_paths)
    
    # 创建大图 - 调整布局比例
    fig = plt.figure(figsize=(18, 3.5 * num_samples))
    gs = GridSpec(num_samples + 1, 3, figure=fig, 
                  height_ratios=[1] * num_samples + [0.25],
                  hspace=0.3, wspace=0.25,
                  left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # 统计信息
    all_original_sizes = []
    all_compression_ratios = []
    
    # 为每张图像创建对比
    for idx, img_path in enumerate(image_paths):
        original_img, resized_img, original_size = load_and_resize_image(img_path, target_size)
        compression_ratio, info_loss = calculate_info_loss(original_size, target_size)
        
        all_original_sizes.append(original_size)
        all_compression_ratios.append(compression_ratio)
        
        # 原始图像
        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.imshow(original_img)
        ax1.set_title(f'原始图像 #{idx+1}\n尺寸: {original_size[0]}×{original_size[1]} ({original_size[0]*original_size[1]:,} 像素)', 
                      fontsize=10, fontweight='bold', pad=8)
        ax1.axis('off')
        
        # 添加边框
        for spine in ax1.spines.values():
            spine.set_edgecolor('#2196F3')
            spine.set_linewidth(2)
        ax1.spines['top'].set_visible(True)
        ax1.spines['right'].set_visible(True)
        ax1.spines['bottom'].set_visible(True)
        ax1.spines['left'].set_visible(True)
        
        # Resize后的图像
        ax2 = fig.add_subplot(gs[idx, 1])
        ax2.imshow(resized_img)
        ax2.set_title(f'训练输入尺寸\n尺寸: {target_size}×{target_size} ({target_size*target_size:,} 像素)', 
                      fontsize=10, fontweight='bold', pad=8)
        ax2.axis('off')
        
        # 添加边框
        for spine in ax2.spines.values():
            spine.set_edgecolor('#FF9800')
            spine.set_linewidth(2)
        ax2.spines['top'].set_visible(True)
        ax2.spines['right'].set_visible(True)
        ax2.spines['bottom'].set_visible(True)
        ax2.spines['left'].set_visible(True)
        
        # 统计信息
        ax3 = fig.add_subplot(gs[idx, 2])
        ax3.axis('off')
        
        info_text = (
            f"尺寸变换统计\n"
            f"{'='*28}\n"
            f"原始尺寸: {original_size[0]} × {original_size[1]}\n"
            f"目标尺寸: {target_size} × {target_size}\n"
            f"宽度缩放: {original_size[0]/target_size:.2f}倍\n"
            f"高度缩放: {original_size[1]/target_size:.2f}倍\n"
            f"{'='*28}\n"
            f"像素压缩: {compression_ratio:.2f}倍降低\n"
            f"信息保留: {100-info_loss:.1f}%\n"
            f"信息损失: {info_loss:.1f}%\n"
            f"{'='*28}\n"
        )
        
        # 根据压缩比添加评价
        if compression_ratio < 2:
            info_text += "[评价] 压缩适中\n细节保留较好"
            color = '#4CAF50'
            verdict = "GOOD"
        elif compression_ratio < 5:
            info_text += "[评价] 压缩明显\n部分细节丢失"
            color = '#FF9800'
            verdict = "MEDIUM"
        else:
            info_text += "[评价] 压缩严重\n细节损失较多"
            color = '#F44336'
            verdict = "POOR"
        
        ax3.text(0.05, 0.5, info_text, 
                fontsize=9, 
                verticalalignment='center',
                horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.8', 
                         facecolor=color, 
                         alpha=0.12,
                         edgecolor=color,
                         linewidth=2),
                family='monospace')
    
    # 底部总结统计
    ax_summary = fig.add_subplot(gs[num_samples, :])
    ax_summary.axis('off')
    
    # 计算平均值
    avg_width = np.mean([size[0] for size in all_original_sizes])
    avg_height = np.mean([size[1] for size in all_original_sizes])
    avg_compression = np.mean(all_compression_ratios)
    avg_info_loss = (1 - 1/avg_compression) * 100
    
    # 统计不同尺寸
    unique_sizes = set(all_original_sizes)
    size_distribution = {size: all_original_sizes.count(size) for size in unique_sizes}
    
    summary_text = (
        f"[数据集统计] 基于 {num_samples} 张随机样本\n"
        f"平均原始尺寸: {avg_width:.0f}×{avg_height:.0f}  |  "
        f"目标尺寸: {target_size}×{target_size}  |  "
        f"平均压缩比: {avg_compression:.2f}倍  |  "
        f"平均信息损失: {avg_info_loss:.1f}%\n"
        f"原始尺寸分布: "
    )
    
    for size, count in sorted(size_distribution.items(), key=lambda x: x[1], reverse=True):
        summary_text += f"{size[0]}×{size[1]}({count}张)  "
    
    ax_summary.text(0.5, 0.5, summary_text,
                   fontsize=10,
                   horizontalalignment='center',
                   verticalalignment='center',
                   bbox=dict(boxstyle='round,pad=1.0',
                            facecolor='#E3F2FD',
                            edgecolor='#2196F3',
                            linewidth=2))
    
    # 主标题
    fig.suptitle(f'图像尺寸变换对比分析 - 原始数据 vs 训练输入({target_size}×{target_size})',
                fontsize=14, fontweight='bold', y=0.995)
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[OK] 对比图已保存至: {save_path}")
    
    return fig


def create_detail_comparison(image_path, target_size=224, save_path=None):
    """创建单张图像的细节对比（放大局部区域）"""
    
    original_img, resized_img, original_size = load_and_resize_image(image_path, target_size)
    
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3,
                  height_ratios=[1, 1])
    
    # 原始图像（完整）
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(original_img)
    ax1.set_title(f'原始图像\n{original_size[0]}×{original_size[1]}', 
                  fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 在原始图上标记裁剪区域 - 红色边框，往左上移25%
    rect_color = '#FF0000'  # 红色
    # 计算裁剪区域 - 从中心往左上移25%
    crop_size = min(original_size) // 3
    center_x = (original_size[0] - crop_size) // 2
    center_y = (original_size[1] - crop_size) // 2
    # 往左上移动25%
    offset_x = int(crop_size * 0.25)
    offset_y = int(crop_size * 0.25)
    crop_x = max(0, center_x - offset_x)
    crop_y = max(0, center_y - offset_y)
    
    rect = mpatches.Rectangle((crop_x, crop_y), crop_size, crop_size,
                               linewidth=3, edgecolor=rect_color, 
                               facecolor='none', linestyle='--')
    ax1.add_patch(rect)
    # 删除"放大区域"文字标注
    
    # Resize后的图像（完整）
    ax2 = fig.add_subplot(gs[:, 1])
    ax2.imshow(resized_img)
    ax2.set_title(f'训练输入\n{target_size}×{target_size}', 
                  fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 在resize图上标记相同比例的区域 - 红色边框
    scale_factor = target_size / original_size[0]
    crop_size_resized = int(crop_size * scale_factor)
    crop_x_resized = int(crop_x * scale_factor)
    crop_y_resized = int(crop_y * scale_factor)
    
    rect2 = mpatches.Rectangle((crop_x_resized, crop_y_resized), 
                                crop_size_resized, crop_size_resized,
                                linewidth=3, edgecolor=rect_color, 
                                facecolor='none', linestyle='--')
    ax2.add_patch(rect2)
    
    # 裁剪原始图像的局部
    original_crop = original_img.crop((crop_x, crop_y, 
                                       crop_x + crop_size, 
                                       crop_y + crop_size))
    
    # 裁剪resize图像的局部
    resized_crop = resized_img.crop((crop_x_resized, crop_y_resized,
                                     crop_x_resized + crop_size_resized,
                                     crop_y_resized + crop_size_resized))
    
    # 保存独立的放大小图
    if save_path:
        output_dir = Path(save_path).parent
        # 保存原始图的放大区域
        original_crop_path = output_dir / 'crop_original.png'
        original_crop.save(original_crop_path)
        print(f"[OK] 原始图放大区域已保存至: {original_crop_path}")
        
        # 保存训练输入的放大区域
        resized_crop_path = output_dir / 'crop_resized.png'
        resized_crop.save(resized_crop_path)
        print(f"[OK] 训练输入放大区域已保存至: {resized_crop_path}")
    
    # 显示裁剪的局部（原始）
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(original_crop)
    ax3.set_title(f'原始图局部\n{crop_size}×{crop_size}px', 
                  fontsize=11, fontweight='bold', color='#2196F3')
    ax3.axis('off')
    for spine in ax3.spines.values():
        spine.set_edgecolor('#2196F3')
        spine.set_linewidth(3)
        spine.set_visible(True)
    
    # 显示裁剪的局部（resize后）
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(resized_crop)
    ax4.set_title(f'训练输入局部\n{crop_size_resized}×{crop_size_resized}px', 
                  fontsize=11, fontweight='bold', color='#FF9800')
    ax4.axis('off')
    for spine in ax4.spines.values():
        spine.set_edgecolor('#FF9800')
        spine.set_linewidth(3)
        spine.set_visible(True)
    
    # 统计信息和评价
    ax5 = fig.add_subplot(gs[1, 2:])
    ax5.axis('off')
    
    compression_ratio, info_loss = calculate_info_loss(original_size, target_size)
    
    detail_text = (
        f"细节对比分析\n"
        f"{'='*50}\n"
        f"原始分辨率: {original_size[0]}×{original_size[1]} = {original_size[0]*original_size[1]:,}像素\n"
        f"训练分辨率: {target_size}×{target_size} = {target_size**2:,}像素\n"
        f"{'='*50}\n"
        f"像素压缩比: {compression_ratio:.2f}倍降低\n"
        f"信息保留率: {100-info_loss:.1f}%\n"
        f"信息损失率: {info_loss:.1f}%\n"
        f"{'='*50}\n"
        f"局部区域像素变化: {crop_size}×{crop_size} -> {crop_size_resized}×{crop_size_resized}\n"
        f"局部压缩比: {(crop_size/crop_size_resized)**2:.2f}倍\n"
        f"{'='*50}\n"
    )
    
    # 评估
    if compression_ratio < 3:
        detail_text += "[评估结果] 细节保留良好\n纹理特征基本可辨，适合深度学习训练"
        bg_color = '#E8F5E9'
        edge_color = '#4CAF50'
    elif compression_ratio < 8:
        detail_text += "[评估结果] 细节有所损失\n主要特征仍然清晰，预训练模型可补偿"
        bg_color = '#FFF3E0'
        edge_color = '#FF9800'
    else:
        detail_text += "[评估结果] 细节损失明显\n精细纹理模糊，建议提高输入尺寸"
        bg_color = '#FFEBEE'
        edge_color = '#F44336'
    
    ax5.text(0.5, 0.5, detail_text,
            fontsize=9,
            horizontalalignment='center',
            verticalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round,pad=1.2',
                     facecolor=bg_color,
                     edgecolor=edge_color,
                     linewidth=3))
    
    fig.suptitle('图像细节放大对比 - 评估信息损失',
                fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[OK] 细节对比图已保存至: {save_path}")
    
    return fig


def main():
    # 数据目录（上一级的data文件夹）
    data_dir = Path(__file__).parent.parent.parent / 'data'
    output_dir = Path(__file__).parent.parent / 'outputs' / 'visualization'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("[开始] 图像尺寸对比可视化\n")
    
    # 获取图像路径
    print("[扫描] 数据集...")
    image_paths = get_image_paths(data_dir, num_samples=6)
    print(f"[完成] 找到 {len(image_paths)} 张图像\n")
    
    if len(image_paths) == 0:
        print("[错误] 未找到图像文件")
        print(f"       请检查数据目录: {data_dir}")
        return
    
    # 生成多图对比
    print("[生成] 多图对比...")
    fig1 = create_comparison_plot(
        image_paths,
        target_size=224,
        save_path=output_dir / 'image_size_comparison.png'
    )
    plt.close(fig1)
    
    # 生成单图细节对比
    print("\n[生成] 细节放大对比...")
    fig2 = create_detail_comparison(
        image_paths[0],
        target_size=224,
        save_path=output_dir / 'image_detail_comparison.png'
    )
    plt.close(fig2)
    
    print(f"\n{'='*60}")
    print("[完成] 可视化生成完毕")
    print(f"{'='*60}")
    print(f"\n[输出] 文件位置:")
    print(f"  1. 多图对比: {output_dir / 'image_size_comparison.png'}")
    print(f"  2. 细节对比: {output_dir / 'image_detail_comparison.png'}")
    print(f"\n[说明] 使用指南:")
    print(f"  - 对比图展示了原始图像与224×224训练输入的差异")
    print(f"  - 细节图放大显示了局部纹理的变化")
    print(f"  - 可根据信息损失率判断是否需要更大的输入尺寸")
    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
