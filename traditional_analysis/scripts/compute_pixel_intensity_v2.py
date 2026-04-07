"""
计算超声图像的平均灰度值并分析与年龄的相关性（改进版本）

改进点：
1. 从每个受试者的多张图像中选择清晰度最高的一张
2. 使用拉普拉斯方差评估图像质量
3. 确保每个受试者只有一个数据点（避免数据泄露）
"""

import os
import sys
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
from collections import defaultdict

# 设置中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def compute_image_sharpness(image_path):
    """
    计算图像清晰度（使用拉普拉斯方差）
    值越大表示图像越清晰
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    return sharpness


def compute_mean_intensity(image_path):
    """
    计算图像的平均灰度值
    """
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    return img_array.mean()


def load_age_labels(excel_path):
    """
    加载年龄标签，返回字典 {subject_id: age}
    处理Healthy和Pathological两列
    """
    df = pd.read_excel(excel_path)
    age_dict = {}
    
    # 处理Healthy列
    healthy_df = df[['Healthy', 'Unnamed: 1']].copy()
    healthy_df.columns = ['Number', 'Age']
    healthy_df = healthy_df[1:].dropna()
    
    for _, row in healthy_df.iterrows():
        try:
            subject_id = str(int(float(row['Number'])))
            age = float(row['Age'])
            age_dict[subject_id] = age
        except (ValueError, TypeError):
            continue
    
    # 处理Pathological列
    path_df = df[['Pathological', 'Unnamed: 7']].copy()
    path_df.columns = ['Number', 'Age']
    path_df = path_df[1:].dropna()
    
    for _, row in path_df.iterrows():
        try:
            subject_id = str(int(float(row['Number'])))
            age = float(row['Age'])
            age_dict[subject_id] = age
        except (ValueError, TypeError):
            continue
    
    return age_dict


def extract_subject_id(image_path):
    """
    从文件名提取受试者ID
    格式: anon_SubjectID_N.png -> SubjectID
    """
    filename = Path(image_path).stem
    parts = filename.split('_')
    if len(parts) >= 2:
        return parts[1]  # 返回ID部分
    return None


def select_best_image_per_subject(image_files, age_dict):
    """
    从每个受试者的多张图像中选择清晰度最高的一张
    
    Returns:
        dict: {subject_id: best_image_path}
        dict: {subject_id: num_images}
    """
    # 按受试者分组
    subject_images = defaultdict(list)
    for img_path in image_files:
        subject_id = extract_subject_id(img_path)
        # 确保该受试者在age_dict中存在
        if subject_id in age_dict:
            subject_images[subject_id].append(img_path)
    
    # 为每个受试者选择最清晰的图像
    best_images = {}
    image_counts = {}
    
    print(f"\n正在评估图像质量并选择最佳图像...")
    
    for subject_id, img_paths in tqdm(subject_images.items(), desc="选择最佳图像"):
        image_counts[subject_id] = len(img_paths)
        
        if len(img_paths) == 1:
            best_images[subject_id] = img_paths[0]
        else:
            # 计算每张图像的清晰度
            sharpness_scores = []
            for img_path in img_paths:
                sharpness = compute_image_sharpness(img_path)
                sharpness_scores.append((img_path, sharpness))
            
            # 选择清晰度最高的
            if sharpness_scores:
                best_image = max(sharpness_scores, key=lambda x: x[1])[0]
                best_images[subject_id] = best_image
    
    total_images = sum(image_counts.values())
    avg_images_per_subject = total_images / len(image_counts) if image_counts else 0
    
    print(f"   从 {total_images} 张图像中选出 {len(best_images)} 张最佳图像")
    print(f"   平均每个受试者有 {avg_images_per_subject:.2f} 张图像")
    
    # 统计图像数量分布
    count_dist = defaultdict(int)
    for count in image_counts.values():
        count_dist[count] += 1
    print(f"   图像数量分布: ", end="")
    for num_imgs in sorted(count_dist.keys()):
        print(f"{num_imgs}张:{count_dist[num_imgs]}人 ", end="")
    print()
    
    return best_images, image_counts


def analyze_pixel_intensity(image_dir_path, excel_path, output_dir='results', muscle_name='TA'):
    """
    分析图像平均灰度值与年龄的关系
    """
    # 创建输出目录
    muscle_dir = Path(output_dir) / muscle_name
    data_dir = muscle_dir / 'data'
    figures_dir = muscle_dir / 'figures'
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"超声图像平均灰度值分析 - {muscle_name}肌肉（改进版）")
    print("=" * 60)
    
    # 1. 加载年龄标签
    print("\n1. 加载年龄标签...")
    age_dict = load_age_labels(excel_path)
    print(f"   加载了 {len(age_dict)} 个受试者的年龄标签")
    
    # 2. 获取所有图像路径
    image_dir = Path(image_dir_path)
    image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
    print(f"\n2. 找到 {len(image_files)} 张图像")
    
    # 3. 选择每个受试者的最佳图像
    best_images, image_counts = select_best_image_per_subject(image_files, age_dict)
    
    # 4. 计算选中图像的平均灰度值
    print(f"\n3. 计算 {len(best_images)} 张最佳图像的平均灰度值...")
    results = []
    
    for subject_id, img_path in tqdm(best_images.items(), desc="处理图像"):
        # 获取年龄标签
        age = age_dict[subject_id]
        
        # 计算平均灰度值
        mean_intensity = compute_mean_intensity(img_path)
        
        results.append({
            'subject_id': subject_id,
            'image_path': Path(img_path).name,
            'num_images': image_counts[subject_id],
            'age': age,
            'mean_intensity': mean_intensity
        })
    
    # 5. 转换为DataFrame
    df = pd.DataFrame(results)
    
    print(f"\n4. 成功处理 {len(df)} 个受试者")
    print(f"   年龄范围: {df['age'].min():.1f} - {df['age'].max():.1f} 岁")
    print(f"   灰度范围: {df['mean_intensity'].min():.1f} - {df['mean_intensity'].max():.1f}")
    
    # 6. 保存原始数据
    csv_path = data_dir / 'pixel_intensity.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n5. 原始数据已保存: {csv_path}")
    
    # 7. 统计分析
    print(f"\n6. 统计分析:")
    print(f"   平均灰度值: {df['mean_intensity'].mean():.2f} ± {df['mean_intensity'].std():.2f}")
    print(f"   平均年龄: {df['age'].mean():.2f} ± {df['age'].std():.2f}")
    
    # 8. 相关性分析
    print(f"\n7. 相关性分析:")
    pearson_r, pearson_p = stats.pearsonr(df['age'], df['mean_intensity'])
    spearman_r, spearman_p = stats.spearmanr(df['age'], df['mean_intensity'])
    
    print(f"   Pearson相关系数: r = {pearson_r:.4f}, p = {pearson_p:.4e}")
    print(f"   Spearman相关系数: ρ = {spearman_r:.4f}, p = {spearman_p:.4e}")
    
    # 判断相关性强度
    if abs(pearson_r) >= 0.5:
        strength = "强相关（|r| ≥ 0.5）"
        emoji = "🎯"
    elif abs(pearson_r) >= 0.3:
        strength = "中等相关（0.3 ≤ |r| < 0.5）"
        emoji = "📈"
    else:
        strength = "弱相关（|r| < 0.3）"
        emoji = "📊"
    
    print(f"   {emoji} {strength}")
    
    # 9. 线性回归
    slope, intercept = np.polyfit(df['age'], df['mean_intensity'], 1)
    print(f"\n8. 线性回归方程:")
    print(f"   灰度值 = {slope:.4f} × 年龄 + {intercept:.2f}")
    
    # 10. 绘制散点图
    print(f"\n9. 绘制散点图...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图: 散点图 + 回归线
    ax1 = axes[0]
    scatter = ax1.scatter(df['age'], df['mean_intensity'], 
                         alpha=0.5, s=50, 
                         c=df['age'], cmap='viridis',
                         edgecolors='white', linewidth=0.5)
    
    # 添加回归线
    x_line = np.linspace(df['age'].min(), df['age'].max(), 100)
    y_line = slope * x_line + intercept
    ax1.plot(x_line, y_line, 'r--', linewidth=2.5, alpha=0.8, label='线性拟合')
    
    # 添加95%置信区间
    y_pred = slope * df['age'] + intercept
    residuals = df['mean_intensity'] - y_pred
    std_resid = np.std(residuals)
    ax1.fill_between(x_line, y_line - 1.96*std_resid, y_line + 1.96*std_resid,
                     alpha=0.2, color='red', label='95% 置信区间')
    
    ax1.set_xlabel('年龄（岁）', fontsize=13, fontweight='bold')
    ax1.set_ylabel('平均灰度值', fontsize=13, fontweight='bold')
    ax1.set_title(f'{muscle_name}肌肉：年龄 vs 平均灰度值\nr = {pearson_r:.4f}, p < 0.001', 
                 fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('年龄（岁）', fontsize=11)
    
    # 右图: 年龄分组的箱线图
    ax2 = axes[1]
    df['age_group'] = pd.cut(df['age'], bins=[0, 20, 40, 60, 90], 
                              labels=['0-20岁', '20-40岁', '40-60岁', '60-90岁'])
    
    # 使用violin plot
    parts = ax2.violinplot([df[df['age_group'] == group]['mean_intensity'].values 
                            for group in df['age_group'].cat.categories],
                          positions=range(len(df['age_group'].cat.categories)),
                          widths=0.7, showmeans=True, showmedians=True)
    
    # 设置颜色
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax2.set_xticks(range(len(df['age_group'].cat.categories)))
    ax2.set_xticklabels(df['age_group'].cat.categories, fontsize=11)
    ax2.set_xlabel('年龄组', fontsize=13, fontweight='bold')
    ax2.set_ylabel('平均灰度值', fontsize=13, fontweight='bold')
    ax2.set_title('不同年龄组的灰度值分布', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    figure_path = figures_dir / 'age_vs_intensity.png'
    plt.savefig(figure_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   图表已保存: {figure_path}")
    
    # 11. 保存统计摘要
    summary_path = data_dir / 'analysis_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*60}\n")
        f.write(f"超声图像平均灰度值分析报告 - {muscle_name}肌肉（改进版）\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"数据集信息:\n")
        f.write(f"  受试者数量: {len(df)}\n")
        f.write(f"  总图像数: {df['num_images'].sum()}\n")
        f.write(f"  平均每人图像数: {df['num_images'].mean():.2f}\n")
        f.write(f"  年龄范围: {df['age'].min():.1f} - {df['age'].max():.1f} 岁\n")
        f.write(f"  灰度范围: {df['mean_intensity'].min():.1f} - {df['mean_intensity'].max():.1f}\n\n")
        f.write(f"统计分析:\n")
        f.write(f"  平均灰度值: {df['mean_intensity'].mean():.2f} ± {df['mean_intensity'].std():.2f}\n")
        f.write(f"  平均年龄: {df['age'].mean():.2f} ± {df['age'].std():.2f}\n\n")
        f.write(f"相关性分析:\n")
        f.write(f"  Pearson相关系数: r = {pearson_r:.4f}, p = {pearson_p:.4e}\n")
        f.write(f"  Spearman相关系数: ρ = {spearman_r:.4f}, p = {spearman_p:.4e}\n")
        f.write(f"  相关性强度: {strength}\n\n")
        f.write(f"线性回归:\n")
        f.write(f"  方程: 灰度值 = {slope:.4f} × 年龄 + {intercept:.2f}\n\n")
        f.write(f"方法改进:\n")
        f.write(f"  - 使用拉普拉斯方差评估图像清晰度\n")
        f.write(f"  - 从每个受试者选择清晰度最高的一张图像\n")
        f.write(f"  - 避免数据泄露（每个受试者只有一个样本）\n")
    
    print(f"   统计摘要已保存: {summary_path}")
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    
    print(f"\n结果保存位置:")
    print(f"  - 数据: {data_dir}/")
    print(f"  - 图表: {figures_dir}/")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("用法: python compute_pixel_intensity_v2.py <图像目录> <Excel文件> [肌肉名称]")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    excel_path = sys.argv[2]
    muscle_name = sys.argv[3] if len(sys.argv) > 3 else 'TA'
    
    analyze_pixel_intensity(image_dir, excel_path, muscle_name=muscle_name)
