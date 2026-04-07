"""
计算超声图像的纹理特征并分析与年龄的相关性（改进版本）

改进点：
1. 从每个受试者的多张图像中选择清晰度最高的一张
2. 优化可视化效果：更大的图表、渐变配色、置信区间、显著性标记
3. 确保每个受试者只有一个数据点（避免数据泄露）
"""

import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops
from collections import defaultdict

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300


def compute_image_sharpness(img):
    """
    计算图像清晰度（使用拉普拉斯方差）
    值越大表示图像越清晰
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    return sharpness


def load_age_labels(excel_path):
    """
    加载年龄标签，返回字典 {subject_id: age}
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
    """
    subject_images = defaultdict(list)
    for img_path in image_files:
        subject_id = extract_subject_id(img_path)
        if subject_id in age_dict:
            subject_images[subject_id].append(img_path)
    
    best_images = {}
    image_counts = {}
    
    print(f"\n正在评估图像质量并选择最佳图像...")
    
    for subject_id, img_paths in tqdm(subject_images.items(), desc="选择最佳图像"):
        image_counts[subject_id] = len(img_paths)
        
        if len(img_paths) == 1:
            best_images[subject_id] = img_paths[0]
        else:
            sharpness_scores = []
            for img_path in img_paths:
                img = cv2.imread(str(img_path))
                if img is not None:
                    sharpness = compute_image_sharpness(img)
                    sharpness_scores.append((img_path, sharpness))
            
            if sharpness_scores:
                best_image = max(sharpness_scores, key=lambda x: x[1])[0]
                best_images[subject_id] = best_image
    
    total_images = sum(image_counts.values())
    avg_images = total_images / len(image_counts) if image_counts else 0
    
    print(f"   从 {total_images} 张图像中选出 {len(best_images)} 张最佳图像")
    print(f"   平均每个受试者有 {avg_images:.2f} 张图像")
    
    count_dist = defaultdict(int)
    for count in image_counts.values():
        count_dist[count] += 1
    print(f"   图像数量分布: ", end="")
    for num_imgs in sorted(count_dist.keys()):
        print(f"{num_imgs}张:{count_dist[num_imgs]}人 ", end="")
    print()
    
    return best_images, image_counts


def compute_statistical_features(img_array):
    """
    计算统计特征
    """
    return {
        'mean': np.mean(img_array),
        'std': np.std(img_array),
        'skewness': skew(img_array.flatten()),
        'kurtosis': kurtosis(img_array.flatten()),
        'entropy': -np.sum(np.histogram(img_array, bins=256, density=True)[0] * 
                          np.log2(np.histogram(img_array, bins=256, density=True)[0] + 1e-10))
    }


def compute_glcm_features(img_array):
    """
    计算GLCM纹理特征
    """
    img_normalized = ((img_array - img_array.min()) / 
                     (img_array.max() - img_array.min()) * 63).astype(np.uint8)
    
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    glcm = graycomatrix(img_normalized, distances=distances, angles=angles,
                       levels=64, symmetric=True, normed=True)
    
    return {
        'contrast': graycoprops(glcm, 'contrast')[0].mean(),
        'dissimilarity': graycoprops(glcm, 'dissimilarity')[0].mean(),
        'homogeneity': graycoprops(glcm, 'homogeneity')[0].mean(),
        'energy': graycoprops(glcm, 'energy')[0].mean(),
        'correlation': graycoprops(glcm, 'correlation')[0].mean(),
        'ASM': graycoprops(glcm, 'ASM')[0].mean()
    }


def compute_all_features(img_array):
    """
    计算所有纹理特征
    """
    features = {}
    features.update(compute_statistical_features(img_array))
    features.update(compute_glcm_features(img_array))
    return features


def analyze_texture_features(image_dir_path, excel_path, output_dir='results', muscle_name='TA'):
    """
    分析纹理特征与年龄的关系
    """
    muscle_dir = Path(output_dir) / muscle_name
    data_dir = muscle_dir / 'data'
    figures_dir = muscle_dir / 'figures'
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"超声图像纹理特征分析 - {muscle_name}肌肉（改进版）")
    print("=" * 60)
    
    # 1. 加载年龄标签
    print("\n1. 加载年龄标签...")
    age_dict = load_age_labels(excel_path)
    print(f"   加载了 {len(age_dict)} 个受试者的年龄标签")
    
    # 2. 获取所有图像
    image_dir = Path(image_dir_path)
    image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
    print(f"\n2. 找到 {len(image_files)} 张图像")
    
    # 3. 选择最佳图像
    best_images, image_counts = select_best_image_per_subject(image_files, age_dict)
    
    # 4. 计算纹理特征
    print(f"\n3. 计算 {len(best_images)} 张最佳图像的纹理特征...")
    results = []
    
    for subject_id, img_path in tqdm(best_images.items(), desc="处理图像"):
        age = age_dict[subject_id]
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = compute_all_features(gray)
        features['subject_id'] = subject_id
        features['image_path'] = Path(img_path).name
        features['num_images'] = image_counts[subject_id]
        features['age'] = age
        
        results.append(features)
    
    df = pd.DataFrame(results)
    
    print(f"\n4. 成功处理 {len(df)} 个受试者")
    
    # 5. 保存数据
    csv_path = data_dir / 'texture_features.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n5. 原始数据已保存: {csv_path}")
    
    # 6. 相关性分析
    print(f"\n6. 相关性分析:\n")
    
    feature_cols = ['mean', 'std', 'skewness', 'kurtosis', 'entropy',
                    'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    
    corr_results = []
    for feature in feature_cols:
        corr, p_value = stats.pearsonr(df['age'], df[feature])
        corr_results.append({
            'feature': feature,
            'correlation': corr,
            'p_value': p_value,
            'abs_correlation': abs(corr)
        })
    
    corr_df = pd.DataFrame(corr_results).sort_values('abs_correlation', ascending=False)
    
    print("   特征排名（按相关性强度）:")
    for _, row in corr_df.iterrows():
        print(f"   {row['feature']:20s}: r = {row['correlation']:7.4f}, p = {row['p_value']:.2e}")
    
    corr_csv_path = data_dir / 'correlations.csv'
    corr_df.to_csv(corr_csv_path, index=False)
    
    # 7. 绘制优化的可视化图表
    print(f"\n7. 绘制优化的可视化图表...")
    
    # 选择前9个特征
    top_features = corr_df.head(9)
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    axes = axes.flatten()
    
    # 特征中文名称
    feature_names_cn = {
        'mean': '平均灰度',
        'std': '标准差',
        'skewness': '偏度',
        'kurtosis': '峰度',
        'entropy': '熵',
        'contrast': '对比度',
        'dissimilarity': '相异性',
        'homogeneity': '同质性',
        'energy': '能量',
        'correlation': 'GLCM相关性',
        'ASM': '角二阶矩'
    }
    
    # 配色方案
    colors = ['#E63946', '#F77F00', '#06D6A0', '#118AB2', '#073B4C',
              '#8338EC', '#3A86FF', '#FB5607', '#FF006E']
    
    for idx, (_, row) in enumerate(top_features.iterrows()):
        ax = axes[idx]
        feature = row['feature']
        corr_val = row['correlation']
        p_val = row['p_value']
        
        # 散点图 - 使用年龄渐变色
        scatter = ax.scatter(df['age'], df[feature], 
                           alpha=0.4, s=40, 
                           c=df['age'], cmap='viridis',
                           edgecolors='white', linewidth=0.8)
        
        # 拟合线 - 加粗，使用对比色
        z = np.polyfit(df['age'], df[feature], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['age'].min(), df['age'].max(), 100)
        ax.plot(x_line, p(x_line), color='darkred', 
               linewidth=3.5, alpha=0.9, linestyle='--', label='线性拟合')
        
        # 添加95%置信区间
        y_pred = p(df['age'])
        residuals = df[feature] - y_pred
        std_resid = np.std(residuals)
        ax.fill_between(x_line, p(x_line) - 1.96*std_resid, 
                        p(x_line) + 1.96*std_resid,
                        alpha=0.2, color='red', label='95% 置信区间')
        
        # 特征中文名称
        feature_cn = feature_names_cn.get(feature, feature)
        
        # 标题 - 添加显著性标记
        if abs(corr_val) >= 0.5:
            strength = '⭐⭐⭐'
            strength_text = '强相关'
        elif abs(corr_val) >= 0.3:
            strength = '⭐⭐'
            strength_text = '中等相关'
        else:
            strength = '⭐'
            strength_text = '弱相关'
        
        title = f"{feature_cn} {strength}\n"
        title += f"r = {corr_val:.4f}"
        if p_val < 0.001:
            title += " (p < 0.001)"
        else:
            title += f" (p = {p_val:.3f})"
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel('年龄（岁）', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{feature_cn}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
        ax.set_xlim([0, 90])
        
        # 添加图例（仅第一个子图）
        if idx == 0:
            ax.legend(fontsize=9, loc='best')
        
        # 添加边框
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(1.8)
    
    plt.tight_layout(pad=2.5)
    scatter_path = figures_dir / 'texture_features_correlation.png'
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   相关性散点图已保存: {scatter_path}")
    
    # 8. 热图
    print(f"   绘制相关性热图...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    corr_matrix = df[feature_cols].corr()
    feature_labels = [feature_names_cn.get(f, f) for f in feature_cols]
    corr_matrix.index = feature_labels
    corr_matrix.columns = feature_labels
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1,
                cbar_kws={'label': '相关系数'},
                linewidths=0.5, linecolor='white',
                ax=ax, square=True)
    
    ax.set_title(f'{muscle_name}肌肉：纹理特征相关性热图', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    heatmap_path = figures_dir / 'feature_correlation_heatmap.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   特征相关性热图已保存: {heatmap_path}")
    
    # 9. 保存统计摘要
    summary_path = data_dir / 'texture_analysis_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*60}\n")
        f.write(f"超声图像纹理特征分析报告 - {muscle_name}肌肉（改进版）\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"数据集信息:\n")
        f.write(f"  受试者数量: {len(df)}\n")
        f.write(f"  总图像数: {df['num_images'].sum()}\n")
        f.write(f"  平均每人图像数: {df['num_images'].mean():.2f}\n\n")
        f.write(f"特征相关性排名:\n")
        for _, row in corr_df.iterrows():
            feature_cn = feature_names_cn.get(row['feature'], row['feature'])
            f.write(f"  {feature_cn:12s}: r = {row['correlation']:7.4f}, p = {row['p_value']:.2e}\n")
        f.write(f"\n方法改进:\n")
        f.write(f"  - 使用拉普拉斯方差评估图像清晰度\n")
        f.write(f"  - 从每个受试者选择清晰度最高的一张图像\n")
        f.write(f"  - 优化可视化：渐变配色、置信区间、显著性标记\n")
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
        print("用法: python compute_texture_features_v2.py <图像目录> <Excel文件> [肌肉名称]")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    excel_path = sys.argv[2]
    muscle_name = sys.argv[3] if len(sys.argv) > 3 else 'TA'
    
    analyze_texture_features(image_dir, excel_path, muscle_name=muscle_name)
