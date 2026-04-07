"""
对比不同肌肉部位的特征与年龄相关性
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

MUSCLE_NAMES = {
    'TA': '胫骨前肌',
    'GM': '腓肠肌内侧头',
    'BB': '肱二头肌'
}

FEATURE_NAMES = {
    'mean': '平均灰度',
    'std': '标准差',
    'skewness': '偏度',
    'kurtosis': '峰度',
    'entropy': '熵',
    'contrast': '对比度',
    'dissimilarity': '相异性',
    'homogeneity': '同质性',
    'energy': '能量',
    'correlation': '相关性',
    'ASM': 'ASM'
}


def load_data(muscle_codes):
    """加载所有肌肉的数据"""
    data = {}
    
    for muscle in muscle_codes:
        muscle_dir = Path('results') / muscle
        
        # 加载灰度值数据
        intensity_file = muscle_dir / 'data' / 'pixel_intensity.csv'
        if intensity_file.exists():
            data[f'{muscle}_intensity'] = pd.read_csv(intensity_file)
        
        # 加载纹理特征数据
        texture_file = muscle_dir / 'data' / 'texture_features.csv'
        if texture_file.exists():
            data[f'{muscle}_texture'] = pd.read_csv(texture_file)
        
        # 加载相关性数据
        corr_file = muscle_dir / 'data' / 'correlations.csv'
        if corr_file.exists():
            data[f'{muscle}_corr'] = pd.read_csv(corr_file)
    
    return data


def plot_intensity_comparison(data, muscle_codes, output_dir):
    """对比不同肌肉的灰度值与年龄关系"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, (muscle, color) in enumerate(zip(muscle_codes, colors)):
        ax = axes[idx]
        df = data[f'{muscle}_intensity']
        
        # 散点图
        ax.scatter(df['age'], df['mean_intensity'], 
                  alpha=0.3, s=20, color=color, edgecolors='none')
        
        # 拟合线
        z = np.polyfit(df['age'], df['mean_intensity'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['age'].min(), df['age'].max(), 100)
        ax.plot(x_line, p(x_line), color='darkred', linewidth=2, linestyle='--')
        
        # 计算相关系数
        corr = df['age'].corr(df['mean_intensity'])
        
        # 标题和标签
        ax.set_title(f'{MUSCLE_NAMES[muscle]}\nr = {corr:.3f}', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('年龄（岁）', fontsize=12)
        ax.set_ylabel('平均灰度值', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([0, 90])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'intensity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   灰度对比图已保存: intensity_comparison.png")


def plot_correlation_heatmap(data, muscle_codes, output_dir):
    """绘制所有特征在不同肌肉中的相关性热图"""
    
    # 构建相关系数矩阵
    corr_matrix = []
    feature_list = []
    
    for muscle in muscle_codes:
        corr_df = data[f'{muscle}_corr']
        
        # 按特征名称排序
        corr_df = corr_df.sort_values('feature')
        
        if len(feature_list) == 0:
            feature_list = corr_df['feature'].tolist()
        
        corr_matrix.append(corr_df['correlation'].values)
    
    corr_matrix = np.array(corr_matrix).T
    
    # 创建DataFrame
    corr_df = pd.DataFrame(
        corr_matrix,
        index=[FEATURE_NAMES.get(f, f) for f in feature_list],
        columns=[MUSCLE_NAMES[m] for m in muscle_codes]
    )
    
    # 绘制热图
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(corr_df, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, vmin=-0.7, vmax=0.7,
                cbar_kws={'label': 'Pearson 相关系数'},
                linewidths=0.5, linecolor='white',
                ax=ax)
    
    ax.set_title('不同肌肉部位的特征-年龄相关性对比', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('肌肉部位', fontsize=12)
    ax.set_ylabel('纹理特征', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   相关性热图已保存: correlation_heatmap_comparison.png")


def plot_top_features(data, muscle_codes, output_dir, top_n=5):
    """绘制每个肌肉的Top N特征"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, (muscle, color) in enumerate(zip(muscle_codes, colors)):
        ax = axes[idx]
        corr_df = data[f'{muscle}_corr']
        
        # 按相关系数绝对值排序
        corr_df['abs_r'] = corr_df['correlation'].abs()
        top_features = corr_df.nlargest(top_n, 'abs_r')
        
        # 绘制条形图
        y_pos = np.arange(len(top_features))
        bars = ax.barh(y_pos, top_features['correlation'].values, color=color, alpha=0.7)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f' {width:.3f}', 
                   va='center', ha='left' if width > 0 else 'right',
                   fontsize=10, fontweight='bold')
        
        # 设置y轴标签
        feature_labels = [FEATURE_NAMES.get(f, f) for f in top_features['feature']]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_labels)
        
        # 标题和标签
        ax.set_title(f'{MUSCLE_NAMES[muscle]}\nTop {top_n} 特征', 
                    fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel('Pearson 相关系数', fontsize=11)
        ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        ax.set_xlim([-0.7, 0.7])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_features_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Top特征对比图已保存: top_features_comparison.png")


def generate_summary_report(data, muscle_codes, output_dir):
    """生成跨肌肉对比总结报告"""
    
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("超声图像传统特征跨肌肉对比分析报告")
    report_lines.append("="*70)
    report_lines.append("")
    
    # 1. 灰度值对比
    report_lines.append("【1】平均灰度值与年龄相关性对比")
    report_lines.append("-" * 70)
    
    for muscle in muscle_codes:
        df = data[f'{muscle}_intensity']
        corr = df['age'].corr(df['mean_intensity'])
        mean_intensity = df['mean_intensity'].mean()
        std_intensity = df['mean_intensity'].std()
        n_samples = len(df)
        
        report_lines.append(f"\n{MUSCLE_NAMES[muscle]} (n={n_samples}):")
        report_lines.append(f"  平均灰度值: {mean_intensity:.2f} ± {std_intensity:.2f}")
        report_lines.append(f"  Pearson相关系数: r = {corr:.4f}")
        
        if abs(corr) >= 0.5:
            strength = "强相关"
        elif abs(corr) >= 0.3:
            strength = "中等相关"
        else:
            strength = "弱相关"
        report_lines.append(f"  相关性强度: {strength}")
    
    # 2. 纹理特征对比
    report_lines.append("\n\n【2】纹理特征相关性对比")
    report_lines.append("-" * 70)
    
    for muscle in muscle_codes:
        corr_df = data[f'{muscle}_corr']
        top3 = corr_df.nlargest(3, 'abs_correlation')
        
        report_lines.append(f"\n{MUSCLE_NAMES[muscle]} - Top 3 特征:")
        for idx, row in top3.iterrows():
            feature_name = FEATURE_NAMES.get(row['feature'], row['feature'])
            report_lines.append(f"  {feature_name:12s}: r = {row['correlation']:7.4f}, p = {row['p_value']:.2e}")
    
    # 3. 跨肌肉发现
    report_lines.append("\n\n【3】跨肌肉关键发现")
    report_lines.append("-" * 70)
    
    # 找出所有肌肉中都强相关的特征
    common_features = {}
    for muscle in muscle_codes:
        corr_df = data[f'{muscle}_corr']
        for _, row in corr_df.iterrows():
            feature = row['feature']
            if feature not in common_features:
                common_features[feature] = []
            common_features[feature].append(row['correlation'])
    
    report_lines.append("\n通用强相关特征（所有肌肉|r|>0.3）:")
    found_any = False
    for feature, correlations in common_features.items():
        if all(abs(r) > 0.3 for r in correlations):
            feature_name = FEATURE_NAMES.get(feature, feature)
            corr_str = ', '.join([f"{r:.3f}" for r in correlations])
            report_lines.append(f"  {feature_name}: [{corr_str}]")
            found_any = True
    if not found_any:
        report_lines.append("  无通用强相关特征")
    
    report_lines.append("\n肌肉特异性特征（仅在单个肌肉|r|>0.5）:")
    found_any = False
    for feature, correlations in common_features.items():
        strong_count = sum(abs(r) > 0.5 for r in correlations)
        if strong_count == 1:
            feature_name = FEATURE_NAMES.get(feature, feature)
            max_idx = np.argmax([abs(r) for r in correlations])
            report_lines.append(f"  {feature_name} in {MUSCLE_NAMES[muscle_codes[max_idx]]}: r={correlations[max_idx]:.3f}")
            found_any = True
    if not found_any:
        report_lines.append("  无显著肌肉特异性特征")
    
    # 4. 数据集统计
    report_lines.append("\n\n【4】数据集统计")
    report_lines.append("-" * 70)
    
    total_images = 0
    for muscle in muscle_codes:
        df = data[f'{muscle}_intensity']
        n_images = len(df)
        n_subjects = df['subject_id'].nunique()
        age_range = f"{df['age'].min():.1f} - {df['age'].max():.1f}"
        
        report_lines.append(f"\n{MUSCLE_NAMES[muscle]}:")
        report_lines.append(f"  图像数量: {n_images}")
        report_lines.append(f"  受试者数量: {n_subjects}")
        report_lines.append(f"  年龄范围: {age_range} 岁")
        
        total_images += n_images
    
    report_lines.append(f"\n总图像数: {total_images}")
    
    report_lines.append("\n" + "="*70)
    report_lines.append("报告生成完成")
    report_lines.append("="*70)
    
    # 保存报告
    report_path = output_dir / 'comparison_summary.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"   对比报告已保存: comparison_summary.txt")
    
    # 同时打印到终端
    print("\n" + '\n'.join(report_lines))


def main():
    if len(sys.argv) > 1:
        muscle_codes = sys.argv[1:]
    else:
        muscle_codes = ['TA', 'GM', 'BB']
    
    print("="*70)
    print("🔬 跨肌肉特征对比分析")
    print("="*70)
    print(f"\n对比肌肉: {', '.join([MUSCLE_NAMES[m] for m in muscle_codes])}\n")
    
    # 加载数据
    print("1. 加载数据...")
    data = load_data(muscle_codes)
    
    # 创建输出目录
    output_dir = Path('results') / 'comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成对比图表
    print("\n2. 生成对比可视化...")
    plot_intensity_comparison(data, muscle_codes, output_dir)
    plot_correlation_heatmap(data, muscle_codes, output_dir)
    plot_top_features(data, muscle_codes, output_dir)
    
    # 生成报告
    print("\n3. 生成对比报告...")
    generate_summary_report(data, muscle_codes, output_dir)
    
    print("\n" + "="*70)
    print("✨ 对比分析完成！")
    print("="*70)
    print(f"\n结果保存在: {output_dir}/")
    print("  - intensity_comparison.png")
    print("  - correlation_heatmap_comparison.png")
    print("  - top_features_comparison.png")
    print("  - comparison_summary.txt")


if __name__ == '__main__':
    main()
