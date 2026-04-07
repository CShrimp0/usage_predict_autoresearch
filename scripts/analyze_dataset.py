"""
分析数据集中的图像分布和相似度
检查每个受试者的图像数量，并计算同一受试者不同图像之间的相似度
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim
import cv2
from tqdm import tqdm


def analyze_image_distribution(image_dir):
    """
    分析图像分布：统计每个ID的图像数量
    
    Args:
        image_dir: 图像文件夹路径
    
    Returns:
        subject_images: {subject_id: [image_paths]}
    """
    image_folder = Path(image_dir)
    subject_images = defaultdict(list)
    
    # 扫描所有图像
    for img_file in sorted(image_folder.glob('*.png')):
        filename = img_file.stem  # anon_xxx_x
        parts = filename.split('_')
        
        if len(parts) >= 3:
            # 格式: anon_xxx_x
            subject_id = parts[1]  # xxx
            sample_num = parts[2]  # x (1/2/3)
            subject_images[subject_id].append({
                'path': str(img_file),
                'sample_num': sample_num,
                'filename': filename
            })
    
    return subject_images


def calculate_image_similarity(img1_path, img2_path, resize=(224, 224)):
    """
    计算两张图像的相似度（使用SSIM）
    
    Args:
        img1_path, img2_path: 图像路径
        resize: 调整大小以加速计算
    
    Returns:
        ssim_score: 结构相似度 (0-1, 越大越相似)
        mse: 均方误差 (越小越相似)
    """
    # 读取图像
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        return None, None
    
    # 调整大小
    img1 = cv2.resize(img1, resize)
    img2 = cv2.resize(img2, resize)
    
    # 转为灰度图
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 计算SSIM
    ssim_score = ssim(gray1, gray2)
    
    # 计算MSE
    mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
    
    return ssim_score, mse


def analyze_subject_similarities(subject_images, max_subjects=None):
    """
    分析每个受试者的图像相似度
    
    Args:
        subject_images: {subject_id: [image_info]}
        max_subjects: 最多分析的受试者数量（用于快速测试）
    
    Returns:
        similarity_stats: 相似度统计信息
    """
    similarity_results = []
    
    subjects = list(subject_images.keys())
    if max_subjects:
        subjects = subjects[:max_subjects]
    
    print(f"\n开始计算图像相似度（共{len(subjects)}个受试者）...")
    
    for subject_id in tqdm(subjects):
        images = subject_images[subject_id]
        
        if len(images) < 2:
            continue
        
        # 计算该受试者所有图像对之间的相似度
        for i in range(len(images)):
            for j in range(i+1, len(images)):
                img1 = images[i]
                img2 = images[j]
                
                ssim_score, mse = calculate_image_similarity(
                    img1['path'], img2['path']
                )
                
                if ssim_score is not None:
                    similarity_results.append({
                        'subject_id': subject_id,
                        'image1': img1['filename'],
                        'image2': img2['filename'],
                        'sample1': img1['sample_num'],
                        'sample2': img2['sample_num'],
                        'ssim': ssim_score,
                        'mse': mse
                    })
    
    return pd.DataFrame(similarity_results)


def print_analysis_summary(subject_images, similarity_df):
    """
    打印分析摘要
    """
    print("\n" + "="*80)
    print("数据集分析报告")
    print("="*80)
    
    # 1. 基本统计
    total_subjects = len(subject_images)
    total_images = sum(len(imgs) for imgs in subject_images.values())
    
    print(f"\n【基本统计】")
    print(f"  总受试者数: {total_subjects}")
    print(f"  总图像数: {total_images}")
    print(f"  平均每人图像数: {total_images / total_subjects:.2f}")
    
    # 2. 图像数量分布
    images_per_subject = [len(imgs) for imgs in subject_images.values()]
    print(f"\n【每个受试者的图像数量分布】")
    for num_images in sorted(set(images_per_subject)):
        count = images_per_subject.count(num_images)
        percentage = count / total_subjects * 100
        print(f"  {num_images}张图像: {count}人 ({percentage:.1f}%)")
    
    # 3. 检查是否所有受试者都有3张图
    subjects_with_3_images = sum(1 for count in images_per_subject if count == 3)
    print(f"\n【完整性检查】")
    print(f"  有3张图像的受试者: {subjects_with_3_images} / {total_subjects} ({subjects_with_3_images/total_subjects*100:.1f}%)")
    
    if subjects_with_3_images < total_subjects:
        print(f"  ⚠️  警告: 有 {total_subjects - subjects_with_3_images} 个受试者图像数量不足3张")
        
        # 列出图像不足的受试者
        print(f"\n  图像数量不足的受试者:")
        for subject_id, images in subject_images.items():
            if len(images) != 3:
                print(f"    - {subject_id}: {len(images)}张")
    
    # 4. 相似度分析
    if not similarity_df.empty:
        print(f"\n【相似度分析】（基于SSIM，范围0-1，越大越相似）")
        print(f"  分析的图像对数: {len(similarity_df)}")
        print(f"  平均SSIM: {similarity_df['ssim'].mean():.4f}")
        print(f"  中位数SSIM: {similarity_df['ssim'].median():.4f}")
        print(f"  最小SSIM: {similarity_df['ssim'].min():.4f}")
        print(f"  最大SSIM: {similarity_df['ssim'].max():.4f}")
        print(f"  标准差: {similarity_df['ssim'].std():.4f}")
        
        # SSIM分布
        print(f"\n【SSIM分布】")
        bins = [(0, 0.5, "低相似度"), (0.5, 0.7, "中等相似度"), 
                (0.7, 0.85, "较高相似度"), (0.85, 1.0, "极高相似度")]
        for low, high, label in bins:
            count = ((similarity_df['ssim'] >= low) & (similarity_df['ssim'] < high)).sum()
            percentage = count / len(similarity_df) * 100
            print(f"  {label} [{low:.2f}-{high:.2f}): {count} 对 ({percentage:.1f}%)")
        
        # MSE统计
        print(f"\n【MSE统计】（均方误差，越小越相似）")
        print(f"  平均MSE: {similarity_df['mse'].mean():.2f}")
        print(f"  中位数MSE: {similarity_df['mse'].median():.2f}")
        print(f"  最小MSE: {similarity_df['mse'].min():.2f}")
        print(f"  最大MSE: {similarity_df['mse'].max():.2f}")
    
    # 5. 建议
    print(f"\n【数据处理建议】")
    if not similarity_df.empty:
        avg_ssim = similarity_df['ssim'].mean()
        
        if avg_ssim >= 0.85:
            print(f"  ✅ 同一受试者的图像相似度很高 (SSIM={avg_ssim:.3f})")
            print(f"     建议: 对三张图像取平均后再做数据增强")
            print(f"     优点: 减少噪声，提高标签质量")
        elif avg_ssim >= 0.70:
            print(f"  ⚠️  同一受试者的图像相似度中等 (SSIM={avg_ssim:.3f})")
            print(f"     建议: 可以选择以下策略之一:")
            print(f"       1. 每张图像独立做数据增强（保留多样性）")
            print(f"       2. 取平均后再增强（减少噪声）")
            print(f"     需要根据实际情况权衡")
        else:
            print(f"  ❌ 同一受试者的图像相似度较低 (SSIM={avg_ssim:.3f})")
            print(f"     建议: 每张图像独立做数据增强")
            print(f"     理由: 图像差异大，可能来自不同位置/角度，应保留多样性")
    
    print(f"\n  🔒 数据泄露防护:")
    print(f"     必须按受试者ID划分训练/验证/测试集")
    print(f"     确保同一受试者的所有图像在同一个集合中")
    
    print("\n" + "="*80)


def save_analysis_results(subject_images, similarity_df, output_dir='./analysis_results'):
    """
    保存分析结果到文件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存每个受试者的图像列表
    subject_summary = []
    for subject_id, images in subject_images.items():
        subject_summary.append({
            'subject_id': subject_id,
            'num_images': len(images),
            'image_files': [img['filename'] for img in images]
        })
    
    subject_df = pd.DataFrame(subject_summary)
    subject_df.to_csv(f'{output_dir}/subject_summary.csv', index=False)
    print(f"\n✅ 受试者统计已保存到: {output_dir}/subject_summary.csv")
    
    # 2. 保存相似度结果
    if not similarity_df.empty:
        similarity_df.to_csv(f'{output_dir}/similarity_analysis.csv', index=False)
        print(f"✅ 相似度分析已保存到: {output_dir}/similarity_analysis.csv")
        
        # 3. 保存统计摘要
        with open(f'{output_dir}/analysis_summary.txt', 'w', encoding='utf-8') as f:
            f.write(f"数据集相似度统计\n")
            f.write(f"="*60 + "\n\n")
            f.write(f"总受试者数: {len(subject_images)}\n")
            f.write(f"总图像数: {sum(len(imgs) for imgs in subject_images.values())}\n")
            f.write(f"分析的图像对数: {len(similarity_df)}\n\n")
            f.write(f"SSIM统计:\n")
            f.write(f"  平均值: {similarity_df['ssim'].mean():.4f}\n")
            f.write(f"  中位数: {similarity_df['ssim'].median():.4f}\n")
            f.write(f"  标准差: {similarity_df['ssim'].std():.4f}\n")
            f.write(f"  范围: [{similarity_df['ssim'].min():.4f}, {similarity_df['ssim'].max():.4f}]\n")
        
        print(f"✅ 统计摘要已保存到: {output_dir}/analysis_summary.txt")


if __name__ == '__main__':
    # 配置路径
    image_dir = '/home/szdx/LNX/data/TA/Healthy/Images'
    output_dir = './analysis_results'
    
    print("开始分析数据集...")
    print(f"图像目录: {image_dir}\n")
    
    # 步骤1: 分析图像分布
    print("步骤1: 统计每个受试者的图像数量...")
    subject_images = analyze_image_distribution(image_dir)
    
    # 步骤2: 计算相似度（可以设置max_subjects限制分析数量以加速）
    # 如果要分析所有受试者，将max_subjects=None
    # 如果只想快速测试，可以设置max_subjects=50
    print("\n步骤2: 计算图像相似度...")
    print("提示: 这可能需要几分钟时间...")
    similarity_df = analyze_subject_similarities(subject_images, max_subjects=None)
    
    # 步骤3: 打印分析摘要
    print_analysis_summary(subject_images, similarity_df)
    
    # 步骤4: 保存结果
    save_analysis_results(subject_images, similarity_df, output_dir)
    
    print("\n✅ 分析完成！")
