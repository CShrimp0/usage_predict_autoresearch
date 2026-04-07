"""
错误样本可视化分析工具
生成HTML报告展示高错误、低错误和离群样本
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import cv2
from datetime import datetime


def find_image_path(filename: str, image_dir: str) -> str:
    """递归搜索图像文件"""
    for root, dirs, files in os.walk(image_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None


def get_relative_path(image_path: str, html_path: str) -> str:
    """计算从HTML文件到图像的相对路径"""
    try:
        # 获取绝对路径
        img_abs = os.path.abspath(image_path)
        html_abs = os.path.abspath(html_path)
        
        # 计算相对路径
        rel_path = os.path.relpath(img_abs, os.path.dirname(html_abs))
        
        # 转换为URL格式（使用正斜杠）
        rel_path = rel_path.replace('\\', '/')
        
        return rel_path
    except Exception as e:
        print(f"警告: 无法计算相对路径 {image_path}: {e}")
        return image_path


def parse_error_file(file_path: str) -> List[Dict]:
    """解析错误样本txt文件"""
    samples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # 跳过标题行和注释行
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('-'):
            continue
            
        # 格式: 文件名 | 真实年龄 | 预测年龄 | MAE | 误差方向 | [异常标记]
        # 新格式有6列，旧格式有5列
        parts = [p.strip() for p in line.split('\t')]
        if len(parts) >= 4:
            try:
                sample = {
                    'filename': parts[0],
                    'true_age': float(parts[1]),
                    'pred_age': float(parts[2]),
                    'error': float(parts[3])
                }
                # 如果有异常标记列（新格式）
                if len(parts) >= 6:
                    sample['outlier_flag'] = parts[5]
                samples.append(sample)
            except (ValueError, IndexError):
                continue
    
    return samples


def calculate_image_stats(image_path: str) -> Dict:
    """计算图像统计特征"""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # 基本统计
        mean_intensity = float(np.mean(img))
        std_intensity = float(np.std(img))
        
        # 清晰度 (Laplacian方差)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        clarity = float(np.var(laplacian))
        
        # 对比度 (标准差 / 均值)
        contrast = std_intensity / mean_intensity if mean_intensity > 0 else 0
        
        # 偏度
        skewness = float(np.mean(((img - mean_intensity) / std_intensity) ** 3)) if std_intensity > 0 else 0
        
        return {
            'mean': mean_intensity,
            'std': std_intensity,
            'clarity': clarity,
            'contrast': contrast,
            'skewness': skewness
        }
    except Exception as e:
        print(f"警告: 无法计算图像 {image_path} 的统计特征: {e}")
        return None


def generate_html_report(
    high_error_samples: List[Dict],
    low_error_samples: List[Dict],
    outlier_samples: List[Dict],
    image_dir: str,
    output_path: str,
    max_samples: int = 50
):
    """生成HTML可视化报告"""
    
    # 计算图像统计特征
    print("正在搜索图像文件并计算统计特征...")
    for sample_list in [high_error_samples, low_error_samples, outlier_samples]:
        for i, sample in enumerate(sample_list[:max_samples]):
            img_path = find_image_path(sample['filename'], image_dir)
            if img_path:
                sample['image_path'] = img_path
                sample['stats'] = calculate_image_stats(img_path)
            else:
                sample['image_path'] = None
                sample['stats'] = None
            
            if (i + 1) % 10 == 0:
                print(f"  已处理 {i + 1} 张图像...")
    
    print("图像统计特征计算完成!")
    
    # 计算统计摘要
    def calc_stats_summary(samples):
        if not samples:
            return None
        valid_stats = [s['stats'] for s in samples if s.get('stats')]
        if not valid_stats:
            return None
        return {
            'mean': np.mean([s['mean'] for s in valid_stats]),
            'std': np.mean([s['std'] for s in valid_stats]),
            'clarity': np.mean([s['clarity'] for s in valid_stats]),
            'contrast': np.mean([s['contrast'] for s in valid_stats]),
            'skewness': np.mean([s['skewness'] for s in valid_stats])
        }
    
    high_stats = calc_stats_summary(high_error_samples[:max_samples])
    low_stats = calc_stats_summary(low_error_samples[:max_samples])
    outlier_stats = calc_stats_summary(outlier_samples[:max_samples])
    
    # 传递output_path给生成函数
    html_output_path = output_path
    
    # 生成HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>错误样本分析报告</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Microsoft YaHei', 'WenQuanYi Micro Hei', sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        h1 {{
            color: #333;
            margin-bottom: 30px;
            text-align: center;
            font-size: 28px;
        }}
        
        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
            border-bottom: 2px solid #e0e0e0;
        }}
        
        .tab {{
            padding: 12px 24px;
            cursor: pointer;
            background: #f9f9f9;
            border: none;
            border-radius: 5px 5px 0 0;
            font-size: 16px;
            transition: all 0.3s;
        }}
        
        .tab:hover {{
            background: #e8e8e8;
        }}
        
        .tab.active {{
            background: #4CAF50;
            color: white;
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        .stats-summary {{
            background: #f0f8ff;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #2196F3;
        }}
        
        .stats-summary h3 {{
            color: #2196F3;
            margin-bottom: 15px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .stat-item {{
            background: white;
            padding: 12px;
            border-radius: 5px;
            border: 1px solid #ddd;
        }}
        
        .stat-label {{
            color: #666;
            font-size: 14px;
            margin-bottom: 5px;
        }}
        
        .stat-value {{
            color: #333;
            font-size: 18px;
            font-weight: bold;
        }}
        
        .samples-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
        }}
        
        .sample-card {{
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
            transition: all 0.3s;
            background: white;
        }}
        
        .sample-card:hover {{
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transform: translateY(-5px);
        }}
        
        .sample-image {{
            width: 100%;
            height: 200px;
            object-fit: cover;
            background: #f0f0f0;
        }}
        
        .sample-info {{
            padding: 15px;
        }}
        
        .sample-filename {{
            font-size: 12px;
            color: #666;
            margin-bottom: 10px;
            word-break: break-all;
        }}
        
        .sample-ages {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }}
        
        .age-item {{
            flex: 1;
        }}
        
        .age-label {{
            font-size: 12px;
            color: #888;
        }}
        
        .age-value {{
            font-size: 16px;
            font-weight: bold;
            color: #333;
        }}
        
        .sample-error {{
            background: #ff5252;
            color: white;
            padding: 8px;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        
        .sample-error.low {{
            background: #4CAF50;
        }}
        
        .sample-stats {{
            font-size: 11px;
            color: #666;
            background: #f9f9f9;
            padding: 8px;
            border-radius: 5px;
            margin-top: 10px;
        }}
        
        .sample-stats div {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 3px;
        }}
        
        .controls {{
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        
        .controls label {{
            font-size: 14px;
            color: #666;
        }}
        
        .controls select {{
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }}
        
        .no-image {{
            width: 100%;
            height: 200px;
            background: #f0f0f0;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #999;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 错误样本分析报告</h1>
        <p style="text-align: center; color: #666; margin-bottom: 30px;">
            生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('high-error')">
                ⚠️ 高错误样本 ({len(high_error_samples[:max_samples])})
            </button>
            <button class="tab" onclick="showTab('low-error')">
                ✅ 低错误样本 ({len(low_error_samples[:max_samples])})
            </button>
            <button class="tab" onclick="showTab('outliers')">
                🚨 离群样本 ({len(outlier_samples[:max_samples])})
            </button>
        </div>
"""

    # 高错误样本
    html += generate_tab_content(
        'high-error', 
        high_error_samples[:max_samples], 
        high_stats,
        '这些样本的预测误差最大，可能包含数据质量问题或模型难以处理的边缘情况',
        True,
        html_output_path
    )
    
    # 低错误样本
    html += generate_tab_content(
        'low-error', 
        low_error_samples[:max_samples], 
        low_stats,
        '这些样本的预测误差最小，代表模型表现最佳的情况',
        False,
        html_output_path
    )
    
    # 离群样本
    html += generate_tab_content(
        'outliers', 
        outlier_samples[:max_samples], 
        outlier_stats,
        '这些样本被识别为统计离群点，误差超过3倍标准差',
        True,
        html_output_path
    )

    # JavaScript
    html += """
        <script>
            function showTab(tabId) {
                // 隐藏所有标签页
                document.querySelectorAll('.tab-content').forEach(tab => {
                    tab.classList.remove('active');
                });
                document.querySelectorAll('.tab').forEach(tab => {
                    tab.classList.remove('active');
                });
                
                // 显示选中的标签页
                document.getElementById(tabId).classList.add('active');
                event.target.classList.add('active');
            }
            
            function sortSamples(containerId, sortBy) {
                const container = document.getElementById(containerId);
                const cards = Array.from(container.querySelectorAll('.sample-card'));
                
                cards.sort((a, b) => {
                    const aVal = parseFloat(a.dataset[sortBy]);
                    const bVal = parseFloat(b.dataset[sortBy]);
                    return sortBy === 'error' ? bVal - aVal : aVal - bVal;
                });
                
                container.innerHTML = '';
                cards.forEach(card => container.appendChild(card));
            }
        </script>
    </div>
</body>
</html>
"""

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ HTML报告已生成: {output_path}")


def generate_tab_content(tab_id: str, samples: List[Dict], stats: Dict, description: str, is_high_error: bool, html_path: str) -> str:
    """生成单个标签页内容"""
    
    html = f"""
        <div id="{tab_id}" class="tab-content {'active' if tab_id == 'high-error' else ''}">
            <div class="stats-summary">
                <h3>📊 图像统计特征</h3>
                <p style="color: #666; margin-bottom: 15px;">{description}</p>
"""
    
    if stats:
        html += f"""
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-label">平均灰度</div>
                        <div class="stat-value">{stats['mean']:.2f}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">灰度标准差</div>
                        <div class="stat-value">{stats['std']:.2f}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">清晰度</div>
                        <div class="stat-value">{stats['clarity']:.2f}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">对比度</div>
                        <div class="stat-value">{stats['contrast']:.3f}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">偏度</div>
                        <div class="stat-value">{stats['skewness']:.3f}</div>
                    </div>
                </div>
"""
    else:
        html += "<p style='color: #999;'>无法计算统计特征</p>"
    
    html += """
            </div>
            
            <div class="controls">
                <label>排序方式:</label>
                <select onchange="sortSamples('""" + tab_id + """-grid', this.value)">
                    <option value="error">按误差排序</option>
                    <option value="trueAge">按真实年龄排序</option>
                    <option value="predAge">按预测年龄排序</option>
                </select>
            </div>
            
            <div id=\"""" + tab_id + """-grid" class="samples-grid">
"""
    
    # 添加样本卡片
    for sample in samples:
        img_path = sample.get('image_path', '')
        stats = sample.get('stats')
        
        error_class = 'low' if not is_high_error else ''
        
        html += f"""
                <div class="sample-card" 
                     data-error="{abs(sample['error']):.2f}" 
                     data-true-age="{sample['true_age']:.1f}"
                     data-pred-age="{sample['pred_age']:.1f}">
"""
        
        # 图像
        if img_path and os.path.exists(img_path):
            # 使用相对路径而非file://协议
            rel_path = get_relative_path(img_path, html_path)
            html += f'                    <img src="{rel_path}" class="sample-image" alt="{sample["filename"]}">\n'
        else:
            html += '                    <div class="no-image">图像未找到</div>\n'
        
        # 信息
        html += f"""
                    <div class="sample-info">
                        <div class="sample-filename">{sample['filename']}</div>
                        <div class="sample-error {error_class}">
                            误差: {sample['error']:.2f} 岁
                        </div>
                        <div class="sample-ages">
                            <div class="age-item">
                                <div class="age-label">真实年龄</div>
                                <div class="age-value">{sample['true_age']:.1f}</div>
                            </div>
                            <div class="age-item">
                                <div class="age-label">预测年龄</div>
                                <div class="age-value">{sample['pred_age']:.1f}</div>
                            </div>
                        </div>
"""
        
        # 图像统计
        if stats:
            html += f"""
                        <div class="sample-stats">
                            <div><span>灰度:</span><span>{stats['mean']:.1f}</span></div>
                            <div><span>清晰度:</span><span>{stats['clarity']:.1f}</span></div>
                            <div><span>对比度:</span><span>{stats['contrast']:.3f}</span></div>
                        </div>
"""
        
        html += """
                    </div>
                </div>
"""
    
    html += """
            </div>
        </div>
"""
    
    return html


def main():
    parser = argparse.ArgumentParser(description='生成错误样本HTML可视化报告')
    parser.add_argument('--result-dir', type=str, required=True,
                        help='评估结果目录 (包含high_error_samples.txt等文件)')
    parser.add_argument('--image-dir', type=str, required=True,
                        help='图像文件目录')
    parser.add_argument('--output', type=str, default=None,
                        help='输出HTML文件路径 (默认保存到结果目录)')
    parser.add_argument('--max-samples', type=int, default=50,
                        help='每个类别显示的最大样本数 (默认: 50)')
    
    args = parser.parse_args()
    
    # 检查目录
    if not os.path.exists(args.result_dir):
        print(f"错误: 结果目录不存在: {args.result_dir}")
        return
    
    if not os.path.exists(args.image_dir):
        print(f"错误: 图像目录不存在: {args.image_dir}")
        return
    
    # 读取错误样本文件
    high_error_file = os.path.join(args.result_dir, 'high_error_samples.txt')
    low_error_file = os.path.join(args.result_dir, 'low_error_samples.txt')
    
    print("正在读取错误样本文件...")
    high_error_samples = parse_error_file(high_error_file) if os.path.exists(high_error_file) else []
    low_error_samples = parse_error_file(low_error_file) if os.path.exists(low_error_file) else []
    
    # 从high_error_samples中提取异常样本（带⚠️标记的）
    outlier_samples = [s for s in high_error_samples if '⚠️异常' in str(s.get('outlier_flag', ''))]
    # 如果没有outlier_flag字段，尝试读取旧格式的outlier文件
    if not outlier_samples:
        outlier_file = os.path.join(args.result_dir, 'outlier_samples.txt')
        if os.path.exists(outlier_file):
            outlier_samples = parse_error_file(outlier_file)
            print("  (使用旧格式的outlier_samples.txt)")
    
    print(f"  高错误样本: {len(high_error_samples)}")
    print(f"  低错误样本: {len(low_error_samples)}")
    print(f"  异常样本: {len(outlier_samples)}")
    
    # 输出路径
    if args.output is None:
        output_path = os.path.join(args.result_dir, 'error_analysis_report.html')
    else:
        output_path = args.output
    
    # 生成报告
    print("\n正在生成HTML报告...")
    generate_html_report(
        high_error_samples,
        low_error_samples,
        outlier_samples,
        args.image_dir,
        output_path,
        args.max_samples
    )
    
    print(f"\n✅ 完成! 请在浏览器中打开: {output_path}")


if __name__ == '__main__':
    main()
