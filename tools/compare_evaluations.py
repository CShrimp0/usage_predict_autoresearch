#!/usr/bin/env python3
"""
对比多个模型的评估结果

Usage:
    python tools/compare_evaluations.py evaluation_results/*/test_metrics.json
    python tools/compare_evaluations.py evaluation_results/run_*/test_metrics.json --output comparison.csv
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
import sys


def load_metrics(json_path: Path) -> Dict:
    """加载单个test_metrics.json文件"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 无法加载 {json_path}: {e}")
        return None


def extract_key_metrics(metrics: Dict, json_path: Path) -> Dict:
    """从metrics中提取关键信息用于对比"""
    try:
        # 兼容新旧格式
        if 'overall_metrics' in metrics:
            # 新格式 (v2.0)
            return {
                'run_name': json_path.parent.name,
                'checkpoint': metrics['evaluation_info']['checkpoint_path'],
                'eval_time': metrics['evaluation_info']['evaluation_time'],
                'model': metrics['model_config']['architecture'],
                'dropout': metrics['model_config']['dropout'],
                'best_epoch': metrics['model_config']['best_epoch'],
                'val_mae': metrics['model_config']['val_mae'],
                'age_range': metrics['dataset_config']['age_range'],
                'test_samples': metrics['dataset_config']['total_samples'],
                'test_mae': metrics['overall_metrics']['MAE']['value'],
                'test_rmse': metrics['overall_metrics']['RMSE']['value'],
                'correlation': metrics['overall_metrics']['Correlation']['value'],
                'acc_5y': metrics['overall_metrics']['Accuracy_5years']['value'],
                'acc_10y': metrics['overall_metrics']['Accuracy_10years']['value'],
                'acc_15y': metrics['overall_metrics']['Accuracy_15years']['value'],
                'outlier_pct': metrics.get('error_analysis', {}).get('outlier_count', {}).get('percentage', 0)
            }
        else:
            # 旧格式 (v1.0)
            return {
                'run_name': json_path.parent.name,
                'checkpoint': 'N/A',
                'eval_time': 'N/A',
                'model': 'N/A',
                'dropout': 'N/A',
                'best_epoch': 'N/A',
                'val_mae': 'N/A',
                'age_range': 'N/A',
                'test_samples': metrics.get('total_samples', 'N/A'),
                'test_mae': metrics['MAE'],
                'test_rmse': metrics['RMSE'],
                'correlation': metrics['Correlation'],
                'acc_5y': metrics['Accuracy_5years'],
                'acc_10y': metrics['Accuracy_10years'],
                'acc_15y': metrics['Accuracy_15years'],
                'outlier_pct': 0
            }
    except KeyError as e:
        print(f"⚠️  {json_path.parent.name}: 缺少字段 {e}")
        return None


def compare_evaluations(json_paths: List[Path], output_path: Path = None):
    """对比多个评估结果"""
    print(f"\n📊 对比 {len(json_paths)} 个评估结果...\n")
    
    # 加载所有metrics
    all_metrics = []
    for json_path in json_paths:
        metrics = load_metrics(json_path)
        if metrics:
            extracted = extract_key_metrics(metrics, json_path)
            if extracted:
                all_metrics.append(extracted)
    
    if not all_metrics:
        print("❌ 未找到有效的评估结果")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(all_metrics)
    
    # 按MAE排序
    df = df.sort_values('test_mae')
    
    # 格式化显示
    display_df = df.copy()
    for col in ['val_mae', 'test_mae', 'test_rmse', 'correlation', 'outlier_pct']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
    
    for col in ['acc_5y', 'acc_10y', 'acc_15y']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x)
    
    # 打印摘要表格
    print("="*120)
    print("评估结果对比")
    print("="*120)
    
    summary_cols = ['run_name', 'model', 'dropout', 'age_range', 'test_samples', 
                    'val_mae', 'test_mae', 'test_rmse', 'correlation']
    available_cols = [col for col in summary_cols if col in display_df.columns]
    print(display_df[available_cols].to_string(index=False))
    print("="*120)
    
    print("\n准确率对比:")
    print("-"*80)
    acc_cols = ['run_name', 'acc_5y', 'acc_10y', 'acc_15y', 'outlier_pct']
    available_acc_cols = [col for col in acc_cols if col in display_df.columns]
    print(display_df[available_acc_cols].to_string(index=False))
    print("-"*80)
    
    # 找出最佳模型
    best_idx = df['test_mae'].idxmin()
    best_run = df.loc[best_idx, 'run_name']
    best_mae = df.loc[best_idx, 'test_mae']
    
    print(f"\n🏆 最佳模型: {best_run}")
    print(f"   Test MAE: {best_mae:.2f} years")
    print(f"   Test RMSE: {df.loc[best_idx, 'test_rmse']:.2f} years")
    print(f"   Correlation: {df.loc[best_idx, 'correlation']:.4f}")
    
    # 保存详细结果
    if output_path:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 详细对比结果已保存: {output_path}")
    
    # 打印年龄段对比（可选）
    if len(json_paths) <= 5:  # 只对少量模型显示年龄段对比
        print("\n年龄段MAE对比:")
        print("-"*80)
        for json_path in json_paths:
            metrics = load_metrics(json_path)
            if metrics and 'age_group_analysis' in metrics:
                run_name = json_path.parent.name
                print(f"\n{run_name}:")
                for group in metrics['age_group_analysis']:
                    age_range = group['age_range']
                    mae = group['mae']
                    count = group['count']
                    print(f"  {age_range:>8}: MAE={mae:>6.2f}, n={count:>3}")
        print("-"*80)


def main():
    parser = argparse.ArgumentParser(
        description='对比多个模型的评估结果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 对比所有评估结果
  python tools/compare_evaluations.py evaluation_results/*/test_metrics.json
  
  # 对比特定runs
  python tools/compare_evaluations.py evaluation_results/run_20260113_*/test_metrics.json
  
  # 保存对比结果到CSV
  python tools/compare_evaluations.py evaluation_results/*/test_metrics.json --output comparison.csv
        """
    )
    
    parser.add_argument('json_files', nargs='+', type=str,
                       help='test_metrics.json文件路径（支持glob模式）')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='输出CSV文件路径（可选）')
    
    args = parser.parse_args()
    
    # 收集所有JSON文件
    json_paths = []
    for pattern in args.json_files:
        path = Path(pattern)
        if path.exists() and path.is_file():
            json_paths.append(path)
        else:
            # 尝试glob匹配
            matched = list(Path('.').glob(pattern))
            json_paths.extend([p for p in matched if p.is_file()])
    
    if not json_paths:
        print(f"❌ 未找到匹配的文件: {args.json_files}")
        sys.exit(1)
    
    # 去重
    json_paths = list(set(json_paths))
    
    # 执行对比
    output_path = Path(args.output) if args.output else None
    compare_evaluations(json_paths, output_path)


if __name__ == '__main__':
    main()
