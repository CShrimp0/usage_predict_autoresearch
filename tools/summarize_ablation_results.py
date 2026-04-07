"""
消融实验结果汇总脚本
读取所有ablation实验的config.json并生成对比表格
"""

import json
from pathlib import Path
import pandas as pd


def load_experiment_results(ablation_dir='./outputs/ablation'):
    """加载所有实验结果"""
    ablation_dir = Path(ablation_dir)
    
    results = []
    
    # 遍历所有实验目录
    for exp_dir in sorted(ablation_dir.glob('*')):
        if not exp_dir.is_dir():
            continue
        
        # config.json可能直接在实验目录下，或在run_YYYYMMDD_HHMMSS子目录下
        config_file = exp_dir / 'config.json'
        if not config_file.exists():
            # 查找最新的run子目录
            run_dirs = sorted(exp_dir.glob('run_*'), reverse=True)
            if run_dirs:
                config_file = run_dirs[0] / 'config.json'
        
        if not config_file.exists():
            print(f"警告: {exp_dir.name} 没有config.json")
            continue
        
        # 同时查找history.json获取训练结果
        history_file = config_file.parent / 'history.json'
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # 从history.json读取性能指标
            best_val_mae = None
            best_val_rmse = None
            best_epoch = None
            
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
                
                if 'val_mae' in history and len(history['val_mae']) > 0:
                    val_mae_list = history['val_mae']
                    best_val_mae = min(val_mae_list)
                    best_epoch = val_mae_list.index(best_val_mae) + 1
                    
                    if 'val_rmse' in history:
                        best_val_rmse = history['val_rmse'][best_epoch - 1]
            
            # 提取关键信息
            exp_name = exp_dir.name
            
            # 从all_args获取参数（如果存在）
            args = config.get('all_args', config)
            
            # 辅助特征配置
            use_aux = args.get('use_aux_features', False)
            gender = args.get('aux_gender', False)
            bmi = args.get('aux_bmi', False)
            skewness = args.get('aux_skewness', False)
            intensity = args.get('aux_intensity', False)
            clarity = args.get('aux_clarity', False)
            aux_hidden = args.get('aux_hidden_dim', 32)
            
            # 构建特征描述
            features = []
            if gender: features.append('性别')
            if bmi: features.append('BMI')
            if skewness: features.append('偏度')
            if intensity: features.append('灰度')
            if clarity: features.append('清晰度')
            
            feature_str = '+'.join(features) if features else '无'
            aux_dim = 0
            if gender: aux_dim += 2
            if bmi: aux_dim += 1
            if skewness: aux_dim += 1
            if intensity: aux_dim += 1
            if clarity: aux_dim += 1
            
            results.append({
                '实验编号': exp_name,
                '辅助特征': feature_str,
                '特征维度': aux_dim,
                '隐藏层': aux_hidden if use_aux else '-',
                '最佳MAE': f'{best_val_mae:.2f}' if best_val_mae else '-',
                '最佳RMSE': f'{best_val_rmse:.2f}' if best_val_rmse else '-',
                '最佳Epoch': best_epoch if best_epoch else '-',
                'MAE (数值)': best_val_mae  # 用于排序
            })
        
        except Exception as e:
            print(f"错误: 无法读取 {exp_dir.name}: {e}")
    
    return results


def main():
    print("=" * 80)
    print("多模态Late Fusion消融实验结果汇总")
    print("=" * 80)
    print()
    
    results = load_experiment_results()
    
    if not results:
        print("未找到实验结果！")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 按MAE排序
    df_sorted = df.sort_values(by='MAE (数值)', ascending=True)
    
    # 删除辅助列
    df_display = df_sorted.drop(columns=['MAE (数值)'])
    
    # 打印表格
    print(df_display.to_string(index=False))
    print()
    
    # 计算改进
    baseline_row = df[df['实验编号'].str.contains('baseline')].iloc[0]
    baseline_mae = baseline_row['MAE (数值)']
    
    print("=" * 80)
    print("相比Baseline的改进:")
    print("=" * 80)
    
    for _, row in df_sorted.iterrows():
        if 'baseline' in row['实验编号']:
            continue
        
        mae = row['MAE (数值)']
        improvement = baseline_mae - mae
        improvement_pct = (improvement / baseline_mae) * 100
        
        print(f"{row['实验编号']:30s} | 特征: {row['辅助特征']:30s} | "
              f"MAE: {mae:6.2f} | 改进: {improvement:+6.2f} ({improvement_pct:+5.2f}%)")
    
    print()
    print("=" * 80)
    
    # 保存CSV
    output_file = Path('./outputs/ablation/results_summary.csv')
    df_display.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"结果已保存至: {output_file}")
    
    # 找出最佳配置
    best_row = df_sorted.iloc[0]
    print()
    print("=" * 80)
    print(f"最佳配置: {best_row['实验编号']}")
    print(f"  辅助特征: {best_row['辅助特征']}")
    print(f"  特征维度: {best_row['特征维度']}")
    print(f"  隐藏层维度: {best_row['隐藏层']}")
    print(f"  最佳MAE: {best_row['最佳MAE']} years")
    print(f"  最佳RMSE: {best_row['最佳RMSE']} years")
    print(f"  相比Baseline改进: {baseline_mae - best_row['MAE (数值)']:.2f} years "
          f"({((baseline_mae - best_row['MAE (数值)']) / baseline_mae * 100):.2f}%)")
    print("=" * 80)


if __name__ == '__main__':
    main()
