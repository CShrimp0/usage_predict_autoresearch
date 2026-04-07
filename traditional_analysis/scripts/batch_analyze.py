"""
批量分析多个肌肉部位的超声图像特征
"""

import sys
import subprocess
from pathlib import Path

# 肌肉部位配置
MUSCLE_CONFIG = {
    'TA': {
        'name': '胫骨前肌',
        'image_dir': '/home/szdx/LNX/data/TA/Healthy/Images',
        'excel_path': '/home/szdx/LNX/data/TA/characteristics.xlsx'
    },
    'GM': {
        'name': '腓肠肌内侧头',
        'image_dir': '/home/szdx/LNX/data/GM/Healthy/Images',
        'excel_path': '/home/szdx/LNX/data/GM/characteristics.xlsx'
    },
    'BB': {
        'name': '肱二头肌',
        'image_dir': '/home/szdx/LNX/data/BB/Healthy/Images',
        'excel_path': '/home/szdx/LNX/data/BB/characteristics.xlsx'
    }
}


def run_analysis(muscle_codes):
    """批量运行分析"""
    script_dir = Path(__file__).parent
    
    for muscle_code in muscle_codes:
        if muscle_code not in MUSCLE_CONFIG:
            print(f"⚠️  未知肌肉代码: {muscle_code}")
            continue
        
        config = MUSCLE_CONFIG[muscle_code]
        print("\n" + "="*70)
        print(f"开始分析 {muscle_code} ({config['name']})")
        print("="*70)
        
        # 1. 平均灰度值分析
        print(f"\n🔍 步骤1: 计算平均灰度值...")
        cmd1 = [
            'python', str(script_dir / 'compute_pixel_intensity.py'),
            config['image_dir'],
            config['excel_path'],
            muscle_code
        ]
        subprocess.run(cmd1)
        
        # 2. 纹理特征分析
        print(f"\n🔍 步骤2: 计算纹理特征...")
        cmd2 = [
            'python', str(script_dir / 'compute_texture_features.py'),
            config['image_dir'],
            config['excel_path'],
            muscle_code
        ]
        subprocess.run(cmd2)
        
        print(f"\n✅ {muscle_code} ({config['name']}) 分析完成！")
    
    print("\n" + "="*70)
    print("✨ 所有分析完成！")
    print("="*70)
    print("\n📊 结果保存在: traditional_analysis/results/")
    print("   - TA/   胫骨前肌")
    print("   - GM/   腓肠肌内侧头")
    print("   - BB/   肱二头肌")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # 从命令行参数获取要分析的肌肉
        muscle_codes = sys.argv[1:]
    else:
        # 默认分析所有三个肌肉
        muscle_codes = ['TA', 'GM', 'BB']
    
    print("="*70)
    print("🔬 超声图像传统特征批量分析")
    print("="*70)
    print(f"\n将分析以下肌肉: {', '.join(muscle_codes)}")
    
    run_analysis(muscle_codes)
