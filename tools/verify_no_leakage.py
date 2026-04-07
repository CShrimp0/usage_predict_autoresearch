"""
终极数据泄露验证 - 模拟数据集划分过程，验证无重叠

这个脚本会：
1. 读取完整的数据集配置
2. 使用相同的参数重新执行数据划分
3. 验证训练集、验证集、测试集的受试者ID完全不重叠
"""
import sys
import json
from pathlib import Path
from collections import defaultdict

# 导入数据集加载模块
sys.path.insert(0, '/home/szdx/LNX/usage_predict')
from dataset import stratified_split_by_age
import pandas as pd


def load_age_dict(excel_path):
    """从Excel加载年龄字典"""
    df = pd.read_excel(excel_path)
    
    age_dict = {}
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
    
    return age_dict


def get_subject_images(image_dir, age_dict, min_age=0, max_age=100):
    """获取符合年龄范围的受试者及其图像"""
    image_dir = Path(image_dir)
    all_image_paths = sorted(list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')))
    
    subject_images = defaultdict(list)
    for img_path in all_image_paths:
        parts = img_path.stem.split('_')
        if len(parts) >= 2:
            subject_id = parts[1]
            if subject_id in age_dict:
                subject_images[subject_id].append(str(img_path))
    
    # 年龄过滤
    filtered_subjects = []
    for subject_id in subject_images.keys():
        age = age_dict[subject_id]
        if min_age <= age <= max_age:
            filtered_subjects.append(subject_id)
    
    return filtered_subjects, subject_images


def verify_no_overlap(result_dir):
    """
    验证数据集划分无重叠
    
    通过重新执行数据划分逻辑，确认训练集、验证集、测试集的受试者ID完全不重叠
    """
    result_dir = Path(result_dir)
    
    # 读取配置
    with open(result_dir / "test_metrics.json", 'r') as f:
        metrics = json.load(f)
    
    checkpoint_path = Path(metrics['evaluation_info']['checkpoint_path'])
    config_file = checkpoint_path.parent / "config.json"
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print("=" * 100)
    print("🔐 终极数据泄露验证 - 重新执行数据划分")
    print("=" * 100)
    
    print(f"\n📋 使用的配置:")
    print(f"   - 图像目录: {config['dataset']['image_dir']}")
    print(f"   - Excel文件: {config['dataset']['excel_path']}")
    print(f"   - 测试集比例: {config['dataset']['test_size']}")
    print(f"   - 验证集比例: {config['dataset']['val_size']}")
    print(f"   - 随机种子: {config['dataset']['random_seed']}")
    print(f"   - 年龄分层: {config['dataset']['use_age_stratify']}")
    print(f"   - 年龄分组宽度: {config['dataset']['age_bin_width']}")
    
    # 读取年龄数据
    print(f"\n📊 加载数据...")
    age_dict = load_age_dict(config['dataset']['excel_path'])
    print(f"   - 总受试者数 (Excel): {len(age_dict)}")
    
    # 获取符合年龄范围的受试者
    # 从metrics中获取年龄范围
    age_range_str = metrics['dataset_config'].get('age_range', '0-100')
    if '-' in age_range_str:
        min_age, max_age = map(float, age_range_str.split('-'))
    else:
        min_age, max_age = 0, 100
    
    print(f"   - 年龄范围限制: {min_age}-{max_age} 岁")
    
    all_subjects, subject_images = get_subject_images(
        config['dataset']['image_dir'], 
        age_dict, 
        min_age, 
        max_age
    )
    
    print(f"   - 符合年龄范围的受试者: {len(all_subjects)}")
    
    # 重新执行数据划分
    print(f"\n🔄 重新执行数据划分...")
    
    if config['dataset']['use_age_stratify']:
        train_subjects, val_subjects, test_subjects = stratified_split_by_age(
            all_subjects,
            age_dict,
            test_size=config['dataset']['test_size'],
            val_size=config['dataset']['val_size'],
            random_state=config['dataset']['random_seed'],
            bin_width=config['dataset']['age_bin_width']
        )
    else:
        from sklearn.model_selection import train_test_split
        train_val_subjects, test_subjects = train_test_split(
            all_subjects,
            test_size=config['dataset']['test_size'],
            random_state=config['dataset']['random_seed'],
            shuffle=True
        )
        train_subjects, val_subjects = train_test_split(
            train_val_subjects,
            test_size=config['dataset']['val_size'],
            random_state=config['dataset']['random_seed'],
            shuffle=True
        )
    
    print(f"\n📊 划分结果:")
    print(f"   - 训练集受试者: {len(train_subjects)}")
    print(f"   - 验证集受试者: {len(val_subjects)}")
    print(f"   - 测试集受试者: {len(test_subjects)}")
    print(f"   - 总计: {len(train_subjects) + len(val_subjects) + len(test_subjects)}")
    
    # 验证与配置一致
    print(f"\n✅ 与训练配置对比:")
    train_match = len(train_subjects) == config['dataset']['train_subjects']
    val_match = len(val_subjects) == config['dataset']['val_subjects']
    test_match = len(test_subjects) == config['dataset']['test_subjects']
    
    print(f"   - 训练集: {len(train_subjects)} vs {config['dataset']['train_subjects']} {'✅' if train_match else '❌'}")
    print(f"   - 验证集: {len(val_subjects)} vs {config['dataset']['val_subjects']} {'✅' if val_match else '❌'}")
    print(f"   - 测试集: {len(test_subjects)} vs {config['dataset']['test_subjects']} {'✅' if test_match else '❌'}")
    
    # 核心验证：检查重叠
    print(f"\n" + "=" * 100)
    print(f"🔍 核心验证: 检查受试者ID重叠")
    print(f"=" * 100)
    
    train_set = set(train_subjects)
    val_set = set(val_subjects)
    test_set = set(test_subjects)
    
    # 检查两两之间的重叠
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set
    
    print(f"\n📊 重叠检查:")
    print(f"   - 训练集 ∩ 验证集: {len(train_val_overlap)} 个重叠受试者")
    if train_val_overlap:
        print(f"      ❌ 发现重叠: {sorted(list(train_val_overlap))[:10]}")
    else:
        print(f"      ✅ 无重叠")
    
    print(f"   - 训练集 ∩ 测试集: {len(train_test_overlap)} 个重叠受试者")
    if train_test_overlap:
        print(f"      ❌ 发现重叠: {sorted(list(train_test_overlap))[:10]}")
    else:
        print(f"      ✅ 无重叠")
    
    print(f"   - 验证集 ∩ 测试集: {len(val_test_overlap)} 个重叠受试者")
    if val_test_overlap:
        print(f"      ❌ 发现重叠: {sorted(list(val_test_overlap))[:10]}")
    else:
        print(f"      ✅ 无重叠")
    
    # 检查并集
    all_split_subjects = train_set | val_set | test_set
    print(f"\n📊 完整性检查:")
    print(f"   - 训练∪验证∪测试: {len(all_split_subjects)} 个受试者")
    print(f"   - 原始受试者数: {len(all_subjects)}")
    
    if len(all_split_subjects) == len(all_subjects):
        print(f"   ✅ 所有受试者都被分配到某个集合")
    else:
        missing = set(all_subjects) - all_split_subjects
        print(f"   ⚠️ 有 {len(missing)} 个受试者未被分配")
        if missing:
            print(f"      未分配的受试者: {sorted(list(missing))[:10]}")
    
    # 读取实际测试集预测结果，验证测试集受试者ID
    with open(result_dir / "predictions.json", 'r') as f:
        pred_data = json.load(f)
    
    actual_test_subjects = set()
    for filename in pred_data['filenames']:
        parts = Path(filename).stem.split('_')
        if len(parts) >= 2:
            actual_test_subjects.add(parts[1])
    
    print(f"\n📊 测试集验证:")
    print(f"   - 重新划分的测试集: {len(test_set)} 个受试者")
    print(f"   - 实际评估的测试集: {len(actual_test_subjects)} 个受试者")
    
    # 检查测试集是否完全一致
    test_match_set = test_set == actual_test_subjects
    test_extra = actual_test_subjects - test_set
    test_missing = test_set - actual_test_subjects
    
    if test_match_set:
        print(f"   ✅ 测试集受试者完全一致")
    else:
        print(f"   ⚠️ 测试集受试者不完全一致")
        if test_extra:
            print(f"      多出的受试者: {sorted(list(test_extra))}")
        if test_missing:
            print(f"      缺失的受试者: {sorted(list(test_missing))}")
    
    # 显示测试集受试者列表对比
    print(f"\n📝 测试集受试者ID对比:")
    print(f"   重新划分: {sorted(test_set)}")
    print(f"   实际评估: {sorted(actual_test_subjects)}")
    
    # 最终结论
    print(f"\n" + "=" * 100)
    print(f"📋 最终验证结论:")
    print(f"=" * 100)
    
    all_checks = [
        ("训练集与验证集无重叠", len(train_val_overlap) == 0),
        ("训练集与测试集无重叠", len(train_test_overlap) == 0),
        ("验证集与测试集无重叠", len(val_test_overlap) == 0),
        ("受试者完整覆盖", len(all_split_subjects) == len(all_subjects)),
        ("测试集受试者一致", test_match_set),
        ("受试者数量匹配", train_match and val_match and test_match)
    ]
    
    all_passed = all(check[1] for check in all_checks)
    
    for check_name, passed in all_checks:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"   {status}: {check_name}")
    
    print(f"\n" + "=" * 100)
    if all_passed:
        print(f"🎉 验证通过: 数据划分完全无重叠!")
        print(f"\n📌 验证证据:")
        print(f"   1. 重新执行数据划分逻辑，使用相同参数（seed={config['dataset']['random_seed']}）")
        print(f"   2. 训练集、验证集、测试集的受试者ID集合两两无交集")
        print(f"   3. 三个集合的并集等于全部受试者集合（无遗漏）")
        print(f"   4. 重新划分的测试集与实际评估的测试集完全一致")
        print(f"   5. 受试者数量与训练配置完全匹配")
        print(f"\n🔒 数学证明:")
        print(f"   设 Train={len(train_subjects)}, Val={len(val_subjects)}, Test={len(test_subjects)}")
        print(f"   Train ∩ Val = ∅ (空集)")
        print(f"   Train ∩ Test = ∅ (空集)")
        print(f"   Val ∩ Test = ∅ (空集)")
        print(f"   Train ∪ Val ∪ Test = All={len(all_subjects)}")
        print(f"   ∴ 不存在任何受试者同时出现在训练集和测试集")
        print(f"   ∴ 不存在数据泄露")
    else:
        print(f"⚠️ 验证失败: 发现问题!")
        failed_checks = [check for check in all_checks if not check[1]]
        for check_name, _ in failed_checks:
            print(f"   ❌ {check_name}")
    print(f"=" * 100)
    
    return all_passed


if __name__ == "__main__":
    if len(sys.argv) > 1:
        result_dir = sys.argv[1]
    else:
        result_dir = "/home/szdx/LNX/usage_predict/evaluation_results/run_20260113_164941"
    
    verify_no_overlap(result_dir)
