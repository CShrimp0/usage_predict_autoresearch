"""
TA超声图像年龄预测数据集加载器（优化版）

功能特性：
- 支持可配置的图像尺寸（224×224, 256×256等）
- 支持按受试者ID划分（防止数据泄露）
- 支持按年龄分层抽样（提高泛化能力）
- 统一的数据增强配置
"""
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict
import warnings


class TAUltrasoundAgeDataset(Dataset):
    """TA超声图像年龄预测数据集"""
    
    def __init__(self, image_paths, ages, transform=None):
        """
        Args:
            image_paths: 图像文件路径列表
            ages: 对应的年龄标签列表
            transform: 图像变换
        """
        self.image_paths = image_paths
        self.ages = ages
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 读取图像（处理中文路径）
        img_path = self.image_paths[idx]
        try:
            # 使用PIL读取，支持中文路径
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # 返回黑色图像作为备用
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        age = self.ages[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(age, dtype=torch.float32)


def get_age_group(age, bin_width=10):
    """
    根据年龄获取年龄组
    
    Args:
        age: 年龄值
        bin_width: 年龄分组宽度（默认10岁）
    
    Returns:
        年龄组标签（如 "0-10", "10-20"）
    """
    lower = int(age // bin_width) * bin_width
    upper = lower + bin_width
    return f"{lower}-{upper}"


def stratified_split_by_age(subject_ids, age_dict, test_size=0.2, val_size=0.1, 
                            random_state=42, bin_width=10):
    """
    按年龄分层抽样划分数据集
    
    Args:
        subject_ids: 受试者ID列表
        age_dict: 受试者ID到年龄的映射 {subject_id: age}
        test_size: 测试集比例
        val_size: 验证集比例（从训练集中划分）
        random_state: 随机种子
        bin_width: 年龄分组宽度（默认10岁）
    
    Returns:
        train_subjects, val_subjects, test_subjects: 三个集合的受试者ID列表
    """
    # 为每个受试者分配年龄组
    subject_age_groups = []
    valid_subjects = []
    
    for sid in subject_ids:
        if sid in age_dict:
            age = age_dict[sid]
            age_group = get_age_group(age, bin_width)
            subject_age_groups.append(age_group)
            valid_subjects.append(sid)
    
    # 统计每个年龄组的样本数
    age_group_counts = defaultdict(int)
    for ag in subject_age_groups:
        age_group_counts[ag] += 1
    
    print(f"\n📊 年龄分层统计（每{bin_width}岁一组）:")
    for age_group in sorted(age_group_counts.keys(), key=lambda x: int(x.split('-')[0])):
        count = age_group_counts[age_group]
        print(f"  {age_group}岁: {count} 个受试者")
    
    # 首先划分训练+验证集 vs 测试集（分层抽样）
    train_val_subjects, test_subjects, _, _ = train_test_split(
        valid_subjects,
        subject_age_groups,
        test_size=test_size,
        random_state=random_state,
        stratify=subject_age_groups
    )
    
    # 再从训练+验证集中划分训练集和验证集（分层抽样）
    train_val_age_groups = [get_age_group(age_dict[sid], bin_width) for sid in train_val_subjects]
    
    # 检查是否有足够的样本进行分层
    group_counts_trainval = defaultdict(int)
    for ag in train_val_age_groups:
        group_counts_trainval[ag] += 1
    
    # 如果某些年龄组样本太少，无法分层，则不使用stratify
    min_samples_per_group = min(group_counts_trainval.values()) if group_counts_trainval else 0
    use_stratify_val = min_samples_per_group >= 2  # 至少需要2个样本才能分层
    
    if use_stratify_val:
        train_subjects, val_subjects, _, _ = train_test_split(
            train_val_subjects,
            train_val_age_groups,
            test_size=val_size,
            random_state=random_state,
            stratify=train_val_age_groups
        )
    else:
        warnings.warn(f"验证集划分时某些年龄组样本过少（最少{min_samples_per_group}个），取消分层抽样")
        train_subjects, val_subjects = train_test_split(
            train_val_subjects,
            test_size=val_size,
            random_state=random_state
        )
    
    return train_subjects, val_subjects, test_subjects


def load_dataset(image_dir, excel_path, test_size=0.2, val_size=0.1, random_state=42,
                image_size=224, use_age_stratify=False, age_bin_width=10,
                min_age=0, max_age=100):
    """
    加载数据集并划分训练、验证、测试集
    
    Args:
        image_dir: 图像文件夹路径
        excel_path: Excel标签文件路径
        test_size: 测试集占总数据的比例
        val_size: 验证集占训练数据的比例
        random_state: 随机种子
        image_size: 图像resize尺寸（默认224）
        use_age_stratify: 是否使用年龄分层抽样（默认False）
        age_bin_width: 年龄分组宽度，仅在use_age_stratify=True时有效（默认10岁）
        min_age: 最小年龄（包含），默认0岁
        max_age: 最大年龄（包含），默认100岁
    
    Returns:
        train_dataset, val_dataset, test_dataset: 训练集、验证集、测试集
    """
    # 读取Excel标签文件
    df = pd.read_excel(excel_path)
    
    # 建立受试者ID到年龄的映射
    age_dict = {}
    
    # 处理Healthy列（第一组数据）
    healthy_df = df[['Healthy', 'Unnamed: 1']].copy()
    healthy_df.columns = ['Number', 'Age']
    healthy_df = healthy_df[1:].dropna()  # 跳过标题行
    
    for _, row in healthy_df.iterrows():
        try:
            subject_id = str(int(float(row['Number'])))
            age = float(row['Age'])
            age_dict[subject_id] = age
        except (ValueError, TypeError):
            continue
    
    # 获取所有图像路径
    image_dir = Path(image_dir)
    all_image_paths = sorted(list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')))
    
    # 按受试者ID分组图像
    subject_images = defaultdict(list)
    for img_path in all_image_paths:
        # 从文件名中提取受试者ID（格式: TAXX_ID_X.png）
        parts = img_path.stem.split('_')
        if len(parts) >= 2:
            subject_id = parts[1]
            if subject_id in age_dict:
                subject_images[subject_id].append(str(img_path))
    
    # 获取所有受试者ID
    all_subjects = list(subject_images.keys())
    
    # 年龄过滤
    if min_age > 0 or max_age < 100:
        filtered_subjects = [sid for sid in all_subjects if min_age <= age_dict[sid] <= max_age]
        filtered_count = len(all_subjects) - len(filtered_subjects)
        if filtered_count > 0:
            print(f"\n🔍 年龄过滤: 保留 {min_age}-{max_age} 岁范围")
            print(f"   过滤前: {len(all_subjects)} 个受试者")
            print(f"   过滤后: {len(filtered_subjects)} 个受试者 (移除 {filtered_count} 个)")
            all_subjects = filtered_subjects
    
    # 统计信息
    total_subjects = len(all_subjects)
    total_images = sum(len(imgs) for imgs in subject_images.values())
    ages_list = [age_dict[sid] for sid in all_subjects]
    
    print(f"找到 {total_subjects} 个受试者，共 {total_images} 张图像")
    print(f"每个受试者图像数: 平均 {total_images/total_subjects:.2f}, "
          f"最少 {min(len(imgs) for imgs in subject_images.values())}, "
          f"最多 {max(len(imgs) for imgs in subject_images.values())}")
    print(f"年龄范围: {min(ages_list):.1f} - {max(ages_list):.1f} 岁")
    print(f"平均年龄: {np.mean(ages_list):.1f} ± {np.std(ages_list):.1f} 岁")
    
    # 划分数据集
    if use_age_stratify:
        print(f"\n🔒 使用按年龄分层抽样划分数据集（每{age_bin_width}岁一组）...")
        train_subjects, val_subjects, test_subjects = stratified_split_by_age(
            all_subjects, age_dict, test_size, val_size, random_state, age_bin_width
        )
    else:
        print(f"\n🔒 按受试者ID划分数据集（防止数据泄露）...")
        # 传统的按ID随机划分
        train_val_subjects, test_subjects = train_test_split(
            all_subjects,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
        
        train_subjects, val_subjects = train_test_split(
            train_val_subjects,
            test_size=val_size,
            random_state=random_state,
            shuffle=True
        )
    
    print(f"\n受试者划分:")
    print(f"  训练集受试者: {len(train_subjects)}")
    print(f"  验证集受试者: {len(val_subjects)}")
    print(f"  测试集受试者: {len(test_subjects)}")
    
    # 收集每个集合的图像和标签
    train_paths, train_ages = [], []
    val_paths, val_ages = [], []
    test_paths, test_ages = [], []
    
    for sid in train_subjects:
        for img_path in subject_images[sid]:
            train_paths.append(img_path)
            train_ages.append(age_dict[sid])
    
    for sid in val_subjects:
        for img_path in subject_images[sid]:
            val_paths.append(img_path)
            val_ages.append(age_dict[sid])
    
    for sid in test_subjects:
        for img_path in subject_images[sid]:
            test_paths.append(img_path)
            test_ages.append(age_dict[sid])
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_paths)} 样本，年龄 {np.mean(train_ages):.1f}±{np.std(train_ages):.1f} 岁")
    print(f"  验证集: {len(val_paths)} 样本，年龄 {np.mean(val_ages):.1f}±{np.std(val_ages):.1f} 岁")
    print(f"  测试集: {len(test_paths)} 样本，年龄 {np.mean(test_ages):.1f}±{np.std(test_ages):.1f} 岁")
    
    # 定义图像变换（支持可配置的图像尺寸）
    print(f"\n使用图像尺寸: {image_size}×{image_size}")
    
    # 构建变换列表
    train_transforms_list = [transforms.Resize((image_size, image_size))]
    eval_transforms_list = [transforms.Resize((image_size, image_size))]
    
    # 训练集增强
    train_transforms_list.extend([
        # 旋转角度和水平翻转在这里控制，后续可以测试不同的增强组合
        transforms.RandomRotation(degrees=10),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        # transforms.Normalize(mean = [0.2637, 0.2637, 0.2637], 
        #                    std  = [0.1449, 0.1449, 0.1449])
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 验证集和测试集不增强
    eval_transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_transform = transforms.Compose(train_transforms_list)
    eval_transform = transforms.Compose(eval_transforms_list)
    
    # 创建数据集
    train_dataset = TAUltrasoundAgeDataset(train_paths, train_ages, train_transform)
    val_dataset = TAUltrasoundAgeDataset(val_paths, val_ages, eval_transform)
    test_dataset = TAUltrasoundAgeDataset(test_paths, test_ages, eval_transform)
    
    return train_dataset, val_dataset, test_dataset


def get_transform(image_size=224, is_train=True):
    """
    获取图像变换
    
    Args:
        image_size: 图像尺寸
        is_train: 是否为训练模式
    
    Returns:
        transform: torchvision变换
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


# 向后兼容：保留旧的接口
def load_dataset_old_interface(base_path, excel_path, test_size=0.2, val_size=0.1, random_state=42):
    """
    旧接口（向后兼容），默认使用224×224，不使用年龄分层
    """
    return load_dataset(
        image_dir=base_path,
        excel_path=excel_path,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        image_size=224,
        use_age_stratify=False
    )


# ========== 多模态数据集 ==========

class MultimodalDataset(Dataset):
    """支持辅助特征的多模态数据集"""
    
    def __init__(self, image_paths, ages, transform=None, aux_feature_extractor=None):
        """
        Args:
            image_paths: 图像文件路径列表
            ages: 对应的年龄标签列表
            transform: 图像变换
            aux_feature_extractor: 辅助特征提取器（AuxiliaryFeatureExtractor实例）
        """
        self.image_paths = image_paths
        self.ages = ages
        self.transform = transform
        self.aux_feature_extractor = aux_feature_extractor
        
        # 过滤缺失辅助特征的样本
        if aux_feature_extractor is not None and aux_feature_extractor.aux_dim > 0:
            valid_indices = []
            valid_subjects = set(aux_feature_extractor.get_valid_subjects())
            
            for idx, img_path in enumerate(image_paths):
                from auxiliary_features import extract_subject_id
                subject_id = extract_subject_id(img_path)
                if subject_id in valid_subjects:
                    valid_indices.append(idx)
            
            # 更新路径和标签
            self.image_paths = [image_paths[i] for i in valid_indices]
            self.ages = [ages[i] for i in valid_indices]
            
            print(f"  过滤后保留 {len(self.image_paths)}/{len(image_paths)} 个样本")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 读取图像
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        age = self.ages[idx]
        
        if self.transform:
            image = self.transform(image)
        
        # 提取辅助特征
        aux_features = None
        if self.aux_feature_extractor is not None and self.aux_feature_extractor.aux_dim > 0:
            from auxiliary_features import extract_subject_id
            subject_id = extract_subject_id(img_path)
            aux_features = self.aux_feature_extractor.extract_features(subject_id, img_path)
        
        # 如果没有辅助特征或提取失败，返回零向量
        if aux_features is None:
            aux_features = torch.zeros(1)
        
        return image, aux_features, torch.tensor(age, dtype=torch.float32)


def load_multimodal_dataset(image_dir, excel_path, test_size=0.2, val_size=0.1, random_state=42,
                            image_size=224, use_age_stratify=False, age_bin_width=10,
                            use_gender=False, use_bmi=False, use_skewness=False, 
                            use_intensity=False, use_clarity=False):
    """
    加载多模态数据集（图像+辅助特征）
    
    Args:
        image_dir: 图像文件夹路径
        excel_path: Excel标签文件路径
        test_size: 测试集占总数据的比例
        val_size: 验证集占训练数据的比例
        random_state: 随机种子
        image_size: 图像resize尺寸（默认224）
        use_age_stratify: 是否使用年龄分层抽样（默认False）
        age_bin_width: 年龄分组宽度（默认10岁）
        use_gender: 是否使用性别特征
        use_bmi: 是否使用BMI特征
        use_skewness: 是否使用偏度特征
        use_intensity: 是否使用平均灰度特征
        use_clarity: 是否使用清晰度特征
    
    Returns:
        train_dataset, val_dataset, test_dataset, aux_dim: 数据集和辅助特征维度
    """
    # 初始化辅助特征提取器
    aux_feature_extractor = None
    aux_dim = 0
    
    if any([use_gender, use_bmi, use_skewness, use_intensity, use_clarity]):
        from auxiliary_features import AuxiliaryFeatureExtractor
        
        print(f"\n🎯 启用辅助特征:")
        if use_gender: print("  - 性别 (2-dim)")
        if use_bmi: print("  - BMI (1-dim)")
        if use_skewness: print("  - 偏度 (1-dim)")
        if use_intensity: print("  - 平均灰度 (1-dim)")
        if use_clarity: print("  - 清晰度 (1-dim)")
        
        aux_feature_extractor = AuxiliaryFeatureExtractor(
            excel_path=excel_path,
            use_gender=use_gender,
            use_bmi=use_bmi,
            use_skewness=use_skewness,
            use_intensity=use_intensity,
            use_clarity=use_clarity
        )
        aux_dim = aux_feature_extractor.aux_dim
        print(f"  总维度: {aux_dim}")
    
    # 读取Excel标签文件
    df = pd.read_excel(excel_path)
    
    # 建立受试者ID到年龄的映射
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
    
    # 获取所有图像路径
    image_dir = Path(image_dir)
    all_image_paths = sorted(list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')))
    
    # 按受试者ID分组图像
    subject_images = defaultdict(list)
    for img_path in all_image_paths:
        parts = img_path.stem.split('_')
        if len(parts) >= 2:
            subject_id = parts[1]
            if subject_id in age_dict:
                subject_images[subject_id].append(str(img_path))
    
    # 过滤有辅助特征的受试者
    if aux_feature_extractor is not None:
        valid_subjects_set = set(aux_feature_extractor.get_valid_subjects())
        subject_images = {sid: imgs for sid, imgs in subject_images.items() 
                         if sid in valid_subjects_set}
        print(f"有效受试者: {len(subject_images)} 个")
    
    all_subjects = list(subject_images.keys())
    
    # 统计信息
    total_subjects = len(all_subjects)
    total_images = sum(len(imgs) for imgs in subject_images.values())
    ages_list = [age_dict[sid] for sid in all_subjects]
    
    print(f"找到 {total_subjects} 个受试者，共 {total_images} 张图像")
    print(f"年龄范围: {min(ages_list):.1f} - {max(ages_list):.1f} 岁")
    
    # 划分数据集（按受试者）
    if use_age_stratify:
        train_subjects, val_subjects, test_subjects = stratified_split_by_age(
            all_subjects, age_dict, test_size, val_size, random_state, age_bin_width
        )
    else:
        train_val_subjects, test_subjects = train_test_split(
            all_subjects, test_size=test_size, random_state=random_state
        )
        train_subjects, val_subjects = train_test_split(
            train_val_subjects, test_size=val_size/(1-test_size), random_state=random_state
        )
    
    # 构建每个集合的图像路径和标签
    def build_split(subjects):
        paths = []
        ages = []
        for sid in subjects:
            for img_path in subject_images[sid]:
                paths.append(img_path)
                ages.append(age_dict[sid])
        return paths, ages
    
    train_paths, train_ages = build_split(train_subjects)
    val_paths, val_ages = build_split(val_subjects)
    test_paths, test_ages = build_split(test_subjects)
    
    print(f"训练集: {len(train_paths)} 张图像 ({len(train_subjects)} 受试者)")
    print(f"验证集: {len(val_paths)} 张图像 ({len(val_subjects)} 受试者)")
    print(f"测试集: {len(test_paths)} 张图像 ({len(test_subjects)} 受试者)")
    
    # 计算辅助特征标准化参数（仅用训练集）
    if aux_feature_extractor is not None:
        print("\n计算辅助特征标准化参数（仅使用训练集）...")
        aux_feature_extractor.set_normalization_params(train_subjects, train_paths)
    
    # 定义图像变换（支持可配置的图像尺寸）
    print(f"\n使用图像尺寸: {image_size}×{image_size}")
    
    # 构建变换列表
    train_transforms_list = [transforms.Resize((image_size, image_size))]
    eval_transforms_list = [transforms.Resize((image_size, image_size))]
    
    # 训练集增强
    train_transforms_list.extend([
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 验证集和测试集不增强
    eval_transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_transform = transforms.Compose(train_transforms_list)
    test_transform = transforms.Compose(eval_transforms_list)
    
    # 创建数据集
    train_dataset = MultimodalDataset(train_paths, train_ages, train_transform, aux_feature_extractor)
    val_dataset = MultimodalDataset(val_paths, val_ages, test_transform, aux_feature_extractor)
    test_dataset = MultimodalDataset(test_paths, test_ages, test_transform, aux_feature_extractor)
    
    return train_dataset, val_dataset, test_dataset, aux_dim
