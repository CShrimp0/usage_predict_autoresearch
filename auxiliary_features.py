"""
多模态特征提取和处理
支持性别、BMI、图像统计特征的提取和标准化
"""

import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import skew
from collections import defaultdict


def compute_image_sharpness(img_array):
    """
    计算图像清晰度（拉普拉斯方差）
    
    Args:
        img_array: numpy数组，可以是灰度或彩色
    
    Returns:
        float: 清晰度分数，值越大越清晰
    """
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    clarity = laplacian.var()
    return clarity


def compute_image_skewness(img_array):
    """
    计算图像偏度
    
    Args:
        img_array: numpy数组
    
    Returns:
        float: 偏度值
    """
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    return skew(gray.flatten())


def compute_image_intensity(img_array):
    """
    计算平均灰度值
    
    Args:
        img_array: numpy数组
    
    Returns:
        float: 平均灰度值 [0-255]
    """
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    return float(gray.mean())


class AuxiliaryFeatureExtractor:
    """辅助特征提取器"""
    
    def __init__(self, 
                 excel_path,
                 use_gender=False,
                 use_bmi=False,
                 use_skewness=False,
                 use_intensity=False,
                 use_clarity=False):
        """
        Args:
            excel_path: Excel文件路径
            use_gender: 是否使用性别
            use_bmi: 是否使用BMI
            use_skewness: 是否使用偏度
            use_intensity: 是否使用平均灰度
            use_clarity: 是否使用清晰度
        """
        self.use_gender = use_gender
        self.use_bmi = use_bmi
        self.use_skewness = use_skewness
        self.use_intensity = use_intensity
        self.use_clarity = use_clarity
        
        # 加载Excel数据
        self._load_excel_data(excel_path)
        
        # 计算维度
        self.aux_dim = self._calculate_dim()
    
    def _load_excel_data(self, excel_path):
        """加载并处理Excel数据"""
        df = pd.read_excel(excel_path)
        
        # 处理Healthy列
        healthy_df = df[['Healthy', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']].copy()
        healthy_df.columns = ['Number', 'Age', 'Length', 'Weight', 'Sex']
        healthy_df = healthy_df[1:].reset_index(drop=True)
        
        # 处理Pathological列
        path_df = df[['Pathological', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10']].copy()
        path_df.columns = ['Number', 'Age', 'Length', 'Weight', 'Sex']
        path_df = path_df[1:].reset_index(drop=True)
        
        # 合并
        self.df = pd.concat([healthy_df, path_df], ignore_index=True)
        self.df = self.df.dropna(subset=['Number'])
        
        # 转换数据类型
        self.df['Number'] = self.df['Number'].astype(str).str.split('.').str[0]
        self.df['Age'] = pd.to_numeric(self.df['Age'], errors='coerce')
        self.df['Length'] = pd.to_numeric(self.df['Length'], errors='coerce')
        self.df['Weight'] = pd.to_numeric(self.df['Weight'], errors='coerce')
        
        # 计算BMI
        self.df['BMI'] = self.df['Weight'] / ((self.df['Length'] / 100) ** 2)
        
        # 过滤异常BMI（<10 或 >60）
        self.df.loc[(self.df['BMI'] < 10) | (self.df['BMI'] > 60), 'BMI'] = np.nan
        
        # 过滤缺失辅助特征的受试者
        if self.use_gender or self.use_bmi:
            self.df = self.df.dropna(subset=['Sex', 'BMI'])
        
        # 构建subject_id到特征的映射
        self.subject_features = {}
        for _, row in self.df.iterrows():
            subject_id = row['Number']
            self.subject_features[subject_id] = {
                'gender': row['Sex'],
                'bmi': row['BMI'],
                'age': row['Age']
            }
        
        # 计算标准化参数（只用训练集，稍后设置）
        self.bmi_mean = None
        self.bmi_std = None
        self.skewness_mean = None
        self.skewness_std = None
        self.intensity_mean = None
        self.intensity_std = None
        self.clarity_mean = None
        self.clarity_std = None
    
    def _calculate_dim(self):
        """计算辅助特征总维度"""
        dim = 0
        if self.use_gender: dim += 2  # One-hot
        if self.use_bmi: dim += 1
        if self.use_skewness: dim += 1
        if self.use_intensity: dim += 1
        if self.use_clarity: dim += 1
        return dim
    
    def set_normalization_params(self, train_subjects, train_image_paths=None):
        """
        根据训练集计算标准化参数
        
        Args:
            train_subjects: 训练集受试者ID列表
            train_image_paths: 训练集图像路径（用于计算图像统计特征）
        """
        # BMI标准化
        if self.use_bmi:
            train_bmis = [self.subject_features[sid]['bmi'] 
                         for sid in train_subjects 
                         if sid in self.subject_features]
            self.bmi_mean = np.mean(train_bmis)
            self.bmi_std = np.std(train_bmis)
        
        # 图像统计特征标准化
        if train_image_paths and (self.use_skewness or self.use_intensity or self.use_clarity):
            skewness_vals = []
            intensity_vals = []
            clarity_vals = []
            
            print("计算训练集图像统计特征...")
            for img_path in train_image_paths[:1000]:  # 采样1000张用于估计
                img = cv2.imread(str(img_path))
                if img is not None:
                    if self.use_skewness:
                        skewness_vals.append(compute_image_skewness(img))
                    if self.use_intensity:
                        intensity_vals.append(compute_image_intensity(img))
                    if self.use_clarity:
                        clarity_vals.append(compute_image_sharpness(img))
            
            if self.use_skewness and skewness_vals:
                self.skewness_mean = np.mean(skewness_vals)
                self.skewness_std = np.std(skewness_vals)
            
            if self.use_intensity and intensity_vals:
                self.intensity_mean = np.mean(intensity_vals)
                self.intensity_std = np.std(intensity_vals)
            
            if self.use_clarity and clarity_vals:
                self.clarity_mean = np.mean(clarity_vals)
                self.clarity_std = np.std(clarity_vals)
    
    def extract_features(self, subject_id, img_path=None):
        """
        提取辅助特征
        
        Args:
            subject_id: 受试者ID
            img_path: 图像路径（用于计算图像统计特征）
        
        Returns:
            torch.Tensor: 辅助特征向量
        """
        if self.aux_dim == 0:
            return None
        
        # 检查subject_id是否存在
        if subject_id not in self.subject_features:
            return None
        
        features = []
        
        # 性别特征（One-hot）
        if self.use_gender:
            gender = self.subject_features[subject_id]['gender']
            gender_onehot = [1.0, 0.0] if gender == 'M' else [0.0, 1.0]
            features.extend(gender_onehot)
        
        # BMI特征（标准化）
        if self.use_bmi:
            bmi = self.subject_features[subject_id]['bmi']
            bmi_norm = (bmi - self.bmi_mean) / (self.bmi_std + 1e-8)
            features.append(bmi_norm)
        
        # 图像统计特征
        if img_path and (self.use_skewness or self.use_intensity or self.use_clarity):
            img = cv2.imread(str(img_path))
            if img is not None:
                if self.use_skewness:
                    skew_val = compute_image_skewness(img)
                    skew_norm = (skew_val - self.skewness_mean) / (self.skewness_std + 1e-8)
                    features.append(skew_norm)
                
                if self.use_intensity:
                    intensity = compute_image_intensity(img)
                    intensity_norm = (intensity - self.intensity_mean) / (self.intensity_std + 1e-8)
                    features.append(intensity_norm)
                
                if self.use_clarity:
                    clarity = compute_image_sharpness(img)
                    clarity_norm = (clarity - self.clarity_mean) / (self.clarity_std + 1e-8)
                    features.append(clarity_norm)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def get_valid_subjects(self):
        """获取有完整辅助特征的受试者ID列表"""
        return list(self.subject_features.keys())
    
    def get_feature_names(self):
        """获取特征名称列表"""
        names = []
        if self.use_gender: names.extend(['gender_M', 'gender_F'])
        if self.use_bmi: names.append('BMI')
        if self.use_skewness: names.append('skewness')
        if self.use_intensity: names.append('intensity')
        if self.use_clarity: names.append('clarity')
        return names


def extract_subject_id(image_path):
    """从图像文件名提取受试者ID"""
    filename = Path(image_path).stem
    parts = filename.split('_')
    if len(parts) >= 2:
        return parts[1]  # anon_ID_N.png -> ID
    return None


if __name__ == '__main__':
    # 测试特征提取器
    extractor = AuxiliaryFeatureExtractor(
        excel_path='/home/szdx/LNX/data/TA/characteristics.xlsx',
        use_gender=True,
        use_bmi=True,
        use_skewness=True,
        use_intensity=True,
        use_clarity=True
    )
    
    print(f"辅助特征维度: {extractor.aux_dim}")
    print(f"特征名称: {extractor.get_feature_names()}")
    print(f"有效受试者数: {len(extractor.get_valid_subjects())}")
    
    # 设置标准化参数（模拟训练集）
    valid_subjects = extractor.get_valid_subjects()[:100]  # 使用前100个作为示例
    test_images = [f'/home/szdx/LNX/data/TA/Healthy/Images/anon_{sid}_1.png' for sid in valid_subjects]
    extractor.set_normalization_params(valid_subjects, test_images)
    
    # 测试特征提取
    test_img = '/home/szdx/LNX/data/TA/Healthy/Images/anon_1001_1.png'
    if Path(test_img).exists():
        features = extractor.extract_features('1001', test_img)
        if features is not None:
            print(f"提取特征: {features}")
            print(f"特征维度: {features.shape}")
        else:
            print("提取失败：受试者不存在")
    else:
        print(f"测试图像不存在: {test_img}")