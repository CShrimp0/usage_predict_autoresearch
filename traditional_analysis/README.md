# 超声图像传统特征分析

本目录包含对肌肉超声图像的非深度学习特征分析，用于研究图像特征与年龄的统计相关性。

## 📌 概述

传统特征分析从图像中提取人工设计的特征，并分析它们与年龄的相关性。这种方法提供：
- **可解释性**: 明确的特征定义和医学意义
- **基线对比**: 与深度学习模型性能对比
- **特征工程**: 为传统机器学习模型提供输入
- **医学验证**: 验证已知的肌肉老化生理学

## 已完成的分析

### 1. 平均像素强度（Mean Pixel Intensity）
- **方法**：计算每张图像的平均灰度值
- **假设**：老年肌肉由于脂肪浸润和纤维化而显示更高的像素强度
- **结果**：所有肌肉均显示出与年龄的正相关（TA: r=0.5581, BB: r=0.3975, GM: r=0.3133）

### 2. 纹理特征（Texture Features）
使用 GLCM（灰度共生矩阵）和统计方法提取：

- **GLCM 特征**：
  - Contrast（对比度）：局部强度变化
  - Correlation（相关性）：灰度级的线性依赖
  - Energy（能量）：灰度分布的均匀性
  - Homogeneity（同质性）：分布接近对角线的程度

- **统计特征**：
  - Mean（均值）：平均像素强度
  - Std Dev（标准差）：强度变异性
  - Skewness（偏度）：强度分布的不对称性（在 TA 中与年龄相关性最强: r=-0.6383）
  - Kurtosis（峰度）：分布的尖峭度
  - Entropy（熵）：随机性/复杂性

### 3. 跨肌肉对比（Cross-Muscle Comparison）
- 比较 TA、GM、BB 三个肌肉的特征相关性
- TA 肌肉显示出最强的年龄相关变化
- 不同肌肉表现出不同的纹理变化模式

## 📁 目录结构

```
traditional_analysis/
├── scripts/
│   ├── compute_pixel_intensity.py    # 平均灰度值分析
│   ├── compute_texture_features.py   # 11个纹理特征分析
│   ├── batch_analyze.py              # 批量处理多个肌肉
│   └── compare_muscles.py            # 跨肌肉对比分析
├── results/
│   ├── TA/                           # 胫骨前肌
│   │   ├── data/
│   │   │   ├── pixel_intensity.csv          # 每张图像的灰度值
│   │   │   ├── texture_features.csv         # 11个纹理特征
│   │   │   ├── correlations.csv             # 特征排名
│   │   │   ├── analysis_summary.txt         # 灰度值分析摘要
│   │   │   └── texture_analysis_summary.txt # 纹理分析摘要
│   │   └── figures/                          # 所有图表300 DPI
│   │       ├── age_vs_intensity.png         # 年龄-灰度值散点图
│   │       ├── texture_features_correlation.png  # 9特征3×3网格
│   │       └── feature_correlation_heatmap.png   # 相关系数热图
│   ├── GM/  （腓肠肌内侧头，相同结构）
│   ├── BB/  （肱二头肌，相同结构）
│   └── comparison/                    # 跨肌肉对比
│       ├── intensity_comparison.png           # 三肌肉灰度对比
│       ├── correlation_heatmap_comparison.png # 特征×肌肉热图
│       ├── top_features_comparison.png        # Top 5特征条形图
│       └── comparison_summary.txt             # 详细对比报告
└── README.md
```

## 🚀 使用方法

### 单个肌肉分析

**平均灰度值分析:**
```bash
python scripts/compute_pixel_intensity.py \
  /path/to/Images/ \
  /path/to/characteristics.xlsx \
  TA
```

**纹理特征分析:**
```bash
python scripts/compute_texture_features.py \
  /path/to/Images/ \
  /path/to/characteristics.xlsx \
  TA
```

### 批量分析

```bash
# 分析所有三个肌肉（TA, GM, BB）
python scripts/batch_analyze.py TA GM BB

# 或只分析特定肌肉
python scripts/batch_analyze.py TA
```

## 📊 关键发现

### 平均灰度值与年龄相关性

| 肌肉 | 样本数 | 平均灰度值 | Pearson r | 相关性强度 |
|------|--------|------------|-----------|------------|
| **胫骨前肌 (TA)** | 3092 | 69.94 ± 8.40 | **0.558** | 强相关 |
| **腓肠肌内侧头 (GM)** | 1480 | 66.36 ± 8.41 | 0.313 | 中等相关 |
| **肱二头肌 (BB)** | 2207 | 68.25 ± 8.93 | 0.398 | 中等相关 |

### 纹理特征 Top 3

#### 胫骨前肌 (TA) 🏆
1. **偏度 (skewness)**: r = -0.638 ⭐（所有分析中最强）
2. **平均灰度 (mean)**: r = 0.558
3. **峰度 (kurtosis)**: r = -0.492

#### 腓肠肌内侧头 (GM)
1. **标准差 (std)**: r = -0.415
2. **偏度 (skewness)**: r = -0.402
3. **GLCM相关性 (correlation)**: r = -0.344

#### 肱二头肌 (BB)
1. **平均灰度 (mean)**: r = 0.398
2. **相异性 (dissimilarity)**: r = 0.382
3. **峰度 (kurtosis)**: r = -0.381

### 通用发现
- **所有肌肉共有的强相关特征**:
  - 偏度 (skewness): 负相关 [-0.638, -0.402, -0.328]
  - 平均灰度 (mean): 正相关 [0.558, 0.313, 0.398]
  
- **肌肉特异性**:
  - 胫骨前肌 (TA) 表现出最强的年龄相关性
  - 偏度和平均灰度在TA中达到|r| > 0.5（强相关）

### 🎯 医学解释

1. **灰度增加（正相关）**: 
   - 脂肪浸润增加
   - 纤维组织增生
   - 肌纤维密度下降
   
2. **偏度减小（负相关）**: 
   - 分布向高灰度值偏移
   - 正常肌肉纹理的丧失
   - 组织异质性增加
   
3. **标准差变化**: 
   - 纹理变得更均匀或更异质
   - 取决于具体肌肉部位
   
4. **肌肉差异**: 
   - 不同肌肉的老化模式不同
   - TA（胫骨前肌）老化特征最明显

## 🔬 特征说明

### 1. 平均灰度值
- **定义**: 图像的平均像素强度值（0-255）
- **医学意义**: 反映肌肉组织的整体回声强度

### 2. 纹理特征（11个）

#### 统计特征（5个）
- **mean**: 平均灰度值
- **std**: 标准差，反映灰度分布离散程度
- **skewness**: 偏度，分布对称性
- **kurtosis**: 峰度，分布尖锐程度
- **entropy**: 熵，图像复杂度/随机性

#### GLCM特征（6个）
基于灰度共生矩阵，描述纹理模式：
- **contrast**: 对比度，局部灰度变化
- **dissimilarity**: 相异性，相邻像素差异
- **homogeneity**: 同质性，纹理均匀程度
- **energy**: 能量，纹理粗细
- **correlation**: 相关性，线性灰度关系
- **ASM**: 角二阶矩，纹理均匀性

## 🔧 技术细节

- **Python版本**: 3.10
- **主要依赖**: numpy, pandas, scikit-image, scipy, matplotlib, seaborn
- **中文字体**: Noto Sans CJK SC
- **图表分辨率**: 300 DPI（适合论文发表）
- **GLCM参数**: 
  - 距离: 1像素
  - 方向: 0°, 45°, 90°, 135°（取平均）
  - 灰度级: 256 → 64（归一化）

## 💡 引用建议

这些传统特征可以作为：
1. **基线模型**: 与深度学习模型对比
2. **可解释性分析**: 理解深度学习学到了什么
3. **特征工程**: 结合传统ML（Ridge/RF/XGBoost）
4. **医学验证**: 验证已知的肌肉老化生理学

## 🚧 下一步

- [ ] 添加更多特征：小波变换、傅里叶频谱、LBP
- [ ] 传统机器学习模型：Ridge回归、随机森林、XGBoost
- [ ] 特征选择和降维：PCA、LASSO
- [ ] 分析RF（股直肌）和VL（股外侧肌）
- [ ] 跨数据集验证

## 📈 与深度学习对比

### 传统特征（最佳结果 - TA肌肉）
- **特征**: 偏度 (skewness)
- **相关性**: r = -0.638
- **R² 等效**: ~0.407（解释了40.7%的年龄方差）

### 深度学习（最佳模型 - TA肌肉）
- **模型**: ResNet34 + Dropout=0.6
- **验证 MAE**: 7.016年
- **总结**: 深度学习性能更优，但传统特征提供可解释性

---
**最后更新**: 2026-01-07  
**分析肌肉**: TA（胫骨前肌）, GM（腓肠肌内侧头）, BB（肱二头肌）  
**总图像数**: 6779张

### 启示
1. 传统特征显示出强相关性，但受限于线性假设
2. 深度学习捕捉非线性模式 → 更好的预测
3. 单一偏度特征就能解释 40% 的方差 - 非常强大！
4. 组合多个传统特征可能接近深度学习性能
