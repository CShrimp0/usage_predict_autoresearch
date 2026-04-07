# 多模态Late Fusion使用指南

## 概述

该项目实现了多模态Late Fusion架构，融合超声图像特征与辅助特征（性别、BMI、图像统计特征）进行年龄预测。

## 架构设计

```
输入:
  ├─ 图像 (3×224×224)
  └─ 辅助特征 (最多6维)

图像分支:
  ResNet50 → 2048-dim features

辅助分支:
  辅助特征 (6-dim) → Linear(6→32) → BN → ReLU → Dropout → Linear(32→32) → BN → ReLU

融合层:
  Concat[2048+32] → Linear(2080→256) → ReLU → Dropout → Linear(256→128) → Dropout → Linear(128→1)

输出:
  年龄预测 (1-dim)
```

## 支持的辅助特征

| 特征 | 维度 | 说明 | 来源 |
|------|------|------|------|
| 性别 | 2 | One-hot编码 [Male, Female] | Excel文件 |
| BMI | 1 | 身体质量指数，标准化 | Excel文件计算 |
| 偏度 | 1 | 图像灰度分布偏度 | 实时计算 |
| 平均灰度 | 1 | 图像平均亮度 | 实时计算 |
| 清晰度 | 1 | 拉普拉斯方差 | 实时计算 |

## 快速开始

### 1. 基础用法

```bash
# Baseline（无辅助特征）
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model resnet50 \
  --batch-size 32 \
  --dropout 0.6 \
  --lr 0.0001 \
  --weight-decay 0.0001 \
  --patience 100 \
  --image-size 224 \
  --seed 42

# 使用所有辅助特征
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model resnet50 \
  --batch-size 32 \
  --dropout 0.6 \
  --lr 0.0001 \
  --weight-decay 0.0001 \
  --patience 100 \
  --image-size 224 \
  --seed 42 \
  --use-aux-features \
  --aux-gender \
  --aux-bmi \
  --aux-skewness \
  --aux-intensity \
  --aux-clarity
```

### 2. 消融实验

```bash
# 运行完整消融实验（10个配置）
bash tools/run_ablation_study.sh

# 查看结果汇总
python tools/summarize_ablation_results.py
```

消融实验包括：
1. Baseline（无辅助特征）
2. 仅性别
3. 仅BMI
4. 性别+BMI
5. 仅图像统计特征（偏度+灰度+清晰度）
6. 所有特征
7. 仅偏度
8. 仅清晰度
9. 仅平均灰度
10. 所有特征+更大隐藏层(64)

## 参数说明

### 多模态专用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use-aux-features` | flag | False | 启用辅助特征（必选） |
| `--aux-gender` | flag | False | 使用性别特征 |
| `--aux-bmi` | flag | False | 使用BMI特征 |
| `--aux-skewness` | flag | False | 使用偏度特征 |
| `--aux-intensity` | flag | False | 使用平均灰度特征 |
| `--aux-clarity` | flag | False | 使用清晰度特征 |
| `--aux-hidden-dim` | int | 32 | 辅助分支隐藏层维度 |

### 示例组合

```bash
# 仅人口学特征
python train.py --use-aux-features --aux-gender --aux-bmi

# 仅图像统计特征
python train.py --use-aux-features --aux-skewness --aux-intensity --aux-clarity

# 性别+图像统计
python train.py --use-aux-features --aux-gender --aux-skewness --aux-clarity

# 自定义隐藏层维度
python train.py --use-aux-features --aux-gender --aux-bmi --aux-hidden-dim 64
```

## 数据处理流程

### 1. 辅助特征加载

- **性别和BMI**: 从`characteristics.xlsx`读取
  - 支持Healthy/Pathological双列格式
  - BMI自动计算：`BMI = weight / (height/100)²`
  - 异常值过滤：BMI<10或>60

- **图像统计特征**: 实时计算
  - 偏度：`scipy.stats.skew(grayscale.flatten())`
  - 平均灰度：`grayscale.mean()`
  - 清晰度：`cv2.Laplacian(grayscale).var()`

### 2. 标准化

**关键点**：所有标准化参数仅使用训练集计算

```python
# BMI标准化
bmi_normalized = (bmi - train_bmi_mean) / train_bmi_std

# 图像统计特征标准化
skewness_norm = (skewness - train_skew_mean) / train_skew_std
intensity_norm = (intensity - train_int_mean) / train_int_std
clarity_norm = (clarity - train_clar_mean) / train_clar_std
```

### 3. 缺失值处理

- 在Dataset.__getitem__中过滤，不删除源数据
- 缺失样本自动跳过，不计入训练
- TA数据集缺失率：0.08% (1/1223)

## 文件结构

```
usage_predict/
├── auxiliary_features.py          # 辅助特征提取器
├── dataset.py                     # 多模态数据集 (MultimodalDataset)
├── model.py                       # 多模态模型 (FlexibleMultimodalModel)
├── train.py                       # 训练脚本（已支持多模态）
├── tools/run_ablation_study.sh          # 消融实验脚本
├── tools/summarize_ablation_results.py  # 结果汇总脚本
├── test_dataset_loading.py        # 数据加载测试
└── test_multimodal.sh             # 快速测试脚本
```

## 实现细节

### 模型架构 (FlexibleMultimodalModel)

```python
class FlexibleMultimodalModel(nn.Module):
    def __init__(self, backbone='resnet50', aux_input_dim=6, aux_hidden_dim=32, dropout=0.5):
        # 图像分支
        self.backbone = ResNet50(pretrained=True)
        self.backbone.fc = nn.Identity()  # 2048-dim
        
        # 辅助分支
        self.aux_branch = nn.Sequential(
            nn.Linear(aux_input_dim, aux_hidden_dim),
            nn.BatchNorm1d(aux_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(aux_hidden_dim, aux_hidden_dim),
            nn.BatchNorm1d(aux_hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 融合头
        self.fusion_head = nn.Sequential(
            nn.Linear(2048 + aux_hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1)
        )
    
    def forward(self, image, aux_features):
        img_feat = self.backbone(image)          # (B, 2048)
        aux_feat = self.aux_branch(aux_features) # (B, 32)
        fused = torch.cat([img_feat, aux_feat], dim=1)  # (B, 2080)
        output = self.fusion_head(fused)         # (B, 1)
        return output
```

### 数据集 (MultimodalDataset)

```python
class MultimodalDataset(Dataset):
    def __init__(self, image_paths, ages, transform=None, aux_feature_extractor=None):
        # 过滤缺失辅助特征的样本
        # 保存提取器引用
        
    def __getitem__(self, idx):
        # 加载图像
        image = load_and_transform(self.image_paths[idx])
        
        # 提取辅助特征
        subject_id = extract_subject_id(self.image_paths[idx])
        aux_features = self.aux_feature_extractor.extract_features(subject_id, image_path)
        
        # 返回(image, aux_features, age)
        return image, aux_features, age
```

## 预期性能

基于TA muscle数据集（1021受试者，3092图像）：

| 配置 | 预期MAE | 说明 |
|------|---------|------|
| Baseline | ~7.02 | 仅图像特征 |
| +性别+BMI | ~6.7-6.8 | 添加人口学特征 |
| +所有特征 | ~6.2-6.4 | Late Fusion完整配置 |

具体结果取决于：
- 数据集大小和质量
- 超参数配置
- 训练随机性

## 调试和测试

### 1. 快速测试

```bash
# 测试数据加载（无训练）
python test_dataset_loading.py

# 快速训练测试（2 epochs）
bash test_multimodal.sh
```

### 2. 验证辅助特征提取

```python
from auxiliary_features import AuxiliaryFeatureExtractor

# 初始化
extractor = AuxiliaryFeatureExtractor(
    excel_path='data/TA/characteristics.xlsx',
    use_gender=True,
    use_bmi=True,
    use_skewness=True,
    use_intensity=True,
    use_clarity=True
)

# 查看维度
print(f"维度: {extractor.aux_dim}")
print(f"特征名: {extractor.get_feature_names()}")

# 设置标准化参数（需要训练集subject列表）
extractor.set_normalization_params(train_subjects, train_image_paths)

# 提取特征
features = extractor.extract_features(subject_id='1001', img_path='path/to/image.png')
print(features)  # tensor([1.0, 0.0, 0.29, -0.13, 0.28, 0.14])
```

### 3. 检查config.json

训练后查看`outputs/run_xxx/config.json`确认配置：

```json
{
  "use_aux_features": true,
  "aux_gender": true,
  "aux_bmi": true,
  "aux_skewness": true,
  "aux_intensity": true,
  "aux_clarity": true,
  "aux_hidden_dim": 32,
  "best_val_mae": 6.35,
  ...
}
```

## 常见问题

### Q1: 缺失辅助特征怎么办？

**A**: 自动过滤。TA数据集只有0.08%缺失，影响极小。如需保留更多样本：
- 仅启用性别和BMI（缺失率更低）
- 或修改`auxiliary_features.py`填充缺失值

### Q2: 图像统计特征计算慢？

**A**: 首次加载需计算1000张图像统计量（~10-30秒），用于标准化。后续不再重新计算。
- 如需加速：减少`set_normalization_params()`中的采样数量
- 建议：保持1000张以获得稳定的统计估计

### Q3: 多模态模型比baseline慢多少？

**A**: 几乎无影响（<5%）：
- 辅助分支极轻量（6→32→32）
- 图像统计特征实时计算（<1ms）
- 主要时间仍在图像backbone

### Q4: 如何确认使用了多模态？

**A**: 检查3个地方：
1. 训练日志显示"使用辅助特征，总维度: X"
2. config.json中`use_aux_features: true`
3. 模型结构显示FlexibleMultimodalModel

### Q5: 能否只用部分特征？

**A**: 完全可以！任意组合：
```bash
# 示例：只用性别和清晰度
python train.py --use-aux-features --aux-gender --aux-clarity
```

## 引用和参考

**Late Fusion架构**:
- 图像分支和辅助分支独立训练特征表示
- 在高层进行特征融合
- 相比Early Fusion更灵活，易于消融分析

**相关工作**:
- Multimodal Deep Learning (Ngiam et al., 2011)
- Late Fusion for Multimedia (Snoek et al., 2005)
- Clinical Multimodal Learning (Huang et al., 2020)

---

**最后更新**: 2025-01-07  
**版本**: v2.0 (多模态)
