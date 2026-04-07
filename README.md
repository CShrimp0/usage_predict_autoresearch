# 超声图像年龄预测（usage_predict）

简洁、可复现的超声图像年龄回归训练框架，支持单/多GPU训练、年龄分层抽样、Top‑3 checkpoint 保存与可复现的命令化实验。

## 🚀 快速开始

### 环境
```bash
# 创建并激活环境
conda create -n us python=3.10 -y
conda activate us

# 安装依赖
pip install -r requirements.txt
```

### 数据准备
- 将图像放入 `data/TA/Healthy/Images`（或在命令行指定 `--image-dir`）
- Excel 标签文件放入 `data/TA/characteristics.xlsx`（或用 `--excel-path` 指定）

### Baseline 训练（完整示例）
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model resnet50 \
  --batch-size 32 \
  --dropout 0.5 \
  --lr 0.0001 \
  --weight-decay 0.0001 \
  --loss mae \
  --epochs 500 \
  --patience 100 \
  --image-size 224 \
  --age-bin-width 10 \
  --seed 42 \
  --num-workers 8
```

### 评估模型
```bash
# 基本评估（自动从checkpoint读取训练参数）
python evaluate.py --checkpoint outputs/run_xxx/best_model.pth

# 指定输出目录
python evaluate.py --checkpoint outputs/run_xxx/best_model.pth --output-dir my_results

# 限制年龄范围评估（例如：只评估18-88岁）
python evaluate.py \
  --checkpoint outputs/run_xxx/best_model.pth \
  --min-age 18 \
  --max-age 88

# 对比多个模型的评估结果
python tools/compare_evaluations.py evaluation_results/*/test_metrics.json

# 可视化错误样本（生成交互式HTML报告）
python tools/analyze_error_samples.py \
  --result-dir evaluation_results/run_xxx \
  --image-dir data/TA \
  --max-samples 30
# 在VS Code中右键error_analysis_report.html → "Open with Live Server"
```

**评估输出文件**:
- `test_metrics.json` - 结构化评估指标（含元数据、模型配置、年龄段分析）
- `predictions.json` - 详细预测结果（每个样本）
- `high_error_samples.txt` / `low_error_samples.txt` - 误差样本列表（⚠️标记异常值）
- `image_feature_analysis.txt` - 图像特征对比分析
- `*.png` - 可视化图表（散点图、Bland-Altman图等）
- `error_analysis_report.html` - 交互式误差分析报告

详见: [docs/TEST_METRICS_FORMAT.md](docs/TEST_METRICS_FORMAT.md)

## 📁 项目结构

### 核心脚本
```
usage_predict/
├── train.py                      # 训练脚本（支持DDP、年龄分层）
├── evaluate.py                   # 评估脚本（含Grad-CAM热力图生成）
├── dataset.py                    # 数据加载（受试者级划分、防数据泄露）
├── model.py                      # 模型定义（ResNet50/EfficientNet/ConvNeXt）
├── auxiliary_features.py         # 多模态辅助特征提取
└── requirements.txt              # Python依赖
```

### 工具脚本
```
├── tools/analyze_error_samples.py      # 交互式错误分析（生成HTML报告）
├── tools/compare_evaluations.py        # 多模型评估结果对比
├── tools/summarize_ablation_results.py # 消融实验结果汇总
├── tools/verify_no_leakage.py          # 数据泄露验证（重新执行划分逻辑）
└── tools/run_ablation_study.sh         # 批量消融实验脚本
```

### 文档
```
├── README.md                     # 项目说明（本文件）
├── GRADCAM.md                    # Grad-CAM热力图使用指南
├── DATA_LEAKAGE_REPORT.md        # 数据泄露检查报告
├── ABLATION_RESULTS.md           # 消融实验结果
└── MULTIMODAL_GUIDE.md           # 多模态训练指南
```

### 数据与输出
```
├── data/                         # 数据集目录
│   └── TA/Healthy/
│       ├── Images/               # 原始图像
│       └── Masks/                # 标注文件
├── outputs/                      # 训练输出（每次运行生成独立文件夹）
└── evaluation_results/           # 评估结果（包含热力图、预测文件等）
```

## 📊 最佳模型

### 单模态 Baseline
- **验证集MAE**: 7.04 years
- **架构**: ResNet50 (ImageNet预训练)
- **配置**: dropout=0.6, lr=1e-4, seed=42

### 多模态 Late Fusion（最佳）🏆
- **验证集MAE**: **6.99 years** (+0.7%)
- **架构**: ResNet50 + 辅助特征分支
- **辅助特征**: 性别+BMI+偏度+灰度+清晰度 (6维)
- **隐藏层维度**: 32

详细消融实验结果见 [ABLATION_RESULTS.md](ABLATION_RESULTS.md)

## 📚 详细文档

| 文档 | 说明 |
|------|------|
| [ABLATION_RESULTS.md](ABLATION_RESULTS.md) | 多模态消融实验结果 |
| [MULTIMODAL_GUIDE.md](MULTIMODAL_GUIDE.md) | 多模态Late Fusion详细指南 |
| [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) | 训练参数详解和使用指南 |
| [docs/DATASET_OPTIMIZATION.md](docs/DATASET_OPTIMIZATION.md) | 数据集划分和增强策略 |
| [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) | 项目结构和文件说明 |
| [docs/TEST_METRICS_FORMAT.md](docs/TEST_METRICS_FORMAT.md) | 评估结果格式说明和使用示例 |
| [docs/ERROR_VISUALIZATION_GUIDE.md](docs/ERROR_VISUALIZATION_GUIDE.md) | 错误分析可视化指南 |

## 🔧 工具脚本

```bash
# 查看项目结构
bash scripts/show_structure.sh

# 数据集分析
python scripts/analyze_dataset.py

# 验证数据泄漏
python tools/verify_no_leakage.py

# 可视化图像尺寸
python scripts/visualize_image_sizes.py

# 绘制误差分析图
python scripts/plot_age_error.py

# 错误样本可视化（HTML报告）
python tools/analyze_error_samples.py --help
```

## 🎯 核心特性

### 数据增强
- ✅ RandomRotation(±10°)
- ✅ RandomHorizontalFlip(p=0.5) - 默认启用
- ✅ ColorJitter(亮度/对比度 ±0.2)

> 注：如需禁用水平翻转，请手动修改 `dataset.py` 第354行

### 训练策略
- **损失函数**: MAE/MSE/SmoothL1/Huber 可选
- **优化器**: AdamW（默认 lr=1e-4）
- **学习率调度**: CosineAnnealingLR（支持线性 warmup，默认 warmup 5 epochs）
- **数据划分**: 按 subject ID 分组（防止数据泄漏）
- **年龄分层**: 支持按 10 岁分组的分层抽样（默认启用）

### 模型架构
- ResNet50 (默认)
- EfficientNet-B0/B1
- ConvNeXt-Tiny
- MobileNetV3-Large
- RegNet

### 多模态Late Fusion（新功能）🆕

支持融合图像特征与辅助特征进行年龄预测，采用Late Fusion架构：

**架构设计**:
- **图像分支**: ResNet50 → 2048-dim features
- **辅助分支**: 辅助特征 → 32-dim hidden (with BN, ReLU, Dropout)
- **融合层**: Concatenate → 256 → 128 → 1

**支持的辅助特征**:
- **性别** (2-dim): One-hot编码，Male=[1,0], Female=[0,1]
- **BMI** (1-dim): 身体质量指数，标准化处理
- **偏度** (1-dim): 图像灰度分布偏度
- **平均灰度** (1-dim): 图像平均亮度
- **清晰度** (1-dim): 拉普拉斯方差，衡量图像锐度

**使用示例**:
```bash
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
  --aux-clarity \
  --aux-hidden-dim 32 \
  --output-dir ./outputs/multimodal_all

# 仅使用人口学特征（性别+BMI）
python train.py --use-aux-features --aux-gender --aux-bmi

# 仅使用图像统计特征
python train.py --use-aux-features --aux-skewness --aux-intensity --aux-clarity
```

**消融实验结果**:

| 排名 | 配置 | 辅助特征 | MAE | 相比Baseline |
|:----:|------|----------|:---:|:------------:|
| 🥇 1 | 全部特征 | 性别+BMI+偏度+灰度+清晰度 | **6.99** | **+0.7%** |
| 2 | Baseline | 无 | 7.04 | - |
| 3 | 全部+hidden64 | 性别+BMI+偏度+灰度+清晰度 | 7.09 | -0.6% |
| 4 | 仅性别 | 性别 | 7.15 | -1.5% |
| 5 | 仅偏度 | 偏度 | 7.17 | -1.9% |

> **关键发现**: 只有全部特征组合才能超越baseline，单独特征反而降低性能。详见 [ABLATION_RESULTS.md](ABLATION_RESULTS.md)

**消融实验**:
```bash
# 运行完整消融实验（10个配置）
bash tools/run_ablation_study.sh

# 查看结果汇总
python tools/summarize_ablation_results.py
```
- `--use-aux-features`: 启用辅助特征（必选）
- `--aux-gender`: 使用性别特征
- `--aux-bmi`: 使用BMI特征
- `--aux-skewness`: 使用偏度特征
- `--aux-intensity`: 使用平均灰度特征
- `--aux-clarity`: 使用清晰度特征
- `--aux-hidden-dim`: 辅助分支隐藏层维度（默认32，推荐）

**注意事项**:
- 辅助特征自动从Excel文件读取并标准化（仅使用训练集统计量）
- 缺失值样本会自动过滤，不影响源数据
- 图像统计特征实时计算（首次加载较慢）
- BMI异常值（<10或>60）自动过滤

## 📈 数据集统计

**TA肌肉数据集**:
- **受试者**: 1,021人
- **图像**: 3,092张
- **年龄范围**: 0.0 - 88.4 岁
- **数据划分**: 训练 2,225 / 验证 402 / 测试 465（按受试者，防止泄漏）

## 💡 常见问题

**Q: 如何继续训练？**
```bash
python train.py --resume outputs/run_xxx/checkpoint_epoch_50.pth
```

**Q: 如何查看训练历史？**
```bash
cat outputs/run_xxx/history.json
```

**Q: 如何使用不同损失函数？**
```bash
python train.py --loss mse  # 或 smoothl1, huber
```

**Q: 如何调整学习率？**
```bash
python train.py --lr 0.0001
```

## 📞 技术支持

- 训练问题: 查看 [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)
- 数据问题: 查看 [docs/DATASET_OPTIMIZATION.md](docs/DATASET_OPTIMIZATION.md)
- 项目结构: 查看 [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)

---

**最后更新**: 2026-01-09  
**版本**: v2.0 (多模态Late Fusion)
