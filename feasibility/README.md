# 粗粒度年龄可行性验证（Feasibility Study）

## 1. 科学动机
当前主线任务是超声图像精确年龄回归。本实验先做一个更直接的可行性验证：将年龄划分为粗粒度阶段（青年/中年/老年）进行三分类，检验图像中是否稳定存在可学习的年龄阶段信息。

该实验不替代主线回归，而是为以下方向提供依据：
- 分年龄段建模
- coarse-to-fine 年龄建模
- ordinal / hierarchical age modeling

## 2. 默认年龄分组定义
默认定义如下（可配置）：
- `young`: `18 <= age < 35`（label `0`）
- `middle`: `35 <= age < 60`（label `1`）
- `old`: `age >= 60`（label `2`）

默认仅保留 `18` 岁及以上受试者（`--min-age 18`）。

## 3. 数据划分与防泄漏
- 严格按 `subject ID` 划分 train/val/test，避免同一受试者跨集合。
- 支持分层：`--stratify-mode coarse|age_bin|none`。
  - 默认 `coarse`：按粗年龄类分层。
- 划分统计会写入 `config.json` 的 `dataset.split_info`，包含：
  - 每个集合的 subject 数
  - image 数
  - 三类样本分布

## 4. 训练
最小示例（与你要求一致）：

```bash
python feasibility/train_age_group.py \
  --model resnet50 \
  --batch-size 32 \
  --dropout 0.6 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --epochs 200 \
  --patience 50 \
  --image-size 224 \
  --seed 42 \
  --young-max 35 \
  --middle-max 60
```

训练输出目录：
- `outputs/feasibility/run_YYYYMMDD_HHMMSS/`

训练输出文件：
- `best_model.pth`
- `history.json`
- `config.json`
- `command.sh`

训练指标：
- `train_loss`
- `val_loss`
- `train_acc`
- `val_acc`
- `val_macro_f1`

损失函数：`CrossEntropyLoss`，支持类别权重：
- `--class-weights auto`（默认，按 train set 自动计算）
- `--class-weights none`
- `--class-weights manual --class-weight-values w0 w1 w2`

## 5. 评估
最小示例：

```bash
python feasibility/evaluate_age_group.py \
  --checkpoint outputs/feasibility/run_xxx/best_model.pth
```

说明：
- 评估脚本会自动读取 checkpoint 内训练参数，重建与训练一致的 split 参数，避免 train/test split 不一致。
- 默认输出到：`<checkpoint_dir>/age_group_eval/`。

评估输出文件：
- `test_metrics.json`
- `test_metrics_full.json`
- `predictions.json`
- `predictions_readable.csv`
- `confusion_matrix.png`
- `confusion_matrix.txt`
- `classification_report.txt`
- `misclassified_samples.txt`
- `inference_summary.md`

评估指标：
- accuracy
- balanced accuracy
- macro precision
- macro recall
- macro F1
- weighted F1

## 6. 输出文件说明
- `test_metrics.json`
  - 紧凑版指标（默认阅读入口）
  - overall 指标、类别指标、错分模式、边界敏感性、置信度分桶
  - `dataset_split_info` 为简化版（不含冗长 subject 列表）
- `test_metrics_full.json`
  - 完整版指标（含完整 `dataset_split_info.subject_ids`）
- `predictions.json`
  - 每条样本包含：
    - `filename`
    - `true_age`
    - `true_label`
    - `true_label_name`
    - `pred_label`
    - `pred_label_name`
    - `pred_probs`
    - `confidence`
- `predictions_readable.csv`
  - 便于直接筛选/排序查看的可读表格版本
  - 包含类别名、置信度、置信度边际、距年龄边界距离、各类概率
- `confusion_matrix.txt`
  - 混淆矩阵文本版（计数 + 行归一化百分比）
- `misclassified_samples.txt`
  - 错分样本清单，便于人工复核
- `inference_summary.md`
  - 一页式评估摘要（overall、per-class、主要错分模式、边界敏感性、置信度分桶）

## 7. 结果解读建议
- 若 coarse classification 效果较好（尤其 macro F1、balanced accuracy 稳定），说明超声图像中粗粒度年龄阶段信息是可学的。
- 若 coarse classification 也较差，则应谨慎看待“直接做精确年龄回归”的任务设定，优先考虑分层建模或重新检查数据质量与标注噪声。

## 8. 一键脚本
可用：

```bash
bash feasibility/run_feasibility_experiment.sh
```

脚本会先训练，再自动对最新 run 的 `best_model.pth` 做评估。
