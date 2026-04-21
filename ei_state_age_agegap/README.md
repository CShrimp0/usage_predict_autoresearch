# EI State Age Agegap

这个目录用于把 EI 直接映射成 18-100 岁范围内的 `state_age`，再计算模型预测年龄与状态年龄之间的差值：

```text
state_age = 18 + (EI - EI_min) / (EI_max - EI_min) * (100 - 18)
agegap = pred_age - state_age
agegap_mae = mean(abs(agegap))
```

默认假设 EI 越高，状态年龄越大。如果你的 EI 定义相反，运行时加 `--reverse`。

## 输入要求

输入表至少需要三类列：

- 真实年龄列：默认自动识别 `age`、`Age`、`true_age`、`年龄`，用于过滤 18-100 岁。
- 预测年龄列：默认自动识别 `prediction`、`pred_age`、`predicted_age`、`预测年龄`。
- EI 列：优先识别 `EI`、`echo_intensity`，也会尝试识别包含 `亮度` / `灰度` 的单个列。多个 EI 候选列时请显式传 `--ei-column`。

## 直接计算

如果预测表里已经有 EI：

```bash
python ei_state_age_agegap/compute_agegap.py \
  --input path/to/predictions_with_ei.csv \
  --ei-column EI \
  --output-dir outputs/ei_state_age_agegap
```

## 从单独 EI 表合并

如果预测结果和 EI 在两个文件里：

```bash
python ei_state_age_agegap/compute_agegap.py \
  --input outputs/run_xxx/tables/predictions.csv \
  --ei-source path/to/ei_table.xlsx \
  --ei-sheet 左连接 \
  --ei-column "股直肌亮度（静息）" \
  --merge-key subject_id:ID \
  --output-dir outputs/ei_state_age_agegap/run_xxx
```

`--merge-key left:right` 表示左侧预测表列和右侧 EI 表列的对应关系；如果两边列名一致，也可以写 `--merge-key subject_id`。

## 输出

脚本会写出：

- `agegap_results.csv`：逐行结果，包含 `state_age`、`agegap`、`agegap_abs`。
- `agegap_subject_results.csv`：如果检测到 `subject_id`，额外输出按人平均后的结果。
- `metrics.json`：row-level 和 subject-level 的 `agegap_mae`、RMSE、bias 等指标。
- `summary.md`：简短可读摘要。

默认 `agegap = pred_age - state_age`，所以正值表示预测年龄比 EI 状态年龄更老。
