# usage_predict autoresearch

这是把 `karpathy/autoresearch` 的实验方式迁移到 `usage_predict` 上的版本。

## Setup

1. 在分支 `autoresearch/apr7` 上工作。
2. 先完整阅读这些文件：
   - `README.md`
   - `train.py`
   - `model.py`
   - `dataset.py`
   - `auxiliary_features.py`
3. 确认数据存在：
   - 图像目录：`/home/szdx/LNX/data/TA/Healthy/Images`
   - 标签文件：`/home/szdx/LNX/data/TA/characteristics.xlsx`
4. `results.tsv` 只保留为未跟踪文件，不要提交进 git。
5. 第一次运行必须是 baseline，不做任何代码修改。

## Experimentation

每次实验只用单卡，固定 5 分钟训练预算。运行命令：

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/run_autoresearch_trial.sh > run.log 2>&1
```

## You CAN modify

- `train.py`
- `model.py`
- `dataset.py`
- `auxiliary_features.py`

## You CANNOT modify

- `tools/run_autoresearch_trial.sh`
- `evaluate.py`
- `requirements.txt`
- 数据文件、输出目录、评估结果目录
- 任何为了“作弊”而改变验证口径的逻辑

## Goal

目标是把 `best_val_mae` 做到更低。越低越好。

训练结束后，从日志里读结果：

```bash
grep "^best_val_mae:\|^peak_vram_mb:\|^training_seconds:" run.log
```

如果 grep 为空，就认为实验崩了，查看：

```bash
tail -n 50 run.log
```

## Logging

每次实验都追加到 `results.tsv`，使用 tab 分隔，列如下：

```tsv
commit	best_val_mae	memory_gb	status	description
```

- `commit`: 短 commit hash
- `best_val_mae`: 本次最优验证 MAE；崩溃写 `0.000000`
- `memory_gb`: `peak_vram_mb / 1024`，保留 1 位小数；崩溃写 `0.0`
- `status`: `keep`、`discard` 或 `crash`
- `description`: 简短说明本次改了什么

## Loop

1. 看当前 git 分支和 commit。
2. 想一个小而清楚的实验，只改允许修改的文件。
3. 提交代码。
4. 跑 `bash tools/run_autoresearch_trial.sh > run.log 2>&1`
5. 读 `best_val_mae` 和 `peak_vram_mb`
6. 记录到 `results.tsv`
7. 如果 MAE 更低，就保留这个 commit
8. 如果更差或相同，就回退到上一个更好的 commit

优先保留“更简单但不更差”的改动。不要为了极小收益堆太多复杂性。
