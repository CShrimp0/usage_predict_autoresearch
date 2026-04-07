# usage_predict_autoresearch

这个目录是 `usage_predict` 的纯代码副本，用来按 `karpathy/autoresearch` 的方式做自动实验。

## 已做的适配

- `train.py` 支持 `--time-budget-minutes`
- 训练结束会打印可 grep 的摘要：
  - `best_val_mae: ...`
  - `training_seconds: ...`
  - `peak_vram_mb: ...`
- `tools/run_autoresearch_trial.sh` 固定了一条 baseline 命令
- `program.md` 规定了 agent 的实验边界
- `results.tsv` 已初始化为表头

## 直接运行 baseline

```bash
cd /home/szdx/LNX/usage_predict_autoresearch
CUDA_VISIBLE_DEVICES=0 bash tools/run_autoresearch_trial.sh > run.log 2>&1
grep "^best_val_mae:\|^peak_vram_mb:\|^training_seconds:" run.log
```

默认 Python 使用：

```bash
/home/szdx/anaconda3/envs/us/bin/python
```

如果你想换环境：

```bash
PYTHON_BIN=/path/to/python CUDA_VISIBLE_DEVICES=0 bash tools/run_autoresearch_trial.sh
```
