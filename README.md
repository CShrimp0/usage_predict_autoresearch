# usage_predict_autoresearch

这是把 `/home/szdx/LNX/usage_predict` 收缩成 `karpathy/autoresearch` 风格后的单卡实验仓库。

核心实验逻辑只看三个文件：

- `prepare.py`
- `train.py`
- `program.md`

## 设计原则

- `prepare.py` 是固定实验框架
  - 固定 5 分钟训练预算
  - 固定数据读取和受试者级划分
  - 固定验证 / 测试划分方式
- `train.py` 是唯一允许被 LLM 修改的文件
  - 模型结构
  - 优化器 / scheduler
  - loss
  - 正则化
  - 通过配置暴露的数据增强参数
- `program.md` 定义 autoresearch 的运行规则

另外保留一个固定的 `evaluate.py`，用于对候选 checkpoint 做显式最终测试评估。

## 数据路径

- 图像目录：`/home/szdx/LNX/data/TA/Healthy/Images`
- 标签文件：`/home/szdx/LNX/data/TA/characteristics.xlsx`

## 运行方式

常规单卡训练（验证集驱动）：

```bash
cd /home/szdx/LNX/usage_predict_autoresearch
conda activate us
CUDA_VISIBLE_DEVICES=0 python train.py > run.log 2>&1
```

启动完整 autoresearch：

```bash
cd /home/szdx/LNX/usage_predict_autoresearch
bash run_autoresearch.sh apr8
```

其中 `apr8` 是本轮实验分支 tag，会创建或切换到 `autoresearch/apr8`。

读取常规训练的关键结果：

```bash
grep "^run_mode:\|^best_val_mae:\|^peak_vram_mb:" run.log
```

对 shortlist 候选做显式最终测试评估：

```bash
conda activate us
RUN_FINAL_EVAL=1 CUDA_VISIBLE_DEVICES=0 python train.py --final-eval > run.log 2>&1
grep "^run_mode:\|^prediction_mae:\|^best_val_mae:\|^peak_vram_mb:" run.log
```

独立评估已有 checkpoint：

```bash
conda activate us
python evaluate.py --checkpoint outputs/autoresearch/run_xxx/best_model.pth
```

输出会写到：

- `outputs/autoresearch/run_*/best_model.pth`
- `outputs/autoresearch/run_*/history.json`
- `outputs/autoresearch/run_*/metrics.json`
- `outputs/autoresearch/run_*/test_eval.json`（仅在显式 final eval 时生成）
- `results_v2.tsv`（新的验证驱动实验日志）

## 当前默认 baseline

默认 `ExperimentConfig` 现在与 `program.md` 中的 golden anchor 对齐：

- `model = "resnet50"`
- `pretrained = True`
- `batch_size = 8`
- `optimizer = "adamw"`
- `lr = 3.893e-05`
- `weight_decay = 4.914e-05`
- `dropout = 0.4125`
- `scheduler = "plateau"`
- `lr_factor = 0.696`
- `lr_patience = 4`
- `lr_min = 3.36e-06`
- `warmup_epochs = 5`
- `ema_decay = 0.999`
- `loss = "mae"`
- 启用辅助特征：`gender + bmi + skewness + intensity + clarity`
- 默认 head 也回到更朴素的 `256 -> 128 -> 1`

## 新的 autoresearch 工作流

- 日常 autoresearch 只看 `best_val_mae`
- `train.py` 默认不会在每次运行后评估 test split
- test set 只用于少量 shortlist 候选的最终确认
- `results.tsv` 保留历史记录，新的搜索请写入 `results_v2.tsv`

后续 autoresearch 直接围绕 `train.py` 做小步、可解释、验证集驱动的实验。
