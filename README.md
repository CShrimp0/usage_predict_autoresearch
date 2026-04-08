# usage_predict_autoresearch

这是把 `/home/szdx/LNX/usage_predict` 收缩成 `karpathy/autoresearch` 风格后的单卡实验仓库。

核心逻辑只看三个文件：

- `prepare.py`
- `train.py`
- `program.md`

## 设计原则

- `prepare.py` 是固定实验框架
  - 固定 5 分钟训练预算
  - 固定数据读取和受试者级划分
  - 固定最终评估方式
- `train.py` 是唯一允许被 LLM 修改的文件
  - 模型结构
  - 优化器 / scheduler
  - loss
  - 正则化
  - 通过配置暴露的数据增强参数
- `program.md` 定义 autoresearch 的运行规则

## 数据路径

- 图像目录：`/home/szdx/LNX/data/TA/Healthy/Images`
- 标签文件：`/home/szdx/LNX/data/TA/characteristics.xlsx`

## 运行方式

单卡运行：

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

读取关键结果：

```bash
grep "^prediction_mae:\|^best_val_mae:\|^peak_vram_mb:" run.log
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
- `outputs/autoresearch/run_*/test_eval.json`

## 当前默认 baseline

默认 `ExperimentConfig` 采用当前较强的多模态 ResNet50 基线：

- `model = "resnet50"`
- `dropout = 0.6`
- `loss = "mae"`
- `batch_size = 32`
- `lr = 1e-4`
- 启用辅助特征：`gender + bmi + skewness + intensity + clarity`

后续 autoresearch 直接围绕 `train.py` 继续做实验。
