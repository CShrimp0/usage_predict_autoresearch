#!/bin/bash
# 多模态Late Fusion消融实验脚本
# 对比不同辅助特征组合的效果

# 基础参数
GPU=0
MODEL="resnet50"
BATCH_SIZE=32
DROPOUT=0.6
LR=0.0001
WD=0.0001
PATIENCE=100
IMG_SIZE=224
SEED=42

echo "========================================="
echo "多模态Late Fusion消融实验"
echo "========================================="
echo ""

# 实验1：Baseline（无辅助特征）
echo "实验1: Baseline (无辅助特征)"
echo "-----------------------------------------"
CUDA_VISIBLE_DEVICES=$GPU python train.py \
  --model $MODEL --batch-size $BATCH_SIZE --dropout $DROPOUT \
  --lr $LR --weight-decay $WD --patience $PATIENCE \
  --image-size $IMG_SIZE --seed $SEED \
  --output-dir ./outputs/ablation/01_baseline
echo ""

# 实验2：仅性别
echo "实验2: 仅性别特征"
echo "-----------------------------------------"
CUDA_VISIBLE_DEVICES=$GPU python train.py \
  --model $MODEL --batch-size $BATCH_SIZE --dropout $DROPOUT \
  --lr $LR --weight-decay $WD --patience $PATIENCE \
  --image-size $IMG_SIZE --seed $SEED \
  --use-aux-features --aux-gender \
  --output-dir ./outputs/ablation/02_gender_only
echo ""

# 实验3：仅BMI
echo "实验3: 仅BMI特征"
echo "-----------------------------------------"
CUDA_VISIBLE_DEVICES=$GPU python train.py \
  --model $MODEL --batch-size $BATCH_SIZE --dropout $DROPOUT \
  --lr $LR --weight-decay $WD --patience $PATIENCE \
  --image-size $IMG_SIZE --seed $SEED \
  --use-aux-features --aux-bmi \
  --output-dir ./outputs/ablation/03_bmi_only
echo ""

# 实验4：性别 + BMI
echo "实验4: 性别 + BMI"
echo "-----------------------------------------"
CUDA_VISIBLE_DEVICES=$GPU python train.py \
  --model $MODEL --batch-size $BATCH_SIZE --dropout $DROPOUT \
  --lr $LR --weight-decay $WD --patience $PATIENCE \
  --image-size $IMG_SIZE --seed $SEED \
  --use-aux-features --aux-gender --aux-bmi \
  --output-dir ./outputs/ablation/04_gender_bmi
echo ""

# 实验5：仅图像统计特征
echo "实验5: 仅图像统计特征 (偏度+平均灰度+清晰度)"
echo "-----------------------------------------"
CUDA_VISIBLE_DEVICES=$GPU python train.py \
  --model $MODEL --batch-size $BATCH_SIZE --dropout $DROPOUT \
  --lr $LR --weight-decay $WD --patience $PATIENCE \
  --image-size $IMG_SIZE --seed $SEED \
  --use-aux-features --aux-skewness --aux-intensity --aux-clarity \
  --output-dir ./outputs/ablation/05_image_stats_only
echo ""

# 实验6：人口学特征 + 图像统计特征
echo "实验6: 人口学特征 + 图像统计特征"
echo "-----------------------------------------"
CUDA_VISIBLE_DEVICES=$GPU python train.py \
  --model $MODEL --batch-size $BATCH_SIZE --dropout $DROPOUT \
  --lr $LR --weight-decay $WD --patience $PATIENCE \
  --image-size $IMG_SIZE --seed $SEED \
  --use-aux-features --aux-gender --aux-bmi --aux-skewness --aux-intensity --aux-clarity \
  --output-dir ./outputs/ablation/06_all_features
echo ""

# 实验7：仅偏度
echo "实验7: 仅偏度特征"
echo "-----------------------------------------"
CUDA_VISIBLE_DEVICES=$GPU python train.py \
  --model $MODEL --batch-size $BATCH_SIZE --dropout $DROPOUT \
  --lr $LR --weight-decay $WD --patience $PATIENCE \
  --image-size $IMG_SIZE --seed $SEED \
  --use-aux-features --aux-skewness \
  --output-dir ./outputs/ablation/07_skewness_only
echo ""

# 实验8：仅清晰度
echo "实验8: 仅清晰度特征"
echo "-----------------------------------------"
CUDA_VISIBLE_DEVICES=$GPU python train.py \
  --model $MODEL --batch-size $BATCH_SIZE --dropout $DROPOUT \
  --lr $LR --weight-decay $WD --patience $PATIENCE \
  --image-size $IMG_SIZE --seed $SEED \
  --use-aux-features --aux-clarity \
  --output-dir ./outputs/ablation/08_clarity_only
echo ""

# 实验9：仅平均灰度
echo "实验9: 仅平均灰度特征"
echo "-----------------------------------------"
CUDA_VISIBLE_DEVICES=$GPU python train.py \
  --model $MODEL --batch-size $BATCH_SIZE --dropout $DROPOUT \
  --lr $LR --weight-decay $WD --patience $PATIENCE \
  --image-size $IMG_SIZE --seed $SEED \
  --use-aux-features --aux-intensity \
  --output-dir ./outputs/ablation/09_intensity_only
echo ""

# 实验10：所有特征 + 更大隐藏层
echo "实验10: 所有特征 + aux_hidden_dim=64"
echo "-----------------------------------------"
CUDA_VISIBLE_DEVICES=$GPU python train.py \
  --model $MODEL --batch-size $BATCH_SIZE --dropout $DROPOUT \
  --lr $LR --weight-decay $WD --patience $PATIENCE \
  --image-size $IMG_SIZE --seed $SEED \
  --use-aux-features --aux-gender --aux-bmi --aux-skewness --aux-intensity --aux-clarity \
  --aux-hidden-dim 64 \
  --output-dir ./outputs/ablation/10_all_hidden64
echo ""

echo "========================================="
echo "所有实验完成！"
echo "========================================="
echo ""
echo "结果汇总脚本："
echo "python tools/summarize_ablation_results.py"
