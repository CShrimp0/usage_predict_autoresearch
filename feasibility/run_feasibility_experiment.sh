#!/bin/bash
set -e

MODEL="resnet50"
BATCH_SIZE=32
DROPOUT=0.6
LR=1e-4
WEIGHT_DECAY=1e-4
EPOCHS=200
PATIENCE=50
IMAGE_SIZE=224
SEED=42
YOUNG_MAX=35
MIDDLE_MAX=60

python feasibility/train_age_group.py \
  --model ${MODEL} \
  --batch-size ${BATCH_SIZE} \
  --dropout ${DROPOUT} \
  --lr ${LR} \
  --weight-decay ${WEIGHT_DECAY} \
  --epochs ${EPOCHS} \
  --patience ${PATIENCE} \
  --image-size ${IMAGE_SIZE} \
  --seed ${SEED} \
  --young-max ${YOUNG_MAX} \
  --middle-max ${MIDDLE_MAX}

LATEST_RUN=$(ls -dt outputs/feasibility/run_* | head -n 1)

python feasibility/evaluate_age_group.py \
  --checkpoint "${LATEST_RUN}/best_model.pth"
