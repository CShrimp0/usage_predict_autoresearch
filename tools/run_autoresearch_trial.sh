#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/szdx/anaconda3/envs/us/bin/python}"
TIME_BUDGET_MINUTES="${TIME_BUDGET_MINUTES:-5}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${ROOT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" train.py \
  --model resnet50 \
  --batch-size 32 \
  --dropout 0.6 \
  --lr 0.0001 \
  --weight-decay 0.0001 \
  --loss mae \
  --epochs 500 \
  --patience 100 \
  --image-size 224 \
  --age-bin-width 10 \
  --seed 42 \
  --num-workers 8 \
  --use-aux-features \
  --aux-gender \
  --aux-bmi \
  --aux-skewness \
  --aux-intensity \
  --aux-clarity \
  --aux-hidden-dim 32 \
  --output-dir ./outputs/autoresearch \
  --time-budget-minutes "${TIME_BUDGET_MINUTES}" \
  "$@"
