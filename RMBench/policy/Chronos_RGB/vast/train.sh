#!/usr/bin/env bash
set -euo pipefail

# Vast paths can be overridden without changing the reviewed training contract.
repo_root="${CHRONOS_REPO_ROOT:-/workspace/test-Chronos}"
venv_root="${CHRONOS_VENV_ROOT:-/workspace/.venvs/chronos-rgb}"
data_root="${CHRONOS_DATA_ROOT:-/workspace/datasets/rmbench_rgb/cover_blocks/demo_clean/data}"
run_root="${CHRONOS_RUN_ROOT:-/workspace/chronos_rgb_runs/cover_blocks}"

source "${venv_root}/bin/activate"
mkdir -p "${run_root}/EE_16_official_rgb"
cd "${repo_root}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

exec python -u RMBench/policy/Chronos_RGB/train_par_2D_IMLE_EE.py \
  --data-root "${data_root}" \
  --task-name cover_blocks \
  --output-dir "${run_root}/EE_16_official_rgb" \
  --scaler-path "${run_root}/scaler_cover_blocks_ee_official_rgb.pth" \
  --expected-episodes 50 \
  --action-target-offset 0 \
  --image-height 480 \
  --image-width 640 \
  --batch-size 2 \
  --accumulate-grad-batches 3 \
  --num-workers 0 \
  --vision-chunk-size 256 \
  --learning-rate 1.7e-4 \
  --weight-decay 1e-4 \
  --eta-min 2e-5 \
  --precision 32-true \
  --epochs 600 \
  --warmup-epochs 15 \
  --periodic-every 100 \
  --resume auto \
  "$@"
