#!/usr/bin/env bash
set -euo pipefail

# Vast defaults can be overridden without editing the repository.
repo_root="${CHRONOS_REPO_ROOT:-/workspace/Chronos}"
venv_root="${CHRONOS_VENV_ROOT:-/workspace/.venvs/chronos-rgb}"
data_root="${CHRONOS_DATA_ROOT:-/workspace/datasets/rmbench_rgb/cover_blocks/demo_clean/data}"
run_root="${CHRONOS_RUN_ROOT:-/workspace/chronos_rgb_runs/cover_blocks}"

source "${venv_root}/bin/activate"
mkdir -p "${run_root}/EE_16"
cd "${repo_root}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

exec python -u RMBench/policy/Chronos_RGB/train_par_2D_IMLE_EE.py \
  --data-root "${data_root}" \
  --task-name cover_blocks \
  --output-dir "${run_root}/EE_16" \
  --scaler-path "${run_root}/scaler_cover_blocks_ee_rgb.pth" \
  --expected-episodes 50 \
  --batch-size 1 \
  --num-workers 4 \
  --vision-chunk-size 128 \
  --supervision-frames 0 \
  --precision 32-true \
  --epochs 600 \
  --resume auto \
  "$@"
