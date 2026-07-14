#!/usr/bin/env bash
set -euo pipefail

# Vast defaults can be overridden without editing the repository.
repo_root="${CHRONOS_REPO_ROOT:-/workspace/Chronos}"
venv_root="${CHRONOS_VENV_ROOT:-/workspace/.venvs/chronos-rgb}"
data_root="${CHRONOS_DATA_ROOT:-/workspace/datasets/rmbench_rgb/cover_blocks/demo_clean/data}"
run_root="${CHRONOS_RUN_ROOT:-/workspace/chronos_rgb_v2_runs/cover_blocks}"
v1_warm_start="${CHRONOS_V1_WARM_START:-}"

source "${venv_root}/bin/activate"
mkdir -p "${run_root}/EE_16_v2"
cd "${repo_root}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

resume_args=(--resume auto)
if [[ ! -f "${run_root}/EE_16_v2/last.ckpt" && -n "${v1_warm_start}" ]]; then
  resume_args=(--warm-start "${v1_warm_start}" --resume none --refit-scaler)
fi

exec python -u RMBench/policy/Chronos_RGB/train_par_2D_IMLE_EE.py \
  --data-root "${data_root}" \
  --task-name cover_blocks \
  --output-dir "${run_root}/EE_16_v2" \
  --scaler-path "${run_root}/scaler_cover_blocks_ee_rgb_v2.pth" \
  --expected-episodes 50 \
  --batch-size 1 \
  --num-workers 4 \
  --vision-chunk-size 64 \
  --supervision-frames 0 \
  --backbone-trainable layer4 \
  --visual-head-lr 1e-4 \
  --backbone-layer4-lr 1e-5 \
  --learning-rate 3e-5 \
  --precision 32-true \
  --epochs 600 \
  --warmup-epochs 15 \
  --periodic-every 50 \
  "${resume_args[@]}" \
  "$@"
