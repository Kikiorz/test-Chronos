#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${CHRONOS_REPO_ROOT:-$(git -C "${script_dir}" rev-parse --show-toplevel)}"
conda_env="${CHRONOS_CONDA_ENV:-RoboTwin}"
data_root="${CHRONOS_DATA_ROOT:-/home/zeno-rp/2026test/rmbench_rgb_dataset/data/cover_blocks/demo_clean/data}"
weights_path="${CHRONOS_RESNET18_WEIGHTS:-${HOME}/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth}"
run_root="${CHRONOS_RUN_ROOT:-/home/zeno-rp/2026test/chronos_rgb_joint_runs/cover_blocks}"

mkdir -p "${run_root}/Joint_14"
cd "${repo_root}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
# Batch-2 with three-step gradient accumulation leaves gradients resident.
# Expandable segments avoid allocator fragmentation between ResNet chunks.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec conda run --no-capture-output -n "${conda_env}" \
  python -u RMBench/policy/Chronos_RGB_Joint/train_par_2D_IMLE_Joint.py \
  --data-root "${data_root}" \
  --backbone-weights "${weights_path}" \
  --task-name cover_blocks \
  --output-dir "${run_root}/Joint_14" \
  --scaler-path "${run_root}/scaler_cover_blocks_joint_rgb.pth" \
  --expected-episodes 50 \
  --seed 42 \
  --split-seed 42 \
  --validation-seed 42 \
  --val-fraction 0.1 \
  --image-height 240 \
  --image-width 320 \
  --batch-size 2 \
  --num-workers 0 \
  --vision-chunk-size 128 \
  --supervision-frames 0 \
  --precision 32-true \
  --epochs 600 \
  --accumulate-grad-batches 3 \
  --learning-rate 1.7e-4 \
  --weight-decay 1e-4 \
  --warmup-epochs 15 \
  --eta-min 2e-5 \
  --gradient-clip-val 1.0 \
  --save-top-k 2 \
  --checkpoint-every-n-epochs 5 \
  --periodic-every-n-epochs 0 \
  --overfit-batches 0 \
  --resume none \
  "$@"
