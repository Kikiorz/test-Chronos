#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash policy/Chronos_RGB_Joint/eval.sh [task] [config] [setting] [seed] [gpu] [test_num] [ckpt] [scaler]

policy_name="Chronos_RGB_Joint"
task_name="${1:-cover_blocks}"
task_config="${2:-demo_clean_rgb_joint}"
ckpt_setting="${3:-rgb_head_joint14_dinov3b}"
seed="${4:-42}"
physical_gpu_id="${5:-0}"
test_num="${6:-5}"
ckpt_path="${7:-policy/Chronos_RGB_Joint/checkpoints/${task_name}/Joint_14/last.ckpt}"
scaler_path="${8:-policy/Chronos_RGB_Joint/scaler_${task_name}_joint_rgb.pth}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rmbench_root="$(cd -- "${script_dir}/../.." && pwd)"
cd "${rmbench_root}"

export CUDA_VISIBLE_DEVICES="${physical_gpu_id}"
echo "[Chronos_RGB_Joint] task=${task_name}, config=${task_config}, seed=${seed}, physical GPU=${physical_gpu_id}"
echo "[Chronos_RGB_Joint] checkpoint=${ckpt_path}"

# CUDA_VISIBLE_DEVICES exposes the requested physical GPU as logical cuda:0.
PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/Chronos_RGB_Joint/deploy_policy.yml \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${ckpt_setting}" \
    --ckpt_path "${ckpt_path}" \
    --scaler_path "${scaler_path}" \
    --seed "${seed}" \
    --policy_name "${policy_name}" \
    --gpu_id 0 \
    --test_num "${test_num}"
