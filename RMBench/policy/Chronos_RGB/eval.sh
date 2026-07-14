#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash policy/Chronos_RGB/eval.sh [task] [config] [setting] [seed] [gpu] [test_num] [ckpt] [scaler] [execution_offset] [visual_architecture]

policy_name="Chronos_RGB"
task_name="${1:-cover_blocks}"
task_config="${2:-demo_clean}"
ckpt_setting="${3:-rgb_v2_head_ee16}"
seed="${4:-42}"
physical_gpu_id="${5:-0}"
test_num="${6:-5}"
ckpt_path="${7:-policy/Chronos_RGB/checkpoints/${task_name}/EE_16_v2/last.ckpt}"
scaler_path="${8:-policy/Chronos_RGB/scaler_${task_name}_ee_rgb_v2.pth}"
execution_horizon_offset="${9:-0}"
visual_architecture="${10:-v2}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rmbench_root="$(cd -- "${script_dir}/../.." && pwd)"
cd "${rmbench_root}"

export CUDA_VISIBLE_DEVICES="${physical_gpu_id}"
echo "[Chronos_RGB] task=${task_name}, config=${task_config}, seed=${seed}, physical GPU=${physical_gpu_id}"
echo "[Chronos_RGB] checkpoint=${ckpt_path}"
echo "[Chronos_RGB] execution_horizon_offset=${execution_horizon_offset}"
echo "[Chronos_RGB] visual_architecture=${visual_architecture}"

# CUDA_VISIBLE_DEVICES exposes the requested physical GPU as logical cuda:0.
PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/Chronos_RGB/deploy_policy.yml \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${ckpt_setting}" \
    --ckpt_path "${ckpt_path}" \
    --scaler_path "${scaler_path}" \
    --execution_horizon_offset "${execution_horizon_offset}" \
    --visual_architecture "${visual_architecture}" \
    --seed "${seed}" \
    --policy_name "${policy_name}" \
    --gpu_id 0 \
    --test_num "${test_num}"
