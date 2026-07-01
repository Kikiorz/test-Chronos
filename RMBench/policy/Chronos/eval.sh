#!/bin/bash

# Usage: bash policy/Chronos/eval.sh <task_name> <task_config> <ckpt_name> <seed> <gpu_id>

policy_name="Chronos"
task_name=${1:-"cover_blocks_three"}      # Default task
task_config=${2} # Default config
ckpt_setting=${3}
seed=${4:-42}
gpu_id=${5:-0}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33m[Chronos] Deploying Policy on GPU: ${gpu_id}\033[0m"

# Move to RoboTwin Root
cd ../.. 

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_dir policy/Chronos/checkpoints/${task_name}/EE_16 \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --gpu_id ${gpu_id}