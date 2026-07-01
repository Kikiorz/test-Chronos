#deploy_policy.py

import sys
import numpy as np
import torch
import cv2
import os
from .mamba_controller import MambaController
import numpy as np
import torch
import cv2
import os


def encode_obs(observation):
    processed_obs = {}
    raw_pc = observation['pointcloud'] 
    
    # 强制将 list 转换为 float32 的 numpy 数组
    if isinstance(raw_pc, list):
        raw_pc = np.array(raw_pc, dtype=np.float32)
        
    if raw_pc.shape[-1] == 6 and raw_pc[..., 3:].max() > 2.0:
        raw_pc[..., 3:] = raw_pc[..., 3:] / 255.0
        
    processed_obs['point_cloud'] = torch.from_numpy(raw_pc).float().unsqueeze(0) # [1, N, 6]
    
    # =========================================================
    # 【鲁棒性提取 16 维 EE 本体感】
    if "endpose" in observation:
        # 如果实机字典也和 HDF5 一样包含顶层的 'endpose'
        left_ee = observation["endpose"]["left_endpose"]
        left_gripper = observation["endpose"]["left_gripper"]
        right_ee = observation["endpose"]["right_endpose"]
        right_gripper = observation["endpose"]["right_gripper"]
    else:
        # 兼容老版 RoboTwin 字典结构
        left_ee = observation.get("left_endpose", observation.get("observation", {}).get("left_endpose", []))
        left_gripper = observation["joint_action"]["left_gripper"]
        right_ee = observation.get("right_endpose", observation.get("observation", {}).get("right_endpose", []))
        right_gripper = observation["joint_action"]["right_gripper"]
    
    # 将标量转换为 float，并将列表拼接
    # EE位姿是7维List，夹爪是标量，必须保证总共拼接成 16 个元素
    ee_list = list(left_ee) + [float(left_gripper)] + list(right_ee) + [float(right_gripper)]
    
    processed_obs['qpos'] = torch.tensor(ee_list).float().unsqueeze(0) # [1, 16]
    
    return processed_obs

def get_model(usr_args):
    controller_args = {
        "device": f"cuda:{usr_args.get('gpu_id', 0)}",
        "ckpt_path": usr_args['ckpt_path'],   
        "scaler_path": usr_args['scaler_path'], 
        "temporal_agg": usr_args.get('temporal_agg', True)
    }
    return MambaController(controller_args)

def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)
    action = model.get_action(obs)
    # [注意]：必须告诉引擎，我们现在传递的是 'ee' 而不是 'qpos'！
    TASK_ENV.take_action(action, action_type='ee') 
    return

def reset_model(model):
    model.reset()