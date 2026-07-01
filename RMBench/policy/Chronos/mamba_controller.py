import torch
import numpy as np
import sys
from . import mamba_policy_par_3D_IMLE
from .mamba_policy_par_3D_IMLE import MambaPolicy, MambaConfig
from .scaler_M import Scaler

class MambaController:
    def __init__(self, args_override):
        self.device = torch.device(args_override.get("device", "cuda:0"))
        
        self.config = MambaConfig()
        self.config.embed_dim = 1024
        self.config.d_model = 1024
        self.config.action_dim = 16   # EE 16维
        self.config.lowdim_dim = 16   # EE 16维
        self.config.num_blocks = 6
        self.future_steps = 16
        
        # 16 维 EE Keys
        obs_keys = [
            'ee_l_x', 'ee_l_y', 'ee_l_z', 'ee_l_qw', 'ee_l_qx', 'ee_l_qy', 'ee_l_qz', 'gripper_l',
            'ee_r_x', 'ee_r_y', 'ee_r_z', 'ee_r_qw', 'ee_r_qx', 'ee_r_qy', 'ee_r_qz', 'gripper_r'
        ]
        act_keys = [k + '_act' for k in obs_keys]
        lowdim_keys = obs_keys + act_keys
        
        lowdim_shapes = {k: 1 if 'act' not in k else (self.future_steps, 1) for k in lowdim_keys}
        
        self.scaler = Scaler(lowdim_dict=lowdim_shapes)
        scaler_path = args_override['scaler_path']
        print(f"[MambaController] Loading 3D EE scaler from {scaler_path}")
        self.scaler.load(scaler_path)
        self.scaler.to(self.device)
        
        print("[MambaController] Initializing 3D EE MambaPolicy...")
        self.policy = MambaPolicy(
            camera_names=[], 
            embed_dim=self.config.embed_dim,
            lowdim_dim=16,            
            d_model=self.config.d_model,
            action_dim=16,
            num_blocks=self.config.num_blocks,
            mamba_cfg=self.config,
            future_steps=self.future_steps
        )
        self.policy.to(self.device)
        self.policy.eval()

        sys.modules['mamba_policy_par_3D_IMLE'] = mamba_policy_par_3D_IMLE
        ckpt_path = args_override['ckpt_path']
        print(f"[MambaController] Loading weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        new_state_dict = {k.replace('policy.', ''): v for k, v in state_dict.items()}
                
        self.policy.load_state_dict(new_state_dict, strict=False)
        print(f"[MambaController] Weights loaded.")
        
        self.hiddens = None
        self.temporal_agg = args_override.get("temporal_agg", True)
        self.num_queries = self.future_steps 
        self.max_timesteps = 5000 
        self.t = 0
        if self.temporal_agg:
            self.all_time_actions = torch.zeros([
                self.max_timesteps,
                self.max_timesteps + self.num_queries,
                self.config.action_dim
            ]).to(self.device)

    def reset(self):
        self.hiddens = self.policy.init_hidden_states(batch_size=1, device=self.device)
        self.t = 0
        if self.temporal_agg:
            self.all_time_actions.zero_()

    def denormalize_action(self, actions_norm):
        arm1_dict = {
            'ee_l_x_act': actions_norm[..., 0:1], 'ee_l_y_act': actions_norm[..., 1:2], 'ee_l_z_act': actions_norm[..., 2:3],
            'ee_l_qw_act': actions_norm[..., 3:4], 'ee_l_qx_act': actions_norm[..., 4:5], 'ee_l_qy_act': actions_norm[..., 5:6], 'ee_l_qz_act': actions_norm[..., 6:7],
            'gripper_l_act': actions_norm[..., 7:8]
        }
        arm2_dict = {
            'ee_r_x_act': actions_norm[..., 8:9], 'ee_r_y_act': actions_norm[..., 9:10], 'ee_r_z_act': actions_norm[..., 10:11],
            'ee_r_qw_act': actions_norm[..., 11:12], 'ee_r_qx_act': actions_norm[..., 12:13], 'ee_r_qy_act': actions_norm[..., 13:14], 'ee_r_qz_act': actions_norm[..., 14:15],
            'gripper_r_act': actions_norm[..., 15:16]
        }
        arm1_denorm = self.scaler.denormalize(arm1_dict)
        arm2_denorm = self.scaler.denormalize(arm2_dict)
        out = torch.cat([
            arm1_denorm['ee_l_x_act'], arm1_denorm['ee_l_y_act'], arm1_denorm['ee_l_z_act'], 
            arm1_denorm['ee_l_qw_act'], arm1_denorm['ee_l_qx_act'], arm1_denorm['ee_l_qy_act'], arm1_denorm['ee_l_qz_act'], arm1_denorm['gripper_l_act'],
            arm2_denorm['ee_r_x_act'], arm2_denorm['ee_r_y_act'], arm2_denorm['ee_r_z_act'], 
            arm2_denorm['ee_r_qw_act'], arm2_denorm['ee_r_qx_act'], arm2_denorm['ee_r_qy_act'], arm2_denorm['ee_r_qz_act'], arm2_denorm['gripper_r_act']
        ], dim=2)
        return out

    @torch.inference_mode()
    def get_action(self, obs_dict):
        qpos_raw = obs_dict['qpos'].to(self.device) # [1, 16]
        pc_tensor = obs_dict['point_cloud'].to(self.device) 
        if pc_tensor.dim() == 2:
            pc_tensor = pc_tensor.unsqueeze(0)
        
        lowdim_input = {
            'ee_l_x': qpos_raw[:, 0:1], 'ee_l_y': qpos_raw[:, 1:2], 'ee_l_z': qpos_raw[:, 2:3],
            'ee_l_qw': qpos_raw[:, 3:4], 'ee_l_qx': qpos_raw[:, 4:5], 'ee_l_qy': qpos_raw[:, 5:6], 'ee_l_qz': qpos_raw[:, 6:7],
            'gripper_l': qpos_raw[:, 7:8],
            'ee_r_x': qpos_raw[:, 8:9], 'ee_r_y': qpos_raw[:, 9:10], 'ee_r_z': qpos_raw[:, 10:11],
            'ee_r_qw': qpos_raw[:, 11:12], 'ee_r_qx': qpos_raw[:, 12:13], 'ee_r_qy': qpos_raw[:, 13:14], 'ee_r_qz': qpos_raw[:, 14:15],
            'gripper_r': qpos_raw[:, 15:16]
        }
        lowdim_norm_dict = self.scaler.normalize(lowdim_input)
        
        obs_keys = [
            'ee_l_x', 'ee_l_y', 'ee_l_z', 'ee_l_qw', 'ee_l_qx', 'ee_l_qy', 'ee_l_qz', 'gripper_l',
            'ee_r_x', 'ee_r_y', 'ee_r_z', 'ee_r_qw', 'ee_r_qx', 'ee_r_qy', 'ee_r_qz', 'gripper_r'
        ]
        lowdim_norm = torch.cat([lowdim_norm_dict[k] for k in obs_keys], dim=1).to(self.device).float() # [1, 16]

        # 融合网络仅返回 1024 维绝对锚点 x_fused_step
        x_fused_step = self.policy.fusion_engine(pc_tensor, lowdim_norm)

        pred_action_norm, self.hiddens = self.policy.step(
            x_fused_step, 
            self.hiddens, 
            sample_steps=5
        )

        if self.temporal_agg:
            self.all_time_actions[[self.t], self.t : self.t + self.num_queries] = pred_action_norm
            actions_for_curr_step = self.all_time_actions[:, self.t]
            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            
            k = 0.01
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
            
            raw_action_norm = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True) 
        else:
            raw_action_norm = pred_action_norm[:, 0, :] 
            
        dummy_seq = torch.zeros(1, self.future_steps, 16).to(self.device)
        dummy_seq[:, 0, :] = raw_action_norm
        denorm_seq = self.denormalize_action(dummy_seq)
        
        final_action = denorm_seq[0, 0, :].cpu().numpy() # [16]
        
        self.t += 1
        return final_action