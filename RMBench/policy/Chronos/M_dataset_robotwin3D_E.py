import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation as scipy_R
import h5py
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
try:
    from .scaler_M import Scaler
except ImportError:
    from scaler_M import Scaler
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class PointCloudTrajectoryDataset(Dataset):
    """
    [EE 末端控制版 Dataset]
    16维: 左臂(XYZ+QwQxQyQz=7) + 左爪(1) + 右臂(7) + 右爪(1)
    """
    def __init__(self, root_dir: str, mode: str = "train", 
                 future_steps: int = 16, scaler: Scaler = None):
        self.root_dir = root_dir
        self.mode = mode
        self.future_steps = future_steps
        self.scaler = scaler
        
        self.file_paths = []
        self._find_files()
        print(f"[{mode}] Found {len(self.file_paths)} trajectories for 3D EE processing.")

    def _find_files(self):
        target_dir = os.path.join(self.root_dir)
        if not os.path.exists(target_dir):
            return
        for file in sorted(os.listdir(target_dir)):
            if file.endswith('.hdf5'):
                self.file_paths.append(os.path.join(target_dir, file))

    def __len__(self):
        return len(self.file_paths)
    
    def fit_scaler(self, num_workers=4):
        print("Gathering data to fit scaler...")
        loader = DataLoader(self, batch_size=1, num_workers=num_workers, collate_fn=lambda x: x)
        
        all_obs = []
        all_actions = []
        
        for batch in tqdm(loader, desc="Reading Trajectories"):
            item = batch[0]
            all_obs.append(item['obs'])          # [L, 16]
            all_actions.append(item['actions'])  # [L, future_steps, 16]
            
        all_obs = torch.cat(all_obs, dim=0)
        all_actions = torch.cat(all_actions, dim=0)
        
        # [修改]：适配 16 维 EE 字典
        data_dict = {}
        # Obs: 左臂 (0:8)
        data_dict['ee_l_x'] = all_obs[:, 0:1]
        data_dict['ee_l_y'] = all_obs[:, 1:2]
        data_dict['ee_l_z'] = all_obs[:, 2:3]
        data_dict['ee_l_qw'] = all_obs[:, 3:4]
        data_dict['ee_l_qx'] = all_obs[:, 4:5]
        data_dict['ee_l_qy'] = all_obs[:, 5:6]
        data_dict['ee_l_qz'] = all_obs[:, 6:7]
        data_dict['gripper_l'] = all_obs[:, 7:8]
        # Obs: 右臂 (8:16)
        data_dict['ee_r_x'] = all_obs[:, 8:9]
        data_dict['ee_r_y'] = all_obs[:, 9:10]
        data_dict['ee_r_z'] = all_obs[:, 10:11]
        data_dict['ee_r_qw'] = all_obs[:, 11:12]
        data_dict['ee_r_qx'] = all_obs[:, 12:13]
        data_dict['ee_r_qy'] = all_obs[:, 13:14]
        data_dict['ee_r_qz'] = all_obs[:, 14:15]
        data_dict['gripper_r'] = all_obs[:, 15:16]

        # Act: 左臂 (0:8)
        data_dict['ee_l_x_act'] = all_actions[:, :, 0:1]
        data_dict['ee_l_y_act'] = all_actions[:, :, 1:2]
        data_dict['ee_l_z_act'] = all_actions[:, :, 2:3]
        data_dict['ee_l_qw_act'] = all_actions[:, :, 3:4]
        data_dict['ee_l_qx_act'] = all_actions[:, :, 4:5]
        data_dict['ee_l_qy_act'] = all_actions[:, :, 5:6]
        data_dict['ee_l_qz_act'] = all_actions[:, :, 6:7]
        data_dict['gripper_l_act'] = all_actions[:, :, 7:8]
        # Act: 右臂 (8:16)
        data_dict['ee_r_x_act'] = all_actions[:, :, 8:9]
        data_dict['ee_r_y_act'] = all_actions[:, :, 9:10]
        data_dict['ee_r_z_act'] = all_actions[:, :, 10:11]
        data_dict['ee_r_qw_act'] = all_actions[:, :, 11:12]
        data_dict['ee_r_qx_act'] = all_actions[:, :, 12:13]
        data_dict['ee_r_qy_act'] = all_actions[:, :, 13:14]
        data_dict['ee_r_qz_act'] = all_actions[:, :, 14:15]
        data_dict['gripper_r_act'] = all_actions[:, :, 15:16]
        
        self.scaler.fit(data_dict)

    def save_scaler(self, filepath):
        self.scaler.save(filepath)

    def load_scaler(self, filepath):
        self.scaler.load(filepath)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        with h5py.File(file_path, 'r') as root:
            # 1. 提取 EE 本体感 -> [L, 16]
            # 根据探查结果，绝对路径定死在顶级目录 endpose/ 下
            left_ee = root["endpose/left_endpose"][()]
            left_gripper = root["endpose/left_gripper"][()]
            right_ee = root["endpose/right_endpose"][()]
            right_gripper = root["endpose/right_gripper"][()]
            
            left_ee = left_ee.reshape(left_ee.shape[0], -1)
            left_gripper = left_gripper.reshape(left_gripper.shape[0], -1)
            right_ee = right_ee.reshape(right_ee.shape[0], -1)
            right_gripper = right_gripper.reshape(right_gripper.shape[0], -1)

            # 组装为 16 维 [L, 16]
            obs = np.concatenate([left_ee, left_gripper, right_ee, right_gripper], axis=-1).astype(np.float32)

            # 动作直接使用偏移的观测或同态复制
            action_data = np.copy(obs).astype(np.float32)
            
            # 3. 提取 3D 点云
            pc_key = 'pointcloud' if 'pointcloud' in root else 'observation/pointcloud'
            raw_pc = root[pc_key][()]
            if raw_pc.shape[-1] == 6 and raw_pc[..., 3:].max() > 2.0:
                raw_pc[..., 3:] = raw_pc[..., 3:] / 255.0
            pc_data = raw_pc.astype(np.float32)

        action_data = np.copy(obs).astype(np.float32)
            
        L = obs.shape[0]
        action_dim = action_data.shape[-1] # 16
        action_target = np.zeros((L, self.future_steps, action_dim), dtype=np.float32)
        
        for t in range(L):
            end_idx = min(t + self.future_steps, L)
            valid_len = end_idx - t
            action_target[t, :valid_len] = action_data[t:end_idx]
            if valid_len < self.future_steps:
                action_target[t, valid_len:] = action_data[-1] 

        obs_tensor = torch.from_numpy(obs).float()
        action_tensor = torch.from_numpy(action_target).float()
        pc_tensor = torch.from_numpy(pc_data).float()
        mask_tensor = torch.zeros(L, dtype=torch.bool)

        # 6. Scaler 归一化 (16维)
        if self.scaler is not None and getattr(self.scaler, 'mean_dict', None) is not None:
            obs_dict = {
                'ee_l_x': obs_tensor[:, 0:1], 'ee_l_y': obs_tensor[:, 1:2], 'ee_l_z': obs_tensor[:, 2:3],
                'ee_l_qw': obs_tensor[:, 3:4], 'ee_l_qx': obs_tensor[:, 4:5], 'ee_l_qy': obs_tensor[:, 5:6], 'ee_l_qz': obs_tensor[:, 6:7],
                'gripper_l': obs_tensor[:, 7:8],
                'ee_r_x': obs_tensor[:, 8:9], 'ee_r_y': obs_tensor[:, 9:10], 'ee_r_z': obs_tensor[:, 10:11],
                'ee_r_qw': obs_tensor[:, 11:12], 'ee_r_qx': obs_tensor[:, 12:13], 'ee_r_qy': obs_tensor[:, 13:14], 'ee_r_qz': obs_tensor[:, 14:15],
                'gripper_r': obs_tensor[:, 15:16]
            }
            act_dict = {
                'ee_l_x_act': action_tensor[:, :, 0:1], 'ee_l_y_act': action_tensor[:, :, 1:2], 'ee_l_z_act': action_tensor[:, :, 2:3],
                'ee_l_qw_act': action_tensor[:, :, 3:4], 'ee_l_qx_act': action_tensor[:, :, 4:5], 'ee_l_qy_act': action_tensor[:, :, 5:6], 'ee_l_qz_act': action_tensor[:, :, 6:7],
                'gripper_l_act': action_tensor[:, :, 7:8],
                'ee_r_x_act': action_tensor[:, :, 8:9], 'ee_r_y_act': action_tensor[:, :, 9:10], 'ee_r_z_act': action_tensor[:, :, 10:11],
                'ee_r_qw_act': action_tensor[:, :, 11:12], 'ee_r_qx_act': action_tensor[:, :, 12:13], 'ee_r_qy_act': action_tensor[:, :, 13:14], 'ee_r_qz_act': action_tensor[:, :, 14:15],
                'gripper_r_act': action_tensor[:, :, 15:16]
            }
            
            norm_obs = self.scaler.normalize(obs_dict)
            norm_act = self.scaler.normalize(act_dict)
            
            obs_tensor = torch.cat([
                norm_obs['ee_l_x'], norm_obs['ee_l_y'], norm_obs['ee_l_z'], norm_obs['ee_l_qw'], norm_obs['ee_l_qx'], norm_obs['ee_l_qy'], norm_obs['ee_l_qz'], norm_obs['gripper_l'],
                norm_obs['ee_r_x'], norm_obs['ee_r_y'], norm_obs['ee_r_z'], norm_obs['ee_r_qw'], norm_obs['ee_r_qx'], norm_obs['ee_r_qy'], norm_obs['ee_r_qz'], norm_obs['gripper_r']
            ], dim=-1)
            
            action_tensor = torch.cat([
                norm_act['ee_l_x_act'], norm_act['ee_l_y_act'], norm_act['ee_l_z_act'], norm_act['ee_l_qw_act'], norm_act['ee_l_qx_act'], norm_act['ee_l_qy_act'], norm_act['ee_l_qz_act'], norm_act['gripper_l_act'],
                norm_act['ee_r_x_act'], norm_act['ee_r_y_act'], norm_act['ee_r_z_act'], norm_act['ee_r_qw_act'], norm_act['ee_r_qx_act'], norm_act['ee_r_qy_act'], norm_act['ee_r_qz_act'], norm_act['gripper_r_act']
            ], dim=-1)

        return {
            'obs': obs_tensor,
            'actions': action_tensor,
            'point_cloud': pc_tensor,
            'mask': mask_tensor
        }

def parallel_collate_fn_3d(batch):
    lengths = [item['obs'].shape[0] for item in batch]
    max_len = max(lengths)
    
    padded_obs, padded_actions, padded_pc, masks = [], [], [], []
    
    for item in batch:
        L = item['obs'].shape[0]
        obs_dim = item['obs'].shape[-1]
        act_dim1, act_dim2 = item['actions'].shape[1:]
        N, pc_dim = item['point_cloud'].shape[1:]
        
        obs_pad = torch.zeros((max_len, obs_dim), dtype=torch.float32)
        obs_pad[:L] = item['obs']
        padded_obs.append(obs_pad)
        
        act_pad = torch.zeros((max_len, act_dim1, act_dim2), dtype=torch.float32)
        act_pad[:L] = item['actions']
        padded_actions.append(act_pad)
        
        pc_pad = torch.zeros((max_len, N, pc_dim), dtype=torch.float32)
        pc_pad[:L] = item['point_cloud']
        padded_pc.append(pc_pad)
        
        mask = torch.ones(max_len, dtype=torch.bool)
        mask[:L] = False
        masks.append(mask)
        
    return {
        'obs': torch.stack(padded_obs),
        'actions': torch.stack(padded_actions),
        'point_cloud': torch.stack(padded_pc),
        'mask': torch.stack(masks)
    }
if __name__ == "__main__":
    # =================== SSD 自动化配置中心 ===================
    TASK_NAME = "cover_blocks"
    SSD_ROOT = "/home/sutai/data"  # 指向您的高速 SSD 挂载点
    # ======================================================

    # 1. 自动解析数据路径
    DATA_ROOT = f"{SSD_ROOT}/{TASK_NAME}/demo_clean/data/train"
    FUTURE_STEPS = 16
    
    print(f"��️  Preparing Scaler for Task: [{TASK_NAME}]")
    print(f"��  Reading from SSD: {DATA_ROOT}")

    # 2. 构造空壳 Scaler (EE 16维)
    OBS_KEYS = [
        'ee_l_x', 'ee_l_y', 'ee_l_z', 'ee_l_qw', 'ee_l_qx', 'ee_l_qy', 'ee_l_qz', 'gripper_l',
        'ee_r_x', 'ee_r_y', 'ee_r_z', 'ee_r_qw', 'ee_r_qx', 'ee_r_qy', 'ee_r_qz', 'gripper_r'
    ]
    ACT_KEYS = [k + '_act' for k in OBS_KEYS]
    
    full_lowdim_dict = {}
    for k in OBS_KEYS: 
        full_lowdim_dict[k] = 1         
    for k in ACT_KEYS: 
        full_lowdim_dict[k] = (FUTURE_STEPS, 1) 
    
    empty_scaler = Scaler(lowdim_dict=full_lowdim_dict)
    
    # 3. 实例化 Dataset
    print("Initializing 3D EE Dataset for Scaler generation...")
    ds = PointCloudTrajectoryDataset(
        root_dir=DATA_ROOT, 
        mode="train", 
        future_steps=FUTURE_STEPS, 
        scaler=empty_scaler
    )
    
    # 4. 拟合与保存
    print(f"Start fitting 3D EE scaler for {FUTURE_STEPS} steps...")
    ds.fit_scaler(num_workers=8) 
    
    save_filename = f'scaler_{TASK_NAME}_ee_3d.pth'
    ds.save_scaler(save_filename)
    print(f"✅ 3D EE Scaler saved to: {save_filename}")
    
    # 5. 验证
    print("\nVerifying Data Shapes...")
    item = ds[0]
    print(f"Observation Shape:    {item['obs'].shape}")          # [L, 16]
    print(f"Action Target Shape:  {item['actions'].shape}")      # [L, 16, 16]
    print(f"Point Cloud Shape:   {item['point_cloud'].shape}")   # [L, N, 6]