import argparse
import os
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from .scaler_M import Scaler
except ImportError:
    from scaler_M import Scaler

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


# =============================================================================
# Pose utils: 6D EE pose [x, y, z, rx, ry, rz] -> pose9 [x, y, z, rot6]
# 注意：别人代码里叫 pose10d，但函数本身输出 3 + 6 = 9 维；
# 加上 gripper 后才是单臂 10 维。
# 这里采用与其脚本一致的 row-based rot6d：取旋转矩阵前两行。
# =============================================================================

def _normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def rodrigues_np(rotvec: np.ndarray) -> np.ndarray:
    """Vectorized Rodrigues. rotvec: [..., 3] -> rotmat: [..., 3, 3]."""
    rv = np.asarray(rotvec, dtype=np.float32)
    orig_shape = rv.shape[:-1]
    r = rv.reshape(-1, 3)

    theta = np.linalg.norm(r, axis=-1, keepdims=True)  # [N, 1]
    axis = r / np.maximum(theta, 1e-7)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]

    n = r.shape[0]
    K = np.zeros((n, 3, 3), dtype=np.float32)
    K[:, 0, 1] = -z
    K[:, 0, 2] = y
    K[:, 1, 0] = z
    K[:, 1, 2] = -x
    K[:, 2, 0] = -y
    K[:, 2, 1] = x

    I = np.eye(3, dtype=np.float32)[None, :, :]
    outer = axis[:, :, None] * axis[:, None, :]
    c = np.cos(theta).reshape(n, 1, 1)
    s = np.sin(theta).reshape(n, 1, 1)

    R = c * I + (1.0 - c) * outer + s * K
    small = (theta.reshape(-1) < 1e-7)
    if np.any(small):
        R[small] = I
    return R.reshape(orig_shape + (3, 3)).astype(np.float32)


def mat_to_rot6d_rows_np(mat: np.ndarray) -> np.ndarray:
    """rotmat [...,3,3] -> row-based rot6d [...,6], consistent with the reference code."""
    batch_shape = mat.shape[:-2]
    return mat[..., :2, :].copy().reshape(batch_shape + (6,)).astype(np.float32)


def pose6d_to_pose9d_np(pose6d: np.ndarray) -> np.ndarray:
    """[pos3, rotvec3] -> [pos3, rot6]."""
    pose6d = np.asarray(pose6d, dtype=np.float32)
    pos = pose6d[..., :3]
    rotvec = pose6d[..., 3:6]
    rotmat = rodrigues_np(rotvec)
    rot6 = mat_to_rot6d_rows_np(rotmat)
    return np.concatenate([pos, rot6], axis=-1).astype(np.float32)


class ImageTrajectoryDataset(Dataset):
    """
    2D image + dual-arm lowdim trajectory dataset.

    目录兼容：
    1) root_dir/rearrange_cube_data + mode="train"  -> 读取 root_dir/train/*
    2) root_dir/rearrange_cube_data/train + mode="train" -> 直接读取 root_dir/*

    输出：
      obs:     [L, D]
      actions: [L, future_steps, D]
      image:   [L, 3, H, W]
      mask:    [L], False 表示有效帧

    当 use_pose10d=True：
      单臂 = pose9(pos3+rot6) + gripper1 = 10D
      双臂 = 20D
    当 use_pose10d=False：
      单臂 = pose6(pos3+rotvec3) + gripper1 = 7D
      双臂 = 14D
    """

    @staticmethod
    def make_lowdim_dict(future_steps: int = 16, use_pose10d: bool = True) -> Dict[str, object]:
        pose_dim = 9 if use_pose10d else 6
        obs_keys = (
            [f"pose_l_{i+1}" for i in range(pose_dim)] + ["gripper_l"] +
            [f"pose_r_{i+1}" for i in range(pose_dim)] + ["gripper_r"]
        )
        act_keys = [k + "_act" for k in obs_keys]
        lowdim_dict: Dict[str, object] = {k: 1 for k in obs_keys}
        for k in act_keys:
            lowdim_dict[k] = (future_steps, 1)
        return lowdim_dict

    @staticmethod
    def make_keys(use_pose10d: bool = True) -> Tuple[List[str], List[str]]:
        pose_dim = 9 if use_pose10d else 6
        obs_keys = (
            [f"pose_l_{i+1}" for i in range(pose_dim)] + ["gripper_l"] +
            [f"pose_r_{i+1}" for i in range(pose_dim)] + ["gripper_r"]
        )
        act_keys = [k + "_act" for k in obs_keys]
        return obs_keys, act_keys

    def __init__(self,
                 root_dir: str,
                 mode: str = "train",
                 future_steps: int = 16,
                 scaler: Optional[Scaler] = None,
                 resize_hw: Tuple[int, int] = (640, 480),
                 use_pose10d: bool = True,
                 image_folder: str = "img"):
        super().__init__()
        self.root_dir = os.path.abspath(root_dir)
        self.mode = mode
        self.future_steps = future_steps
        self.scaler = scaler
        self.resize_hw = resize_hw  # cv2 format: (W, H)
        self.use_pose10d = use_pose10d
        self.image_folder = image_folder

        self.obs_keys, self.act_keys = self.make_keys(use_pose10d=use_pose10d)
        self.lowdim_dict = self.make_lowdim_dict(future_steps=future_steps, use_pose10d=use_pose10d)
        self.obs_dim = len(self.obs_keys)
        self.action_dim = self.obs_dim

        if self.scaler is None and mode == "train":
            self.scaler = Scaler(lowdim_dict=self.lowdim_dict)

        self.dataset_dir = self._resolve_dataset_dir(self.root_dir, mode)
        self.records = self._find_records(self.dataset_dir)
        print(
            f"[{mode}] dataset_dir={self.dataset_dir} | found {len(self.records)} trajectories | "
            f"use_pose10d={self.use_pose10d} | dim={self.obs_dim}"
        )

        if len(self.records) == 0:
            raise RuntimeError(
                f"No valid trajectory folders found in {self.dataset_dir}. "
                f"Expected subfolders containing pose.npy/pose2.npy/target_pose.npy/..."
            )

    @staticmethod
    def _has_record_dirs(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        for item in os.listdir(path):
            p = os.path.join(path, item)
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "pose.npy")):
                return True
        return False

    def _resolve_dataset_dir(self, root_dir: str, mode: str) -> str:
        """兼容 root_dir 指向数据根目录或已指向 train/test 目录。"""
        candidates = []
        split_candidate = os.path.join(root_dir, mode)
        candidates.append(split_candidate)
        if mode == "test":
            candidates.append(os.path.join(root_dir, "val"))
            candidates.append(os.path.join(root_dir, "valid"))
        candidates.append(root_dir)

        seen = set()
        for c in candidates:
            c = os.path.abspath(c)
            if c in seen:
                continue
            seen.add(c)
            if self._has_record_dirs(c):
                return c
        # 返回最合理路径，后续报错会显示该路径
        return os.path.abspath(split_candidate if os.path.isdir(split_candidate) else root_dir)

    @staticmethod
    def _find_records(dataset_dir: str) -> List[str]:
        required = [
            "pose.npy", "pose2.npy", "gripper_pos.npy", "gripper_pos2.npy",
            "target_pose.npy", "target_pose2.npy", "gripper.npy", "gripper2.npy",
        ]
        records = []
        if not os.path.isdir(dataset_dir):
            return records
        for item in sorted(os.listdir(dataset_dir)):
            p = os.path.join(dataset_dir, item)
            if not os.path.isdir(p):
                continue
            if all(os.path.exists(os.path.join(p, name)) for name in required):
                records.append(p)
        return records

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _ensure_col(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]
        return x

    def _load_lowdim_np(self, record_path: str) -> Tuple[np.ndarray, np.ndarray]:
        pose_l = np.load(os.path.join(record_path, "pose.npy")).astype(np.float32)
        pose_r = np.load(os.path.join(record_path, "pose2.npy")).astype(np.float32)
        grip_l = self._ensure_col(np.load(os.path.join(record_path, "gripper_pos.npy")).astype(np.float32))
        grip_r = self._ensure_col(np.load(os.path.join(record_path, "gripper_pos2.npy")).astype(np.float32))

        act_pose_l = np.load(os.path.join(record_path, "target_pose.npy")).astype(np.float32)
        act_pose_r = np.load(os.path.join(record_path, "target_pose2.npy")).astype(np.float32)
        act_grip_l = self._ensure_col(np.load(os.path.join(record_path, "gripper.npy")).astype(np.float32))
        act_grip_r = self._ensure_col(np.load(os.path.join(record_path, "gripper2.npy")).astype(np.float32))

        if self.use_pose10d:
            pose_l = pose6d_to_pose9d_np(pose_l)
            pose_r = pose6d_to_pose9d_np(pose_r)
            act_pose_l = pose6d_to_pose9d_np(act_pose_l)
            act_pose_r = pose6d_to_pose9d_np(act_pose_r)

        obs = np.concatenate([pose_l, grip_l, pose_r, grip_r], axis=-1).astype(np.float32)
        action_data = np.concatenate([act_pose_l, act_grip_l, act_pose_r, act_grip_r], axis=-1).astype(np.float32)

        L, D = action_data.shape
        action_target = np.zeros((L, self.future_steps, D), dtype=np.float32)
        for t in range(L):
            end = min(t + self.future_steps, L)
            valid = end - t
            action_target[t, :valid] = action_data[t:end]
            if valid < self.future_steps:
                action_target[t, valid:] = action_data[-1]

        return obs, action_target

    def _split_for_scaler(self, obs: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        data: Dict[str, torch.Tensor] = {}
        for i, key in enumerate(self.obs_keys):
            data[key] = obs[..., i:i + 1]
        for i, key in enumerate(self.act_keys):
            data[key] = actions[..., i:i + 1]
        return data

    def _apply_scaler(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.scaler is None or getattr(self.scaler, "mean_dict", None) is None:
            return obs, actions

        data = self._split_for_scaler(obs, actions)
        norm = self.scaler.normalize(data)
        obs_out = torch.cat([norm[k] for k in self.obs_keys], dim=-1)
        act_out = torch.cat([norm[k] for k in self.act_keys], dim=-1)
        return obs_out, act_out

    def fit_scaler(self, num_workers: int = 0):
        """
        只读取 npy，不读取图像；用于生成归一化文件，避免因图像路径/显存/IO 拖垮 scaler 生成。
        num_workers 保留为兼容参数，不再依赖 DataLoader。
        """
        if self.scaler is None:
            self.scaler = Scaler(lowdim_dict=self.lowdim_dict)

        all_obs, all_actions = [], []
        for record_path in tqdm(self.records, desc="Fitting scaler from npy only"):
            obs_np, act_np = self._load_lowdim_np(record_path)
            all_obs.append(torch.from_numpy(obs_np).float())
            all_actions.append(torch.from_numpy(act_np).float())

        all_obs_t = torch.cat(all_obs, dim=0)
        all_actions_t = torch.cat(all_actions, dim=0)
        data_dict = self._split_for_scaler(all_obs_t, all_actions_t)
        self.scaler.fit(data_dict)
        return self.scaler

    def save_scaler(self, filepath: str):
        if self.scaler is None:
            raise ValueError("Scaler is not initialized.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self.scaler.save(filepath)
        print(f"Scaler saved to {filepath}")

    def load_scaler(self, filepath: str):
        if self.scaler is None:
            self.scaler = Scaler(lowdim_dict=self.lowdim_dict)
        self.scaler.load(filepath)
        print(f"Scaler loaded from {filepath}")

    def _read_image(self, record_path: str, frame_idx: int, fallback: Optional[np.ndarray]) -> np.ndarray:
        w, h = self.resize_hw
        img_path = os.path.join(record_path, self.image_folder, f"{frame_idx:05d}.jpg")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            if fallback is not None:
                return fallback.copy()
            return np.zeros((3, h, w), dtype=np.float32)

        img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_rgb = np.transpose(img_rgb, (2, 0, 1))
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        img_rgb = (img_rgb - mean) / std
        return img_rgb.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record_path = self.records[idx]
        obs_np, actions_np = self._load_lowdim_np(record_path)
        L = obs_np.shape[0]

        images = []
        prev = None
        for t in range(L):
            img = self._read_image(record_path, t, prev)
            images.append(img)
            prev = img

        obs = torch.from_numpy(obs_np).float()
        actions = torch.from_numpy(actions_np).float()
        image = torch.from_numpy(np.stack(images, axis=0)).float()
        mask = torch.zeros(L, dtype=torch.bool)

        obs, actions = self._apply_scaler(obs, actions)
        return {
            "obs": obs,
            "actions": actions,
            "image": image,
            "mask": mask,
            "record_path": record_path,
        }


def parallel_collate_fn_image(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    lengths = [item["obs"].shape[0] for item in batch]
    max_len = max(lengths)

    padded_obs, padded_actions, padded_image, masks = [], [], [], []
    for item in batch:
        L = item["obs"].shape[0]
        obs_dim = item["obs"].shape[-1]
        F_steps, act_dim = item["actions"].shape[1:]
        C, H, W = item["image"].shape[1:]

        obs_pad = torch.zeros((max_len, obs_dim), dtype=torch.float32)
        obs_pad[:L] = item["obs"]
        padded_obs.append(obs_pad)

        act_pad = torch.zeros((max_len, F_steps, act_dim), dtype=torch.float32)
        act_pad[:L] = item["actions"]
        padded_actions.append(act_pad)

        img_pad = torch.zeros((max_len, C, H, W), dtype=torch.float32)
        img_pad[:L] = item["image"]
        padded_image.append(img_pad)

        mask = torch.ones(max_len, dtype=torch.bool)
        mask[:L] = False
        masks.append(mask)

    return {
        "obs": torch.stack(padded_obs, dim=0),
        "actions": torch.stack(padded_actions, dim=0),
        "image": torch.stack(padded_image, dim=0),
        "mask": torch.stack(masks, dim=0),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Fit a scaler for the real-world image dataset.")
    parser.add_argument("--task-name", default="cover_blocks")
    parser.add_argument("--data-root", required=True, help="Dataset root or train split directory.")
    parser.add_argument("--future-steps", type=int, default=16)
    parser.add_argument("--output-dir", default="scalers")
    parser.add_argument("--pose6d", action="store_true", help="Use raw 6D pose instead of pose10d layout.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    use_pose10d = not args.pose6d
    ds = ImageTrajectoryDataset(
        root_dir=args.data_root,
        mode="train",
        future_steps=args.future_steps,
        scaler=None,
        resize_hw=(640, 480),
        use_pose10d=use_pose10d,
    )
    ds.fit_scaler(num_workers=0)
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(
        args.output_dir,
        f"scaler_{args.task_name}_image_{'pose10d' if use_pose10d else 'pose6d'}.pth",
    )
    ds.save_scaler(save_path)
