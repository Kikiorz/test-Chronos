"""RMBench trajectory dataset for the single-head-camera RGB experiment.

RMBench writes each RGB frame as a fixed-width JPEG byte string.  Its writer
passes the simulator's RGB array directly to OpenCV; consequently OpenCV decode
already returns the original numeric RGB channel order.  Do *not* apply an
additional BGR->RGB swap for these files.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .scaler_M import Scaler
except ImportError:  # direct script execution
    from scaler_M import Scaler


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


OBS_KEYS = [
    "ee_l_x", "ee_l_y", "ee_l_z", "ee_l_qw", "ee_l_qx", "ee_l_qy", "ee_l_qz", "gripper_l",
    "ee_r_x", "ee_r_y", "ee_r_z", "ee_r_qw", "ee_r_qx", "ee_r_qy", "ee_r_qz", "gripper_r",
]
ACT_KEYS = [f"{key}_act" for key in OBS_KEYS]

# Match both official Chronos datasets: the action window associated with
# frame t starts at t and extends through t + future_steps - 1.  RMBench does
# not store a separate command stream, so its EE observations are reused as
# action targets, exactly as in M_dataset_robotwin3D_E.py.
ACTION_TARGET_OFFSET = 0


def make_lowdim_dict(future_steps: int = 16) -> Dict[str, object]:
    result: Dict[str, object] = {key: 1 for key in OBS_KEYS}
    result.update({key: (future_steps, 1) for key in ACT_KEYS})
    return result


def make_ee_scaler(future_steps: int = 16) -> Scaler:
    return Scaler(make_lowdim_dict(future_steps))


def make_future_action_targets(
    action_data: np.ndarray,
    future_steps: int = 16,
    target_offset: int = ACTION_TARGET_OFFSET,
) -> np.ndarray:
    """Return targets ``[t+offset, ..., t+offset+Q-1]`` with tail clamping."""

    action_data = np.asarray(action_data)
    if action_data.ndim < 1 or action_data.shape[0] == 0:
        raise ValueError("Empty action trajectory")
    if future_steps <= 0:
        raise ValueError(f"future_steps must be positive, got {future_steps}")
    if target_offset < 0:
        raise ValueError(f"target_offset must be non-negative, got {target_offset}")
    indices = np.minimum(
        np.arange(action_data.shape[0], dtype=np.int64)[:, None]
        + int(target_offset)
        + np.arange(int(future_steps), dtype=np.int64)[None, :],
        action_data.shape[0] - 1,
    )
    return action_data[indices].astype(np.float32, copy=False)


def _natural_episode_key(path: Path) -> Tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name))


def _files_in(path: Path) -> List[Path]:
    if not path.is_dir():
        return []
    return sorted(
        [*path.glob("*.hdf5"), *path.glob("*.h5")],
        key=_natural_episode_key,
    )


def discover_episode_files(
    root_dir: str | Path,
    mode: str,
    val_fraction: float = 0.1,
    split_seed: int = 42,
) -> List[Path]:
    """Discover files and perform a deterministic episode-level split.

    Explicit ``train/`` and ``val/`` (or ``test/``) directories take priority.
    Otherwise, a flat RMBench ``data/`` directory is shuffled once by episode;
    no frames from an episode can leak between train and validation.
    """

    root = Path(root_dir).expanduser().resolve()
    mode = mode.lower()
    if mode not in {"train", "val", "valid", "test", "all"}:
        raise ValueError(f"Unsupported mode {mode!r}; use train, val, test, or all")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0,1), got {val_fraction}")

    aliases = {
        "train": ("train",),
        "val": ("val", "valid", "test"),
        "valid": ("valid", "val", "test"),
        "test": ("test", "val", "valid"),
    }
    split_dirs = {name: root / name for name in ("train", "val", "valid", "test")}
    has_explicit_splits = any(_files_in(path) for path in split_dirs.values())
    if has_explicit_splits and mode == "all":
        merged = {path for split_path in split_dirs.values() for path in _files_in(split_path)}
        return sorted(merged, key=_natural_episode_key)
    if has_explicit_splits and mode != "all":
        for alias in aliases[mode]:
            files = _files_in(split_dirs[alias])
            if files:
                return files
        raise FileNotFoundError(f"No explicit {mode!r} HDF5 split found below {root}")

    # If root itself is an explicit split directory, do not split it again.
    flat_files = _files_in(root)
    if root.name.lower() in {"train", "val", "valid", "test"} and flat_files:
        return flat_files

    if not flat_files:
        raise FileNotFoundError(
            f"No .hdf5/.h5 episodes found directly in {root}. "
            "Pass the RMBench .../demo_clean/data directory."
        )
    if mode == "all":
        return flat_files
    if len(flat_files) < 2:
        if mode == "train":
            return flat_files
        raise ValueError("At least two episodes are required for a held-out validation split")
    if val_fraction <= 0:
        if mode == "train":
            return flat_files
        raise ValueError("val_fraction must be > 0 when requesting validation data")

    rng = np.random.default_rng(split_seed)
    permutation = rng.permutation(len(flat_files))
    num_val = max(1, int(round(len(flat_files) * val_fraction)))
    num_val = min(num_val, len(flat_files) - 1)
    val_indices = set(permutation[:num_val].tolist())
    if mode == "train":
        return [path for index, path in enumerate(flat_files) if index not in val_indices]
    return [path for index, path in enumerate(flat_files) if index in val_indices]


def _as_column(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    return array.reshape(array.shape[0], -1)


def _decode_robotwin_rgb(value: object) -> np.ndarray:
    """Decode one RMBench frame while preserving its original RGB ordering."""

    if isinstance(value, (bytes, bytearray, np.bytes_)):
        encoded = np.frombuffer(bytes(value), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("cv2.imdecode failed for an RMBench JPEG frame")
        # No cvtColor here: see module docstring and pkl2hdf5.images_encoding.
        return image

    array = np.asarray(value)
    if array.ndim == 1 and array.dtype == np.uint8:
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("cv2.imdecode failed for an RMBench uint8 JPEG frame")
        return image
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        return np.asarray(array[..., :3], dtype=np.uint8)
    raise TypeError(f"Unsupported RGB frame encoding: dtype={array.dtype}, shape={array.shape}")


class RGBTrajectoryDataset(Dataset):
    """One HDF5 episode per sample, with 16-D EE state/action windows.

    Returns ``image`` as uint8 RGB ``[L,3,H,W]``.  Keeping full trajectories in
    uint8 reduces loader RAM by 4x.  Float conversion and ImageNet normalization
    intentionally live in :class:`ImageMambaFusion`, so online deployment and
    offline training share exactly the same preprocessing.
    """

    def __init__(
        self,
        root_dir: str | Path,
        mode: str = "train",
        future_steps: int = 16,
        scaler: Optional[Scaler] = None,
        image_hw: Sequence[int] = (480, 640),
        camera_name: str = "head_camera",
        val_fraction: float = 0.1,
        split_seed: int = 42,
        action_target_offset: int = ACTION_TARGET_OFFSET,
        sequence_length: Optional[int] = None,
        random_window: Optional[bool] = None,
    ):
        super().__init__()
        if future_steps != 16:
            raise ValueError("Official RMBench Chronos fixes future_steps=16")
        if len(image_hw) != 2 or min(int(image_hw[0]), int(image_hw[1])) <= 0:
            raise ValueError(f"image_hw must be (height,width), got {image_hw}")
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.mode = mode
        self.future_steps = int(future_steps)
        self.scaler = scaler
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.camera_name = camera_name
        self.val_fraction = float(val_fraction)
        self.split_seed = int(split_seed)
        self.action_target_offset = int(action_target_offset)
        if self.action_target_offset < 0:
            raise ValueError("action_target_offset must be non-negative")
        # Optional short windows exist only for lightweight data diagnostics.
        # The formal training CLI never enables this: RGB fusion, Mamba and the
        # released action loss all see the complete padded episode.
        self.sequence_length = None if sequence_length in (None, 0) else int(sequence_length)
        if self.sequence_length is not None and self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive, zero, or None")
        self.random_window = (mode.lower() == "train") if random_window is None else bool(random_window)
        self.file_paths = discover_episode_files(
            self.root_dir, mode=mode, val_fraction=self.val_fraction, split_seed=self.split_seed
        )
        print(
            f"[{mode}] Chronos RGB: {len(self.file_paths)} episodes | "
            f"camera={camera_name} | image_hw={self.image_hw} | "
            f"action_offset={self.action_target_offset} | "
            f"window={self.sequence_length or 'full'} | root={self.root_dir}"
        )

    def __len__(self) -> int:
        return len(self.file_paths)

    @staticmethod
    def _load_lowdim(root: h5py.File) -> np.ndarray:
        required = (
            "endpose/left_endpose",
            "endpose/left_gripper",
            "endpose/right_endpose",
            "endpose/right_gripper",
        )
        missing = [key for key in required if key not in root]
        if missing:
            raise KeyError(f"HDF5 episode is missing EE keys: {missing}")
        left_ee = _as_column(root[required[0]][()])
        left_gripper = _as_column(root[required[1]][()])
        right_ee = _as_column(root[required[2]][()])
        right_gripper = _as_column(root[required[3]][()])
        lengths = {array.shape[0] for array in (left_ee, left_gripper, right_ee, right_gripper)}
        if len(lengths) != 1:
            raise ValueError(f"EE arrays are not time-aligned: lengths={sorted(lengths)}")
        obs = np.concatenate([left_ee, left_gripper, right_ee, right_gripper], axis=-1)
        if obs.shape[-1] != 16:
            raise ValueError(f"Expected dual-arm 16-D EE state, got shape={obs.shape}")
        return obs.astype(np.float32, copy=False)

    def _make_future_actions(self, action_data: np.ndarray) -> np.ndarray:
        return make_future_action_targets(
            action_data,
            future_steps=self.future_steps,
            target_offset=self.action_target_offset,
        )

    @staticmethod
    def _to_key_dict(tensor: torch.Tensor, keys: Sequence[str]) -> Dict[str, torch.Tensor]:
        return {key: tensor[..., index : index + 1] for index, key in enumerate(keys)}

    def _normalize(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.scaler is None:
            return obs, actions
        norm_obs = self.scaler.normalize(self._to_key_dict(obs, OBS_KEYS))
        norm_actions = self.scaler.normalize(self._to_key_dict(actions, ACT_KEYS))
        return (
            torch.cat([norm_obs[key] for key in OBS_KEYS], dim=-1),
            torch.cat([norm_actions[key] for key in ACT_KEYS], dim=-1),
        )

    def _window_indices(self, episode_length: int) -> np.ndarray:
        if self.sequence_length is None or self.sequence_length >= episode_length:
            return np.arange(episode_length, dtype=np.int64)
        max_start = episode_length - self.sequence_length
        if self.random_window:
            # DataLoader seeds torch independently in every worker.  Drawing
            # here yields a fresh contiguous window each epoch without ever
            # mixing frames across episodes or train/validation splits.
            start = int(torch.randint(max_start + 1, (1,)).item())
        else:
            start = max_start // 2
        return np.arange(start, start + self.sequence_length, dtype=np.int64)

    def _load_images(
        self, dataset: h5py.Dataset, expected_length: int, frame_indices: np.ndarray
    ) -> torch.Tensor:
        if len(dataset) != expected_length:
            raise ValueError(
                f"RGB/EE trajectory length mismatch: rgb={len(dataset)}, ee={expected_length}"
            )
        target_h, target_w = self.image_hw
        images: List[np.ndarray] = []
        for frame_index in frame_indices.tolist():
            image = _decode_robotwin_rgb(dataset[frame_index])
            if image.shape[:2] != (target_h, target_w):
                # Match real_wolrd training/inference exactly, including its
                # fixed INTER_AREA choice.  RMBench frames are upsampled from
                # 320x240 to the encoder's required 640x480 input.
                image = cv2.resize(
                    image, (target_w, target_h), interpolation=cv2.INTER_AREA
                )
            images.append(np.ascontiguousarray(image.transpose(2, 0, 1)))
        image_array = np.stack(images, axis=0)
        return torch.from_numpy(image_array)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path = self.file_paths[index]
        with h5py.File(path, "r") as root:
            obs_np = self._load_lowdim(root)
            image_key = f"observation/{self.camera_name}/rgb"
            if image_key not in root:
                raise KeyError(f"{path.name} does not contain {image_key!r}")
            frame_indices = self._window_indices(obs_np.shape[0])
            images = self._load_images(
                root[image_key], expected_length=obs_np.shape[0], frame_indices=frame_indices
            )

        actions_np = self._make_future_actions(obs_np.copy())
        obs_np = obs_np[frame_indices]
        actions_np = actions_np[frame_indices]
        obs = torch.from_numpy(obs_np)
        actions = torch.from_numpy(actions_np)
        obs, actions = self._normalize(obs, actions)
        return {
            "obs": obs,
            "actions": actions,
            "image": images,
            "mask": torch.zeros(obs.shape[0], dtype=torch.bool),
            "episode_index": torch.tensor(index, dtype=torch.long),
        }

    def fit_scaler(self) -> Scaler:
        """Fit only from this dataset's episodes; images are not decoded."""

        all_obs: List[torch.Tensor] = []
        all_actions: List[torch.Tensor] = []
        for path in self.file_paths:
            with h5py.File(path, "r") as root:
                obs_np = self._load_lowdim(root)
            all_obs.append(torch.from_numpy(obs_np))
            all_actions.append(torch.from_numpy(self._make_future_actions(obs_np)))
        if not all_obs:
            raise RuntimeError("Cannot fit scaler on an empty training split")
        obs = torch.cat(all_obs, dim=0)
        actions = torch.cat(all_actions, dim=0)
        scaler = self.scaler if self.scaler is not None else make_ee_scaler(self.future_steps)
        data: Dict[str, torch.Tensor] = self._to_key_dict(obs, OBS_KEYS)
        data.update(self._to_key_dict(actions, ACT_KEYS))
        scaler.fit(data)
        self.scaler = scaler
        return scaler


def parallel_collate_fn_rgb(batch: Sequence[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    lengths = [int(item["obs"].shape[0]) for item in batch]
    max_length = max(lengths)
    batch_size = len(batch)
    future_steps, action_dim = batch[0]["actions"].shape[1:]
    channels, height, width = batch[0]["image"].shape[1:]

    obs = torch.zeros(batch_size, max_length, 16, dtype=torch.float32)
    actions = torch.zeros(
        batch_size, max_length, future_steps, action_dim, dtype=torch.float32
    )
    images = torch.zeros(
        batch_size, max_length, channels, height, width, dtype=torch.uint8
    )
    mask = torch.ones(batch_size, max_length, dtype=torch.bool)
    episode_indices = torch.empty(batch_size, dtype=torch.long)
    for batch_index, item in enumerate(batch):
        length = lengths[batch_index]
        if tuple(item["image"].shape[1:]) != (channels, height, width):
            raise ValueError("All images in a batch must share a resized shape")
        obs[batch_index, :length] = item["obs"]
        actions[batch_index, :length] = item["actions"]
        images[batch_index, :length] = item["image"]
        mask[batch_index, :length] = False
        episode_indices[batch_index] = item["episode_index"]
    return {
        "obs": obs,
        "actions": actions,
        "image": images,
        "mask": mask,
        "episode_index": episode_indices,
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }


# Compatibility alias for code that followed the 3D dataset's naming style.
ImageTrajectoryDataset = RGBTrajectoryDataset


def _main() -> None:
    parser = argparse.ArgumentParser(description="Fit the Chronos RGB 16-D EE scaler")
    parser.add_argument("--data-root", required=True, help="RMBench .../demo_clean/data")
    parser.add_argument("--output", required=True, help="Output .pth scaler path")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--action-target-offset", type=int, default=ACTION_TARGET_OFFSET)
    args = parser.parse_args()
    dataset = RGBTrajectoryDataset(
        args.data_root,
        mode="train",
        scaler=None,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        action_target_offset=args.action_target_offset,
    )
    scaler = dataset.fit_scaler()
    scaler.save(args.output)
    print(f"Saved scaler fitted on {len(dataset)} train episodes to {args.output}")


if __name__ == "__main__":
    _main()
