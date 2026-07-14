"""RMBench trajectories for single-camera RGB plus dual-arm joint control.

RMBench writes each RGB frame as a fixed-width JPEG byte string.  Its writer
passes the simulator's RGB array directly to OpenCV; consequently OpenCV decode
already returns the original numeric RGB channel order.  Do *not* apply an
additional BGR->RGB swap for these files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    from .contracts import JOINT_ORDER, TARGET_OFFSET
    from .dinov3_backbone import (
        DINOV3_CACHE_AMP_DTYPE,
        DINOV3_CACHE_BATCH_IMAGES,
        DINOV3_CACHE_DTYPE,
        DINOV3_CACHE_DEVICE_TYPE,
        DINOV3_CACHE_FORMAT_VERSION,
        DINOV3_CACHE_TOKENS,
        DINOV3_FEATURE_DIM,
        DINOV3_IMAGE_HW,
        DINOV3_MODEL_NAME,
        DINOV3_NORMALIZATION_MEAN,
        DINOV3_NORMALIZATION_STD,
        DINOV3_PATCH_SIZE,
        DINOV3_POOL_HW,
        DINOV3_RGB_SCALE,
        DINOV3_RESIZE_MODE,
        FLOAT32_NUMERICS,
        RMBENCH_RGB_HW,
        dinov3_cache_contract,
    )
except ImportError:  # direct script execution
    from scaler_M import Scaler
    from contracts import JOINT_ORDER, TARGET_OFFSET  # type: ignore
    from dinov3_backbone import (  # type: ignore
        DINOV3_CACHE_AMP_DTYPE,
        DINOV3_CACHE_BATCH_IMAGES,
        DINOV3_CACHE_DTYPE,
        DINOV3_CACHE_DEVICE_TYPE,
        DINOV3_CACHE_FORMAT_VERSION,
        DINOV3_CACHE_TOKENS,
        DINOV3_FEATURE_DIM,
        DINOV3_IMAGE_HW,
        DINOV3_MODEL_NAME,
        DINOV3_NORMALIZATION_MEAN,
        DINOV3_NORMALIZATION_STD,
        DINOV3_PATCH_SIZE,
        DINOV3_POOL_HW,
        DINOV3_RGB_SCALE,
        DINOV3_RESIZE_MODE,
        FLOAT32_NUMERICS,
        RMBENCH_RGB_HW,
        dinov3_cache_contract,
    )


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


JOINT_KEYS = list(JOINT_ORDER)
ACTION_KEYS = [f"{key}_act" for key in JOINT_KEYS]
JOINT_DIM = len(JOINT_KEYS)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_lowdim_dict(future_steps: int = 16) -> Dict[str, object]:
    result: Dict[str, object] = {key: 1 for key in JOINT_KEYS}
    result.update({key: (future_steps, 1) for key in ACTION_KEYS})
    return result


def make_joint_scaler(future_steps: int = 16) -> Scaler:
    return Scaler(make_lowdim_dict(future_steps))


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


class RGBJointTrajectoryDataset(Dataset):
    """One HDF5 episode with 14-D joint state and future joint targets.

    Without a feature cache, returns ``image`` as uint8 RGB ``[L,3,H,W]``.
    With ``feature_root``, returns frozen DINOv3 tokens as float32
    ``[L,21,768]`` and skips JPEG decoding.  The cache metadata is validated
    against the online encoder contract before any training sample is used.
    """

    def __init__(
        self,
        root_dir: str | Path,
        mode: str = "train",
        future_steps: int = 16,
        scaler: Optional[Scaler] = None,
        image_hw: Sequence[int] = (240, 320),
        camera_name: str = "head_camera",
        val_fraction: float = 0.1,
        split_seed: int = 42,
        sequence_length: Optional[int] = None,
        random_window: Optional[bool] = None,
        feature_root: str | Path | None = None,
    ):
        super().__init__()
        if future_steps != 16:
            raise ValueError("Chronos_RGB_Joint fixes future_steps=16")
        if len(image_hw) != 2 or min(int(image_hw[0]), int(image_hw[1])) <= 0:
            raise ValueError(f"image_hw must be (height,width), got {image_hw}")
        if tuple(int(value) for value in image_hw) != RMBENCH_RGB_HW:
            raise ValueError(
                "Chronos_RGB_Joint fixes the source RMBench RGB size at "
                f"{RMBENCH_RGB_HW}; got {tuple(image_hw)}. Intermediate OpenCV resizing "
                "would make raw RGB and cached DINOv3 paths inconsistent."
            )
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.mode = mode
        self.future_steps = int(future_steps)
        self.scaler = scaler
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.camera_name = camera_name
        self.val_fraction = float(val_fraction)
        self.split_seed = int(split_seed)
        # Optional short windows exist only for lightweight data diagnostics.
        # The formal training CLI never enables this: RGB fusion and Mamba see
        # the full episode, while only expensive head-loss timesteps are sampled.
        self.sequence_length = None if sequence_length in (None, 0) else int(sequence_length)
        if self.sequence_length is not None and self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive, zero, or None")
        self.random_window = (mode.lower() == "train") if random_window is None else bool(random_window)
        self.feature_root = (
            None if feature_root is None else Path(feature_root).expanduser().resolve()
        )
        self.feature_metadata: dict[str, object] | None = None
        if self.feature_root is not None:
            metadata_path = self.feature_root / "metadata.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(f"DINOv3 feature metadata is missing: {metadata_path}")
            with metadata_path.open("r", encoding="utf-8") as stream:
                metadata = json.load(stream)
            weights_sha256 = metadata.get("weights_sha256")
            if not isinstance(weights_sha256, str) or len(weights_sha256) != 64:
                raise ValueError("Feature metadata must contain a 64-character weights_sha256")
            expected = dinov3_cache_contract(self.camera_name, weights_sha256)
            mismatches = {
                key: (metadata.get(key), value)
                for key, value in expected.items()
                if metadata.get(key) != value
            }
            if mismatches:
                raise ValueError(f"Incompatible DINOv3 feature cache: {mismatches}")
            if metadata.get("cache_contract_sha256") != _json_sha256(expected):
                raise ValueError(
                    "DINOv3 cache contract fingerprint is missing or invalid; "
                    "regenerate the cache instead of relabeling existing .npy files"
                )
            episodes_manifest = metadata.get("episodes")
            if metadata.get("cache_dataset_sha256") != _json_sha256(episodes_manifest):
                raise ValueError("DINOv3 cache episode-manifest fingerprint is invalid")
            self.feature_metadata = metadata
        self.file_paths = discover_episode_files(
            self.root_dir, mode=mode, val_fraction=self.val_fraction, split_seed=self.split_seed
        )
        print(
            f"[{mode}] Chronos RGB Joint: {len(self.file_paths)} episodes | "
            f"camera={camera_name} | raw_image_hw={self.image_hw} | "
            f"dinov3_image_hw={DINOV3_IMAGE_HW} | "
            f"window={self.sequence_length or 'full'} | "
            f"vision={'cached-dinov3' if self.feature_root else 'online-rgb'} | "
            f"root={self.root_dir}"
        )
        if self.feature_root is not None:
            missing_features = [
                path.name for path in self.file_paths if not self._feature_path(path).is_file()
            ]
            if missing_features:
                raise FileNotFoundError(
                    f"Missing {len(missing_features)} DINOv3 feature files, "
                    f"including {missing_features[:5]}"
                )
            assert self.feature_metadata is not None
            episode_metadata = self.feature_metadata.get("episodes")
            if not isinstance(episode_metadata, Mapping):
                raise ValueError("Feature metadata is missing its episode manifest")
            all_source_files = discover_episode_files(self.root_dir, mode="all")
            expected_episode_names = {path.name for path in all_source_files}
            if set(episode_metadata) != expected_episode_names:
                raise ValueError(
                    "DINOv3 feature manifest does not exactly cover the source episodes"
                )
            manifest_total = sum(
                int(entry.get("frames", -1))
                for entry in episode_metadata.values()
                if isinstance(entry, Mapping)
            )
            if self.feature_metadata.get("total_frames") != manifest_total:
                raise ValueError("DINOv3 cache total_frames does not match its manifest")
            for path in self.file_paths:
                entry = episode_metadata.get(path.name)
                if not isinstance(entry, Mapping):
                    raise ValueError(f"Feature metadata has no entry for {path.name}")
                if entry.get("source_bytes") != path.stat().st_size:
                    raise ValueError(
                        f"Source HDF5 size changed after caching: {path.name}; "
                        "regenerate DINOv3 features"
                    )
                with h5py.File(path, "r") as root:
                    image_key = f"observation/{self.camera_name}/rgb"
                    if image_key not in root or "joint_action/vector" not in root:
                        raise KeyError(f"{path.name} lacks RGB or joint_action/vector")
                    frames = len(root["joint_action/vector"])
                    if len(root[image_key]) != frames or entry.get("frames") != frames:
                        raise ValueError(f"Feature frame manifest mismatch: {path.name}")
                source_sha256 = entry.get("source_sha256")
                feature_sha256 = entry.get("feature_sha256")
                if not isinstance(source_sha256, str) or len(source_sha256) != 64:
                    raise ValueError(f"Missing source SHA-256 for {path.name}")
                if not isinstance(feature_sha256, str) or len(feature_sha256) != 64:
                    raise ValueError(f"Missing feature SHA-256 for {path.name}")
                if _sha256_file(path) != source_sha256:
                    raise ValueError(f"Source HDF5 checksum changed: {path.name}")
                if _sha256_file(self._feature_path(path)) != feature_sha256:
                    raise ValueError(f"DINOv3 feature checksum changed: {path.name}")

    def __len__(self) -> int:
        return len(self.file_paths)

    @staticmethod
    def _load_lowdim(root: h5py.File) -> np.ndarray:
        component_keys = (
            "joint_action/left_arm",
            "joint_action/left_gripper",
            "joint_action/right_arm",
            "joint_action/right_gripper",
        )
        missing = [key for key in component_keys if key not in root]
        if missing:
            raise KeyError(f"HDF5 episode is missing joint keys: {missing}")

        left_arm = _as_column(root[component_keys[0]][()])
        left_gripper = _as_column(root[component_keys[1]][()])
        right_arm = _as_column(root[component_keys[2]][()])
        right_gripper = _as_column(root[component_keys[3]][()])
        lengths = {
            array.shape[0]
            for array in (left_arm, left_gripper, right_arm, right_gripper)
        }
        if len(lengths) != 1:
            raise ValueError(f"Joint arrays are not time-aligned: lengths={sorted(lengths)}")
        if left_arm.shape[1] != 6 or right_arm.shape[1] != 6:
            raise ValueError(
                "Chronos_RGB_Joint targets RMBench's aloha-agilex dual ARX5 "
                "embodiment with six joints per arm; "
                f"got left={left_arm.shape}, right={right_arm.shape}"
            )
        obs = np.concatenate(
            [left_arm, left_gripper, right_arm, right_gripper], axis=-1
        ).astype(np.float32, copy=False)
        if obs.shape[-1] != JOINT_DIM:
            raise ValueError(f"Expected dual-arm {JOINT_DIM}-D joint state, got {obs.shape}")
        if not np.isfinite(obs).all():
            raise FloatingPointError("RMBench joint state contains NaN or infinity")
        for label, column in (("left", 6), ("right", 13)):
            gripper = obs[:, column]
            if np.any(gripper < -1e-5) or np.any(gripper > 1.0 + 1e-5):
                raise ValueError(
                    f"{label} RMBench gripper is not a normalized [0,1] command"
                )

        # The writer stores this exact ordering as joint_action/vector.  Verify
        # it when present so a silent data-convention change cannot corrupt a run.
        if "joint_action/vector" in root:
            vector = np.asarray(root["joint_action/vector"][()], dtype=np.float32)
            if vector.shape != obs.shape or not np.allclose(vector, obs, rtol=0.0, atol=1e-6):
                raise ValueError(
                    "joint_action/vector does not match "
                    "[left_arm, left_gripper, right_arm, right_gripper]"
                )
        return obs

    def _make_future_actions(self, action_data: np.ndarray) -> np.ndarray:
        length = action_data.shape[0]
        if length == 0:
            raise ValueError("Empty HDF5 episode")
        indices = np.minimum(
            np.arange(length, dtype=np.int64)[:, None]
            + TARGET_OFFSET
            + np.arange(self.future_steps, dtype=np.int64)[None, :],
            length - 1,
        )
        return action_data[indices].astype(np.float32, copy=False)

    @staticmethod
    def _to_key_dict(tensor: torch.Tensor, keys: Sequence[str]) -> Dict[str, torch.Tensor]:
        return {key: tensor[..., index : index + 1] for index, key in enumerate(keys)}

    def _normalize(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.scaler is None:
            return obs, actions
        norm_obs = self.scaler.normalize(self._to_key_dict(obs, JOINT_KEYS))
        norm_actions = self.scaler.normalize(self._to_key_dict(actions, ACTION_KEYS))
        return (
            torch.cat([norm_obs[key] for key in JOINT_KEYS], dim=-1),
            torch.cat([norm_actions[key] for key in ACTION_KEYS], dim=-1),
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
                f"RGB/joint trajectory length mismatch: rgb={len(dataset)}, "
                f"joint={expected_length}"
            )
        images: List[np.ndarray] = []
        for frame_index in frame_indices.tolist():
            image = _decode_robotwin_rgb(dataset[frame_index])
            if tuple(image.shape[:2]) != self.image_hw:
                raise ValueError(
                    f"RMBench RGB frame has {tuple(image.shape[:2])}, expected {self.image_hw}"
                )
            images.append(np.ascontiguousarray(image.transpose(2, 0, 1)))
        image_array = np.stack(images, axis=0)
        return torch.from_numpy(image_array)

    def _feature_path(self, episode_path: Path) -> Path:
        if self.feature_root is None:
            raise RuntimeError("No feature_root was configured")
        return self.feature_root / f"{episode_path.stem}.npy"

    def _load_cached_features(
        self, episode_path: Path, expected_length: int, frame_indices: np.ndarray
    ) -> torch.Tensor:
        features = np.load(self._feature_path(episode_path), mmap_mode="r", allow_pickle=False)
        expected_shape = (expected_length, DINOV3_CACHE_TOKENS, DINOV3_FEATURE_DIM)
        if features.shape != expected_shape or features.dtype != np.float32:
            raise ValueError(
                f"Bad DINOv3 cache for {episode_path.name}: "
                f"shape={features.shape}, dtype={features.dtype}, expected={expected_shape}/float32"
            )
        selected = np.asarray(features[frame_indices], dtype=np.float32)
        if not np.isfinite(selected).all():
            raise FloatingPointError(f"Non-finite DINOv3 cache: {episode_path.name}")
        return torch.from_numpy(selected.copy())

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path = self.file_paths[index]
        with h5py.File(path, "r") as root:
            obs_np = self._load_lowdim(root)
            image_key = f"observation/{self.camera_name}/rgb"
            if image_key not in root:
                raise KeyError(f"{path.name} does not contain {image_key!r}")
            frame_indices = self._window_indices(obs_np.shape[0])
            if self.feature_root is None:
                vision = self._load_images(
                    root[image_key], expected_length=obs_np.shape[0], frame_indices=frame_indices
                )
            else:
                if len(root[image_key]) != obs_np.shape[0]:
                    raise ValueError(
                        f"RGB/joint length mismatch: rgb={len(root[image_key])}, "
                        f"joint={obs_np.shape[0]}"
                    )
                vision = self._load_cached_features(
                    path, expected_length=obs_np.shape[0], frame_indices=frame_indices
                )

        actions_np = self._make_future_actions(obs_np.copy())
        obs_np = obs_np[frame_indices]
        actions_np = actions_np[frame_indices]
        obs = torch.from_numpy(obs_np)
        actions = torch.from_numpy(actions_np)
        obs, actions = self._normalize(obs, actions)
        result = {
            "obs": obs,
            "actions": actions,
            "mask": torch.zeros(obs.shape[0], dtype=torch.bool),
            "episode_index": torch.tensor(index, dtype=torch.long),
        }
        result["image_features" if self.feature_root is not None else "image"] = vision
        return result

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
        scaler = self.scaler if self.scaler is not None else make_joint_scaler(self.future_steps)
        data: Dict[str, torch.Tensor] = self._to_key_dict(obs, JOINT_KEYS)
        data.update(self._to_key_dict(actions, ACTION_KEYS))
        scaler.fit(data)
        scaler.validate_fitted()
        self.scaler = scaler
        return scaler


def parallel_collate_fn_rgb_joint(
    batch: Sequence[Mapping[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    lengths = [int(item["obs"].shape[0]) for item in batch]
    max_length = max(lengths)
    batch_size = len(batch)
    future_steps, action_dim = batch[0]["actions"].shape[1:]
    has_cached_features = "image_features" in batch[0]
    if any(("image_features" in item) != has_cached_features for item in batch):
        raise ValueError("A batch cannot mix raw images and cached DINOv3 features")
    if has_cached_features:
        token_count, feature_dim = batch[0]["image_features"].shape[1:]
    else:
        channels, height, width = batch[0]["image"].shape[1:]

    obs = torch.zeros(batch_size, max_length, JOINT_DIM, dtype=torch.float32)
    actions = torch.zeros(
        batch_size, max_length, future_steps, action_dim, dtype=torch.float32
    )
    if has_cached_features:
        vision = torch.zeros(
            batch_size,
            max_length,
            token_count,
            feature_dim,
            dtype=torch.float32,
        )
    else:
        vision = torch.zeros(
            batch_size, max_length, channels, height, width, dtype=torch.uint8
        )
    mask = torch.ones(batch_size, max_length, dtype=torch.bool)
    episode_indices = torch.empty(batch_size, dtype=torch.long)
    for batch_index, item in enumerate(batch):
        length = lengths[batch_index]
        vision_key = "image_features" if has_cached_features else "image"
        if tuple(item[vision_key].shape[1:]) != tuple(vision.shape[2:]):
            raise ValueError("All vision entries in a batch must share a shape")
        obs[batch_index, :length] = item["obs"]
        actions[batch_index, :length] = item["actions"]
        vision[batch_index, :length] = item[vision_key]
        mask[batch_index, :length] = False
        episode_indices[batch_index] = item["episode_index"]
    result = {
        "obs": obs,
        "actions": actions,
        "mask": mask,
        "episode_index": episode_indices,
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }
    result["image_features" if has_cached_features else "image"] = vision
    return result


# Compatibility alias for code that followed the 3D dataset's naming style.
ImageJointTrajectoryDataset = RGBJointTrajectoryDataset


def _main() -> None:
    parser = argparse.ArgumentParser(description="Fit the Chronos RGB 14-D joint scaler")
    parser.add_argument("--data-root", required=True, help="RMBench .../demo_clean/data")
    parser.add_argument("--output", required=True, help="Output .pth scaler path")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    dataset = RGBJointTrajectoryDataset(
        args.data_root,
        mode="train",
        scaler=None,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
    )
    scaler = dataset.fit_scaler()
    scaler.save(args.output)
    print(f"Saved scaler fitted on {len(dataset)} train episodes to {args.output}")


if __name__ == "__main__":
    _main()
