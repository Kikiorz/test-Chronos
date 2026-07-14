"""RoboTwin/RMBench adapter for the single-head-camera RGB Chronos policy.

The environment provides RGB images in HWC layout and dual-arm end-effector
state in ``observation['endpose']``.  The controller consumes NCHW RGB plus a
16-D EE state and produces a 16-D EE action.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from .mamba_controller_rgb import MambaRGBController


def _extract_head_rgb(observation: Mapping[str, Any], camera_name: str) -> np.ndarray:
    camera_root = observation.get("observation")
    if not isinstance(camera_root, Mapping):
        raise KeyError("Expected observation['observation'] to contain camera data")

    camera = camera_root.get(camera_name)
    if not isinstance(camera, Mapping) or "rgb" not in camera:
        available = sorted(str(key) for key in camera_root.keys())
        raise KeyError(
            f"Missing observation['observation']['{camera_name}']['rgb']; "
            f"available camera entries: {available}"
        )

    image = np.asarray(camera["rgb"])
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(
            f"Expected {camera_name} RGB in HWC layout with 3 or 4 channels, got {image.shape}"
        )
    image = image[..., :3]
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError(f"RGB image must be numeric, got dtype={image.dtype}")

    # SAPIEN/RoboTwin already returns RGB (not OpenCV BGR), so there is no
    # red/blue channel swap here.
    image = image.astype(np.float32, copy=False)
    if image.size and float(np.nanmax(image)) > 1.5:
        image = image / 255.0
    if not np.isfinite(image).all():
        raise ValueError("RGB image contains NaN or infinity")
    image = np.clip(image, 0.0, 1.0)
    return image


def _as_scalar(value: Any, name: str) -> float:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != 1:
        raise ValueError(f"{name} must contain one scalar, got shape {np.asarray(value).shape}")
    result = float(array[0])
    if not np.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _extract_ee_state(observation: Mapping[str, Any]) -> np.ndarray:
    endpose = observation.get("endpose")
    if not isinstance(endpose, Mapping):
        raise KeyError("Expected observation['endpose'] for the dual-arm EE state")

    required = ("left_endpose", "left_gripper", "right_endpose", "right_gripper")
    missing = [key for key in required if key not in endpose]
    if missing:
        raise KeyError(f"Missing dual-arm endpose fields: {missing}")

    left = np.asarray(endpose["left_endpose"], dtype=np.float32).reshape(-1)
    right = np.asarray(endpose["right_endpose"], dtype=np.float32).reshape(-1)
    if left.size != 7 or right.size != 7:
        raise ValueError(
            "Chronos_RGB expects 7-D [xyz, quaternion] poses for both arms; "
            f"got left={left.shape}, right={right.shape}"
        )

    state = np.concatenate(
        (
            left,
            np.asarray([_as_scalar(endpose["left_gripper"], "left_gripper")], dtype=np.float32),
            right,
            np.asarray([_as_scalar(endpose["right_gripper"], "right_gripper")], dtype=np.float32),
        )
    )
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError(f"Expected one finite 16-D EE state, got {state.shape}")
    return state


def encode_obs(observation: Mapping[str, Any], camera_name: str = "head_camera") -> dict[str, torch.Tensor]:
    """Convert one online RMBench observation without changing RGB channel order."""

    image = _extract_head_rgb(observation, camera_name)
    qpos = _extract_ee_state(observation)
    image_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0)
    qpos_tensor = torch.from_numpy(qpos).unsqueeze(0)
    return {"image": image_tensor.float(), "qpos": qpos_tensor.float()}


def get_model(usr_args: Mapping[str, Any]) -> MambaRGBController:
    requested_device = str(usr_args.get("device", "auto"))
    if requested_device.lower() == "auto":
        requested_device = f"cuda:{int(usr_args.get('gpu_id', 0))}" if torch.cuda.is_available() else "cpu"

    controller_args = {
        "device": requested_device,
        "ckpt_path": usr_args["ckpt_path"],
        "scaler_path": usr_args["scaler_path"],
        "camera_name": usr_args.get("camera_name", "head_camera"),
        "future_steps": int(usr_args.get("future_steps", 16)),
        "sample_steps": int(usr_args.get("sample_steps", 5)),
        "temporal_agg": bool(usr_args.get("temporal_agg", True)),
        "temporal_decay": float(usr_args.get("temporal_decay", 0.01)),
    }
    return MambaRGBController(controller_args)


def eval(TASK_ENV: Any, model: MambaRGBController, observation: Mapping[str, Any]) -> None:
    obs = encode_obs(observation, camera_name=model.camera_name)
    action = model.get_action(obs)
    TASK_ENV.take_action(action, action_type="ee")


def reset_model(model: MambaRGBController) -> None:
    """Start every evaluation episode with fresh Mamba and temporal state."""

    model.reset()
