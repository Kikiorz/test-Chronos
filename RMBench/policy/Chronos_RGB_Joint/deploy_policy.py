"""RoboTwin/RMBench adapter for DINOv3 RGB plus native joint Chronos.

The joint vector is the 14-D drive-target state saved by RMBench, ordered as
left six joints, left gripper, right six joints, right gripper.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from .dinov3_backbone import RMBENCH_RGB_HW
from .contracts import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES, RMBENCH_EMBODIMENT
from .mamba_controller_rgb_joint import MambaRGBJointController


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

    if tuple(image.shape[:2]) != RMBENCH_RGB_HW:
        raise ValueError(
            f"Expected native RMBench RGB size {RMBENCH_RGB_HW}, got {image.shape[:2]}"
        )
    # SAPIEN/RoboTwin already returns RGB (not OpenCV BGR), so there is no
    # red/blue channel swap here.  Preserve dtype information: an extremely
    # dark uint8 frame with max 0 or 1 still needs division by 255.
    integer_input = np.issubdtype(image.dtype, np.integer)
    image = image.astype(np.float32, copy=False)
    if not np.isfinite(image).all():
        raise ValueError("RGB image contains NaN or infinity")
    minimum = float(image.min()) if image.size else 0.0
    maximum = float(image.max()) if image.size else 0.0
    if minimum < 0.0 or maximum > 255.0:
        raise ValueError("RGB values must lie in [0,255]")
    if not integer_input and 1.0 < maximum <= 1.5:
        raise ValueError(
            "Float RGB has ambiguous scale: expected [0,1] or an unambiguous [0,255] frame"
        )
    if integer_input or maximum > 1.5:
        # Multiplication by the rounded float32 reciprocal is bit-identical to
        # CUDA's uint8.float().div(255) for every possible byte value.  NumPy's
        # direct float32 division differs by one ULP for some values, which is
        # then amplified by bicubic resize and DINOv3.
        image = image * np.float32(1.0 / 255.0)
    if image.size and (float(image.min()) < 0.0 or float(image.max()) > 1.0):
        raise ValueError("RGB scaling did not produce values in [0,1]")
    return image


def _extract_joint_state(observation: Mapping[str, Any]) -> np.ndarray:
    joint_root = observation.get("joint_action")
    if not isinstance(joint_root, Mapping) or "vector" not in joint_root:
        raise KeyError("Expected observation['joint_action']['vector']")
    state = np.asarray(joint_root["vector"], dtype=np.float32).reshape(-1)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError(f"Expected one finite 14-D RMBench joint vector, got {state.shape}")
    if not (0.0 - 1e-5 <= state[6] <= 1.0 + 1e-5):
        raise ValueError(f"Left gripper drive target is outside [0,1]: {state[6]}")
    if not (0.0 - 1e-5 <= state[13] <= 1.0 + 1e-5):
        raise ValueError(f"Right gripper drive target is outside [0,1]: {state[13]}")
    return state


def encode_obs(observation: Mapping[str, Any], camera_name: str = "head_camera") -> dict[str, torch.Tensor]:
    """Convert one online RMBench observation without changing RGB channel order."""

    image = _extract_head_rgb(observation, camera_name)
    qpos = _extract_joint_state(observation)
    image_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0)
    qpos_tensor = torch.from_numpy(qpos).unsqueeze(0)
    return {"image": image_tensor.float(), "qpos": qpos_tensor.float()}


def get_model(usr_args: Mapping[str, Any]) -> MambaRGBJointController:
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
        "clip_to_training_range": bool(usr_args.get("clip_to_training_range", True)),
        "training_range_margin": float(usr_args.get("training_range_margin", 0.05)),
    }
    return MambaRGBJointController(controller_args)


def _validate_environment_joint_identity(task_env: Any) -> None:
    robot = getattr(task_env, "robot", None)
    if robot is None:
        raise RuntimeError("RMBench task has no initialized robot to validate")
    left = tuple(str(value) for value in getattr(robot, "left_arm_joints_name", ()))
    right = tuple(str(value) for value in getattr(robot, "right_arm_joints_name", ()))
    if left != LEFT_ARM_JOINT_NAMES or right != RIGHT_ARM_JOINT_NAMES:
        raise RuntimeError(
            f"This checkpoint is bound to {RMBENCH_EMBODIMENT} joints "
            f"left={LEFT_ARM_JOINT_NAMES}, right={RIGHT_ARM_JOINT_NAMES}; "
            f"the active environment exposes left={left}, right={right}."
        )


def eval(TASK_ENV: Any, model: MambaRGBJointController, observation: Mapping[str, Any]) -> None:
    _validate_environment_joint_identity(TASK_ENV)
    obs = encode_obs(observation, camera_name=model.camera_name)
    action = model.get_action(obs)
    if action.shape != (14,) or not np.isfinite(action).all():
        raise ValueError(f"Controller returned an invalid qpos action: {action.shape}")
    before_drive = obs["qpos"][0].cpu().numpy().copy()
    TASK_ENV.take_action(action, action_type="qpos")
    robot = getattr(TASK_ENV, "robot", None)
    if robot is not None and hasattr(model, "record_execution"):
        try:
            after_drive = np.asarray(
                robot.get_left_arm_jointState() + robot.get_right_arm_jointState(),
                dtype=np.float32,
            )
            after_real = np.asarray(
                robot.get_left_arm_real_jointState()
                + robot.get_right_arm_real_jointState(),
                dtype=np.float32,
            )
            model.record_execution(before_drive, action, after_drive, after_real)
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            print(f"[Chronos_RGB_Joint] execution monitor unavailable: {exc}")


def reset_model(model: MambaRGBJointController) -> None:
    """Start every evaluation episode with fresh Mamba and temporal state."""

    model.reset()
