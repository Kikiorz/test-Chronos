"""RoboTwin/RMBench adapter for ResNet-18 RGB plus native joint Chronos.

The joint vector is the 14-D drive-target state saved by RMBench, ordered as
left six joints, left gripper, right six joints, right gripper.

RoboTwin already exposes the head-camera frame as RGB.  This adapter keeps the
native uint8 frame untouched; the controller applies the official real-world
resize and ImageNet normalization exactly once, immediately before inference.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from .contracts import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES, RMBENCH_EMBODIMENT
from .mamba_controller_rgb_joint import MambaRGBJointController
from .resnet18_backbone import RMBENCH_RGB_HW


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
    expected_shape = (*RMBENCH_RGB_HW, 3)
    if image.shape != expected_shape:
        raise ValueError(
            f"Expected native {camera_name} RGB uint8 HWC frame with shape "
            f"{expected_shape}, got {image.shape}"
        )
    if image.dtype != np.uint8:
        raise TypeError(
            f"Native RMBench RGB must be uint8 so its scale is unambiguous, got {image.dtype}"
        )
    # No BGR swap and no /255 or ImageNet normalization here.  Copying into a
    # contiguous array prevents later mutation of the environment-owned frame.
    return np.ascontiguousarray(image).copy()


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
    return {"image": image_tensor, "qpos": qpos_tensor.float()}


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
