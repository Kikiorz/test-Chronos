"""Shared immutable contracts for RGB-Joint training, resume, and deployment."""

from __future__ import annotations

try:
    from .dinov3_backbone import (
        DINOV3_IMAGE_HW,
        DINOV3_CACHE_TOKENS,
        DINOV3_FEATURE_DIM,
        DINOV3_MODEL_NAME,
        DINOV3_NORMALIZATION_MEAN,
        DINOV3_NORMALIZATION_STD,
        DINOV3_RESIZE_MODE,
        DINOV3_RGB_SCALE,
        DINOV3_PATCH_SIZE,
        DINOV3_POOL_HW,
        FLOAT32_NUMERICS,
        RMBENCH_RGB_HW,
    )
except ImportError:  # direct script execution
    from dinov3_backbone import (  # type: ignore
        DINOV3_IMAGE_HW,
        DINOV3_CACHE_TOKENS,
        DINOV3_FEATURE_DIM,
        DINOV3_MODEL_NAME,
        DINOV3_NORMALIZATION_MEAN,
        DINOV3_NORMALIZATION_STD,
        DINOV3_RESIZE_MODE,
        DINOV3_RGB_SCALE,
        DINOV3_PATCH_SIZE,
        DINOV3_POOL_HW,
        FLOAT32_NUMERICS,
        RMBENCH_RGB_HW,
    )


POLICY_VARIANT = "chronos_rgb_joint14_dinov3b"
POLICY_CONTRACT_VERSION = 5
CAMERA_NAME = "head_camera"
TARGET_OFFSET = 1
INFERENCE_SAMPLE_STEPS = 5
RMBENCH_EMBODIMENT = "aloha-agilex"
LEFT_ARM_JOINT_NAMES = tuple(f"fl_joint{index}" for index in range(1, 7))
RIGHT_ARM_JOINT_NAMES = tuple(f"fr_joint{index}" for index in range(1, 7))
JOINT_ORDER = (
    *LEFT_ARM_JOINT_NAMES,
    "left_gripper_normalized",
    *RIGHT_ARM_JOINT_NAMES,
    "right_gripper_normalized",
)


def base_policy_contract() -> dict[str, object]:
    """Return a fresh JSON-serializable static policy contract."""

    return {
        "contract_version": POLICY_CONTRACT_VERSION,
        "variant": POLICY_VARIANT,
        "rmbench_embodiment": RMBENCH_EMBODIMENT,
        "robot_model": "AgileX Aloha dual ARX5",
        "camera_name": CAMERA_NAME,
        "joint_order": list(JOINT_ORDER),
        "joint_state_semantics": (
            "RMBench arm drive_target radians plus normalized gripper command"
        ),
        "action_dim": 14,
        "future_steps": 16,
        "target_offset": TARGET_OFFSET,
        "action_type": "qpos",
        "backbone": DINOV3_MODEL_NAME,
        "source_image_hw": list(RMBENCH_RGB_HW),
        "image_hw": list(DINOV3_IMAGE_HW),
        "image_preprocess": {
            "channel_order": "RGB",
            "input_scale": DINOV3_RGB_SCALE,
            "resize": {"mode": DINOV3_RESIZE_MODE, "antialias": True},
            "normalization_mean": list(DINOV3_NORMALIZATION_MEAN),
            "normalization_std": list(DINOV3_NORMALIZATION_STD),
        },
        "vision_token_layout": {
            "patch_size": DINOV3_PATCH_SIZE,
            "feature_dim": DINOV3_FEATURE_DIM,
            "pool_hw": list(DINOV3_POOL_HW),
            "cache_tokens": DINOV3_CACHE_TOKENS,
            "token_order": "CLS_then_adaptive_avg_pool_row_major_height_width",
        },
        "architecture": {
            "history_model": "released_Chronos_Mamba",
            "embed_dim": 1024,
            "d_model": 1024,
            "num_blocks": 6,
            "visual_global_adapter_dim": 128,
            "visual_spatial_adapter_dim": 384,
            "proprio_adapter_dim": 512,
            "fusion_dim": 1024,
            "action_head": "released_IMLE_plus_symplectic_bridge",
        },
        "training_objective": {
            "imle_samples": 5,
            "imle_reduction": "nearest_mode_mse",
            "bridge_sigma_peak": 0.03,
            "force_loss_weight": 0.1,
        },
        "inference_solver": {
            "method": "semi_implicit_euler",
            "sample_steps": INFERENCE_SAMPLE_STEPS,
            "time_grid": "left_endpoint_i_over_steps",
        },
        "float32_numerics": dict(FLOAT32_NUMERICS),
        "normalization": {
            "type": "per_key_zscore",
            "fit_split": "train_only",
            "std_unbiased": False,
            "std_epsilon": 1e-6,
            "action_statistics": "independent_per_joint_per_horizon",
            "first_action_target": "next_recorded_drive_target_t_plus_1",
            "denormalize_before_temporal_aggregation": True,
        },
    }
