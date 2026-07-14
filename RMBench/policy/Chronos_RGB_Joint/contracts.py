"""Shared immutable contracts for RGB-Joint training, resume, and deployment."""

from __future__ import annotations

try:
    from .resnet18_backbone import (
        FLOAT32_NUMERICS,
        RESNET18_IMAGE_HW,
        RESNET18_MODEL_NAME,
        RESNET18_NORMALIZATION_MEAN,
        RESNET18_NORMALIZATION_STD,
        RESNET18_RESIZE_MODE,
        RESNET18_WEIGHTS_SHA256,
        RMBENCH_RGB_HW,
    )
except ImportError:  # direct script execution
    from resnet18_backbone import (  # type: ignore
        FLOAT32_NUMERICS,
        RESNET18_IMAGE_HW,
        RESNET18_MODEL_NAME,
        RESNET18_NORMALIZATION_MEAN,
        RESNET18_NORMALIZATION_STD,
        RESNET18_RESIZE_MODE,
        RESNET18_WEIGHTS_SHA256,
        RMBENCH_RGB_HW,
    )


POLICY_VARIANT = "chronos_rgb_joint14_resnet18"
POLICY_CONTRACT_VERSION = 6
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
    """Return the immutable, JSON-serializable policy contract.

    The modality-specific path mirrors the released real-world image policy;
    the action semantics and temporal objective mirror the released RMBench
    policy because the demonstrations and evaluator are RMBench joint data.
    """

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
        "backbone": RESNET18_MODEL_NAME,
        "backbone_reference_weights_sha256": RESNET18_WEIGHTS_SHA256,
        "source_image_hw": list(RMBENCH_RGB_HW),
        "image_hw": list(RESNET18_IMAGE_HW),
        "image_preprocess": {
            "channel_order": "RGB",
            "input_dtype": "uint8",
            "input_scale": "divide_by_255_float32",
            "resize": {
                "library": "opencv",
                "mode": RESNET18_RESIZE_MODE,
                "output_wh": [RESNET18_IMAGE_HW[1], RESNET18_IMAGE_HW[0]],
            },
            "normalization_mean": list(RESNET18_NORMALIZATION_MEAN),
            "normalization_std": list(RESNET18_NORMALIZATION_STD),
            "output_layout": "CHW_float32",
        },
        "architecture": {
            "history_model": "released_Chronos_Mamba",
            "embed_dim": 1024,
            "d_model": 1024,
            "num_blocks": 6,
            "vision_trunk": (
                "torchvision_resnet18_without_avgpool_fc_then_all_batchnorm2d_"
                "replaced_by_fresh_groupnorm32"
            ),
            "vision_trunk_output": [512, 15, 20],
            "vision_trunk_frozen": True,
            "visual_adapter": (
                "conv3x3_512_256_gn32_silu_conv3x3s2_256_128_gn16_silu_"
                "flatten_linear10240_1024_ln_silu_dropout0.10"
            ),
            "proprio_adapter": "linear14_128_relu_linear128_512_ln",
            "fusion": "linear1536_1024_ln",
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
            "std_unbiased": True,
            "std_epsilon": 1e-8,
            "action_statistics": "independent_per_joint_per_horizon",
            "first_action_target": "next_recorded_drive_target_t_plus_1",
            "denormalize_before_temporal_aggregation": True,
        },
    }
