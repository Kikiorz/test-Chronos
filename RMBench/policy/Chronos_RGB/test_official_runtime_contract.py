"""Lightweight regression tests for the official RGB runtime contract.

These tests intentionally avoid constructing the full ResNet/Mamba policy and
never request pretrained weights.  Run them with::

    python -m unittest RMBench.policy.Chronos_RGB.test_official_runtime_contract
"""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from .deploy_policy import encode_obs, get_model
from .mamba_controller_rgb import MambaRGBController
from .train_par_2D_IMLE_EE import LitMambaRGB, build_arg_parser


def _sim_observation(rgb: np.ndarray) -> dict[str, object]:
    """Build the smallest valid dual-arm RMBench observation."""

    return {
        "observation": {"head_camera": {"rgb": rgb}},
        "endpose": {
            "left_endpose": np.arange(7, dtype=np.float32),
            "left_gripper": np.float32(0.25),
            "right_endpose": np.arange(7, 14, dtype=np.float32),
            "right_gripper": np.float32(0.75),
        },
    }


class DeploymentImageContractTest(unittest.TestCase):
    def test_240x320_rgb_is_channel_preserved_and_resized_to_official_shape(self) -> None:
        rgb = np.empty((240, 320, 3), dtype=np.uint8)
        rgb[...] = np.array([0, 127, 255], dtype=np.uint8)

        encoded = encode_obs(_sim_observation(rgb))
        image = encoded["image"]

        self.assertEqual(image.shape, (1, 3, 480, 640))
        self.assertEqual(image.dtype, torch.float32)
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)
        torch.testing.assert_close(
            image[0, :, 239, 319],
            torch.tensor([0.0, 127.0 / 255.0, 1.0]),
            rtol=0,
            atol=1e-7,
        )
        self.assertEqual(encoded["qpos"].shape, (1, 16))
        self.assertEqual(encoded["qpos"].dtype, torch.float32)


class TrainerDefaultsContractTest(unittest.TestCase):
    def test_official_cli_defaults_are_pinned(self) -> None:
        args = build_arg_parser().parse_args(["--data-root", "/unused/rmbench/data"])

        expected = {
            "batch_size": 2,
            "accumulate_grad_batches": 3,
            "epochs": 600,
            "learning_rate": 1.7e-4,
            "weight_decay": 1e-4,
            "warmup_epochs": 15,
            "eta_min": 2e-5,
            "image_height": 480,
            "image_width": 640,
            "vision_chunk_size": 256,
            "periodic_every": 100,
            "action_target_offset": 0,
        }
        for name, value in expected.items():
            self.assertEqual(getattr(args, name), value, name)

    def test_optimizer_is_one_official_adamw_group_with_15_epoch_warmup(self) -> None:
        # Build only the Lightning shell.  A tiny policy is sufficient because
        # configure_optimizers depends on parameters and scalar configuration,
        # not the ResNet/Mamba forward graph.
        module = LitMambaRGB.__new__(LitMambaRGB)
        super(LitMambaRGB, module).__init__()
        module.policy = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
        module.policy[1].bias.requires_grad_(False)
        module.learning_rate = 1.7e-4
        module.weight_decay = 1e-4
        module.warmup_epochs = 15
        module.eta_min = 2e-5
        module._trainer = SimpleNamespace(max_epochs=600)

        configured = LitMambaRGB.configure_optimizers(module)
        optimizer = configured["optimizer"]
        scheduler_config = configured["lr_scheduler"]
        scheduler = scheduler_config["scheduler"]

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.defaults["lr"], 1.7e-4)
        self.assertEqual(optimizer.param_groups[0]["initial_lr"], 1.7e-4)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 1e-4)
        expected_ids = {
            id(parameter) for parameter in module.policy.parameters() if parameter.requires_grad
        }
        actual_ids = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
        self.assertEqual(actual_ids, expected_ids)

        self.assertEqual(scheduler_config["interval"], "epoch")
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
        self.assertEqual(scheduler._milestones, [15])
        warmup, cosine = scheduler._schedulers
        self.assertIsInstance(warmup, torch.optim.lr_scheduler.LinearLR)
        self.assertEqual(warmup.start_factor, 0.01)
        self.assertEqual(warmup.end_factor, 1.0)
        self.assertEqual(warmup.total_iters, 15)
        self.assertIsInstance(cosine, torch.optim.lr_scheduler.CosineAnnealingLR)
        self.assertEqual(cosine.T_max, 585)
        self.assertEqual(cosine.eta_min, 2e-5)


class ControllerContractTest(unittest.TestCase):
    def test_deploy_defaults_forward_fixed_official_architecture_and_offset(self) -> None:
        captured: dict[str, object] = {}

        class CaptureController:
            def __init__(self, args):
                captured.update(args)

        # Patch the deployment adapter only; no checkpoint or network access is
        # needed to verify its immutable controller contract.
        import RMBench.policy.Chronos_RGB.deploy_policy as deploy_module

        original = deploy_module.MambaRGBController
        deploy_module.MambaRGBController = CaptureController
        try:
            get_model({"device": "cpu", "ckpt_path": "unused", "scaler_path": "unused"})
        finally:
            deploy_module.MambaRGBController = original

        self.assertEqual(captured["visual_architecture"], "official_realworld")
        self.assertEqual(captured["execution_horizon_offset"], 0)

    def test_controller_rejects_nonofficial_architecture_before_checkpoint_io(self) -> None:
        with self.assertRaisesRegex(ValueError, "official_realworld"):
            MambaRGBController(
                {
                    "device": "cpu",
                    "ckpt_path": "unused",
                    "scaler_path": "unused",
                    "visual_architecture": "v2",
                }
            )

    def test_controller_rejects_nonzero_execution_offset_before_checkpoint_io(self) -> None:
        with self.assertRaisesRegex(ValueError, "execution_horizon_offset=0"):
            MambaRGBController(
                {
                    "device": "cpu",
                    "ckpt_path": "unused",
                    "scaler_path": "unused",
                    "execution_horizon_offset": 1,
                }
            )


if __name__ == "__main__":
    unittest.main()
