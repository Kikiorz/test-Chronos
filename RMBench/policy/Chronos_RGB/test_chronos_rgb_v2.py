"""Focused contracts for the official real-world Chronos RGB encoder.

Run without pytest:
    python -m unittest RMBench.policy.Chronos_RGB.test_chronos_rgb_v2
"""

from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from .M_dataset_robotwinRGB_E import ACTION_TARGET_OFFSET, make_future_action_targets
from .mamba_policy_par_2D_IMLE_EE import ImageMambaFusion, MambaConfig


class FutureTargetContractTest(unittest.TestCase):
    def test_same_step_offset_and_tail_clamp(self) -> None:
        trajectory = np.arange(4, dtype=np.float32)[:, None]
        actual = make_future_action_targets(
            trajectory, future_steps=3, target_offset=ACTION_TARGET_OFFSET
        )
        expected = np.array(
            [
                [[0.0], [1.0], [2.0]],
                [[1.0], [2.0], [3.0]],
                [[2.0], [3.0], [3.0]],
                [[3.0], [3.0], [3.0]],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_single_frame_episode_clamps_every_horizon(self) -> None:
        trajectory = np.array([[7.0, 8.0]], dtype=np.float64)
        actual = make_future_action_targets(trajectory, future_steps=4)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(actual, np.tile([[7.0, 8.0]], (1, 4, 1)))

    def test_invalid_target_arguments_fail(self) -> None:
        with self.assertRaises(ValueError):
            make_future_action_targets(np.empty((0, 16)), future_steps=16)
        with self.assertRaises(ValueError):
            make_future_action_targets(np.ones((2, 16)), future_steps=0)
        with self.assertRaises(ValueError):
            make_future_action_targets(np.ones((2, 16)), target_offset=-1)


class VisualBackboneContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fusion = ImageMambaFusion(pretrained=False)

    def test_config_selects_only_the_official_frozen_encoder(self) -> None:
        config = MambaConfig()
        self.assertEqual(config.visual_architecture, "official_realworld")
        self.assertEqual(config.backbone_trainable, "none")
        self.assertEqual(config.lowdim_dim, 16)
        self.assertEqual(config.action_dim, 16)
        self.assertEqual(config.future_steps, 16)

    def test_recursive_bn_to_gn_and_whole_trunk_is_frozen_eval(self) -> None:
        batch_norms = [
            module
            for module in self.fusion.vision_net.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]
        group_norms = [
            module for module in self.fusion.vision_net.modules() if isinstance(module, nn.GroupNorm)
        ]
        self.assertEqual(batch_norms, [])
        self.assertTrue(group_norms)
        self.assertTrue(all(module.num_groups == 32 for module in group_norms))
        self.assertTrue(all(not parameter.requires_grad for parameter in self.fusion.vision_net.parameters()))

        self.fusion.train()
        self.assertTrue(self.fusion.training)
        self.assertFalse(self.fusion.vision_net.training)
        self.assertTrue(self.fusion.visual_adapter.training)
        self.assertTrue(all(not module.training for module in self.fusion.vision_net.modules()))

    def test_legacy_arguments_cannot_select_v1_v2_or_unfreeze_layer4(self) -> None:
        fusion = ImageMambaFusion(
            pretrained=False,
            backbone_trainable="all",
            freeze_backbone=False,
            visual_architecture="v1",
            spatial_pool=(4, 5),
        )
        self.assertEqual(fusion.visual_architecture, "official_realworld")
        self.assertEqual(fusion.backbone_trainable, "none")
        self.assertTrue(fusion.freeze_backbone)
        self.assertTrue(all(not parameter.requires_grad for parameter in fusion.vision_net.parameters()))

    def test_official_adapter_proprio_and_fusion_parameter_shapes(self) -> None:
        adapter = self.fusion.visual_adapter
        self.assertEqual(adapter[0].weight.shape, (256, 512, 3, 3))
        self.assertEqual(adapter[1].num_groups, 32)
        self.assertEqual(adapter[3].weight.shape, (128, 256, 3, 3))
        self.assertEqual(adapter[3].stride, (2, 2))
        self.assertEqual(adapter[4].num_groups, 16)
        self.assertIsInstance(adapter[6], nn.Flatten)
        self.assertEqual(adapter[7].weight.shape, (1024, 128 * 8 * 10))
        self.assertEqual(adapter[8].normalized_shape, (1024,))
        self.assertIsInstance(adapter[9], nn.SiLU)
        self.assertEqual(adapter[10].p, 0.10)

        proprio = self.fusion.proprio_projector
        self.assertEqual(proprio[0].weight.shape, (128, 16))
        self.assertEqual(proprio[2].weight.shape, (512, 128))
        self.assertEqual(proprio[3].normalized_shape, (512,))
        self.assertEqual(self.fusion.fusion_proj[0].weight.shape, (1024, 1536))
        self.assertEqual(self.fusion.fusion_proj[1].normalized_shape, (1024,))

    def test_uint8_and_unit_float_normalization_are_identical(self) -> None:
        raw = torch.randint(0, 256, (1, 3, 480, 640), dtype=torch.uint8)
        unit_float = raw.float() / 255.0
        actual_uint8 = self.fusion._prepare_image(raw)
        actual_float = self.fusion._prepare_image(unit_float)
        manual = (unit_float - self.fusion.image_mean) / self.fusion.image_std
        torch.testing.assert_close(actual_uint8, actual_float, rtol=0, atol=0)
        torch.testing.assert_close(actual_uint8, manual, rtol=0, atol=0)

    def test_strict_image_shape_and_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "480x640"):
            self.fusion._prepare_image(torch.zeros(1, 3, 240, 320))
        with self.assertRaisesRegex(ValueError, "480x640"):
            self.fusion._prepare_image(torch.zeros(1, 480, 640, 3))
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            self.fusion._prepare_image(torch.full((1, 3, 480, 640), 2.0))

    def test_forward_uses_15x20_to_8x10_and_returns_1024(self) -> None:
        trunk_shapes: list[tuple[int, ...]] = []
        adapter_shapes: list[tuple[int, ...]] = []
        trunk_hook = self.fusion.vision_net.register_forward_hook(
            lambda _module, _inputs, output: trunk_shapes.append(tuple(output.shape))
        )
        adapter_hook = self.fusion.visual_adapter[3].register_forward_hook(
            lambda _module, _inputs, output: adapter_shapes.append(tuple(output.shape))
        )
        try:
            self.fusion.eval()
            output = self.fusion(
                torch.zeros(1, 3, 480, 640, dtype=torch.uint8),
                torch.zeros(1, 16),
            )
        finally:
            trunk_hook.remove()
            adapter_hook.remove()
        self.assertEqual(trunk_shapes, [(1, 512, 15, 20)])
        self.assertEqual(adapter_shapes, [(1, 128, 8, 10)])
        self.assertEqual(output.shape, (1, 1024))


if __name__ == "__main__":
    unittest.main()
