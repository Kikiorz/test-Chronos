"""Focused contract tests for the Chronos_RGB V2 changes.

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
from .train_par_2D_IMLE_EE import (
    load_compatible_policy_state,
    make_optimizer_parameter_groups,
)


class FutureTargetContractTest(unittest.TestCase):
    def test_next_state_offset_and_tail_clamp(self) -> None:
        trajectory = np.arange(4, dtype=np.float32)[:, None]
        actual = make_future_action_targets(
            trajectory, future_steps=3, target_offset=ACTION_TARGET_OFFSET
        )
        expected = np.array(
            [
                [[1.0], [2.0], [3.0]],
                [[2.0], [3.0], [3.0]],
                [[3.0], [3.0], [3.0]],
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
        self.fusion = ImageMambaFusion(pretrained=False, backbone_trainable="layer4")

    def test_v2_adapter_preserves_eight_by_ten_budget(self) -> None:
        self.assertEqual(MambaConfig().visual_architecture, "v2")
        second_conv = self.fusion.visual_adapter[3]
        self.assertEqual(second_conv.out_channels, 64)
        self.assertEqual(second_conv.stride, (1, 1))
        self.assertEqual(self.fusion.visual_adapter[6].output_size, (8, 10))
        self.assertEqual(self.fusion.visual_adapter[8].in_features, 64 * 8 * 10)
        self.assertEqual(self.fusion.visual_adapter[8].out_features, 256)
        self.assertEqual(self.fusion.visual_adapter[11].out_features, 512)

        v2_parameters = sum(parameter.numel() for parameter in self.fusion.visual_adapter.parameters())
        v1_parameters = 2_787_968
        self.assertLess(abs(v2_parameters - v1_parameters) / v1_parameters, 0.02)

    def test_visual_architecture_strict_load_matrix(self) -> None:
        v1_source = ImageMambaFusion(
            pretrained=False, backbone_trainable="none", visual_architecture="v1"
        )
        with self.assertRaises(RuntimeError):
            self.fusion.load_state_dict(v1_source.state_dict(), strict=True)

        explicit_v1_target = ImageMambaFusion(
            pretrained=False, backbone_trainable="none", visual_architecture="v1"
        )
        explicit_v1_target.load_state_dict(v1_source.state_dict(), strict=True)

        v2_source = ImageMambaFusion(
            pretrained=False, backbone_trainable="none", visual_architecture="v2"
        )
        v2_target = ImageMambaFusion(
            pretrained=False, backbone_trainable="none", visual_architecture="v2"
        )
        v2_target.load_state_dict(v2_source.state_dict(), strict=True)

    def test_only_layer4_non_bn_parameters_train(self) -> None:
        for stage in list(self.fusion.vision_backbone.children())[:-1]:
            self.assertTrue(all(not parameter.requires_grad for parameter in stage.parameters()))

        trainable_layer4 = [
            parameter
            for name, parameter in self.fusion.vision_backbone[7].named_parameters()
            if "bn" not in name and "downsample.1" not in name
        ]
        self.assertTrue(trainable_layer4)
        self.assertTrue(all(parameter.requires_grad for parameter in trainable_layer4))

        self.fusion.train()
        batch_norms = [
            module
            for module in self.fusion.vision_backbone.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]
        self.assertTrue(batch_norms)
        self.assertTrue(all(not module.training for module in batch_norms))
        self.assertTrue(
            all(not parameter.requires_grad for module in batch_norms for parameter in module.parameters())
        )

    def test_optimizer_groups_are_complete_and_disjoint(self) -> None:
        class TinyPolicy(nn.Module):
            def __init__(self, fusion: ImageMambaFusion):
                super().__init__()
                self.fusion_engine = fusion
                self.temporal = nn.Linear(4, 4)

        policy = TinyPolicy(self.fusion)
        groups = make_optimizer_parameter_groups(policy, 1e-4, 1e-5, 3e-5)
        self.assertEqual([group["name"] for group in groups], [
            "visual_head", "resnet_layer4", "chronos"
        ])
        self.assertEqual([group["lr"] for group in groups], [1e-4, 1e-5, 3e-5])
        grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]
        expected_ids = [id(parameter) for parameter in policy.parameters() if parameter.requires_grad]
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
        self.assertEqual(set(grouped_ids), set(expected_ids))


class WarmStartContractTest(unittest.TestCase):
    @staticmethod
    def _policy() -> nn.Module:
        class Fusion(nn.Module):
            def __init__(self):
                super().__init__()
                layers: list[nn.Module] = [nn.Identity() for _ in range(13)]
                layers[3] = nn.Linear(2, 2)
                layers[11] = nn.Linear(2, 2)
                layers[12] = nn.LayerNorm(2)
                self.visual_adapter = nn.Sequential(*layers)

        class Policy(nn.Module):
            def __init__(self):
                super().__init__()
                self.fusion_engine = Fusion()
                self.temporal = nn.Linear(2, 2)

        return Policy()

    def test_loads_all_compatible_and_only_skips_changed_visual_head(self) -> None:
        policy = self._policy()
        source = {key: value.clone() for key, value in policy.state_dict().items()}
        source.pop("fusion_engine.visual_adapter.11.weight")
        source.pop("fusion_engine.visual_adapter.11.bias")
        source.pop("fusion_engine.visual_adapter.12.weight")
        source.pop("fusion_engine.visual_adapter.12.bias")
        source["fusion_engine.visual_adapter.3.weight"] = torch.zeros(3, 2)
        source["fusion_engine.visual_adapter.3.bias"] = torch.zeros(3)
        source["temporal.weight"].fill_(4.0)

        report = load_compatible_policy_state(policy, source)
        self.assertIn("fusion_engine.visual_adapter.3.weight", report["skipped_shape"])
        self.assertEqual(set(report["initialized_v2"]), {
            "fusion_engine.visual_adapter.11.weight",
            "fusion_engine.visual_adapter.11.bias",
            "fusion_engine.visual_adapter.12.weight",
            "fusion_engine.visual_adapter.12.bias",
        })
        torch.testing.assert_close(policy.temporal.weight, torch.full_like(policy.temporal.weight, 4.0))

    def test_rejects_shape_mismatch_outside_visual_head(self) -> None:
        policy = self._policy()
        source = {key: value.clone() for key, value in policy.state_dict().items()}
        source["temporal.weight"] = torch.zeros(3, 2)
        with self.assertRaisesRegex(RuntimeError, "outside the expected V1-to-V2 visual changes"):
            load_compatible_policy_state(policy, source)


if __name__ == "__main__":
    unittest.main()
