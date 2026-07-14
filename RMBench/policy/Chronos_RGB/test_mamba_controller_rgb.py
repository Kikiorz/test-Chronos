"""Small CPU tests for bounded temporal aggregation semantics."""

import math

import torch

from .mamba_controller_rgb import MambaRGBController


def test_temporal_ring_matches_released_old_to_new_weighting():
    controller = MambaRGBController.__new__(MambaRGBController)
    controller.future_steps = 3
    controller.action_dim = 1
    controller.temporal_decay = 0.01
    controller.device = torch.device("cpu")
    controller._action_ring = torch.empty(3, 3, 1)
    controller._ring_source_steps = [None, None, None]

    controller.t = 0
    first = controller._ensemble(torch.tensor([[1.0], [2.0], [3.0]]))
    torch.testing.assert_close(first, torch.tensor([1.0]))

    controller.t = 1
    second = controller._ensemble(torch.tensor([[10.0], [20.0], [30.0]]))
    decay = math.exp(-0.01)
    expected = torch.tensor([(2.0 + decay * 10.0) / (1.0 + decay)])
    torch.testing.assert_close(second, expected)
    assert controller._action_ring.numel() == 3 * 3 * 1
