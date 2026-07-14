"""Small CPU tests for bounded temporal aggregation semantics."""

import math

import torch

from .mamba_controller_rgb_joint import ACTION_KEYS, JOINT_KEYS, MambaRGBJointController
from .scaler_M import Scaler


def test_temporal_ring_matches_released_old_to_new_weighting():
    controller = MambaRGBJointController.__new__(MambaRGBJointController)
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


def test_joint_safety_uses_physical_training_bounds_and_gripper_contract():
    controller = MambaRGBJointController.__new__(MambaRGBJointController)
    controller.clip_to_training_range = True
    controller.training_range_margin = 0.0
    shapes = {key: 1 for key in JOINT_KEYS}
    shapes.update({key: (16, 1) for key in ACTION_KEYS})
    controller.scaler = Scaler(shapes)
    with torch.no_grad():
        for key in ACTION_KEYS:
            controller.scaler.min_dict[key].fill_(-1.0)
            controller.scaler.max_dict[key].fill_(1.0)
    action = torch.linspace(-2.0, 2.0, 14)
    safe = controller._apply_joint_safety(action)
    assert torch.all(safe >= -1.0) and torch.all(safe <= 1.0)
    assert 0.0 <= safe[6] <= 1.0
    assert 0.0 <= safe[13] <= 1.0

    action[0] = float("inf")
    try:
        controller._apply_joint_safety(action)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("Infinite joint output was silently clipped to a boundary")
