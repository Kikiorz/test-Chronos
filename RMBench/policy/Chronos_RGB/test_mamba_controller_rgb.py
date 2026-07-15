"""Small CPU tests for bounded temporal aggregation semantics."""

import math

import torch

from .mamba_controller_rgb import MambaRGBController


def test_temporal_ring_matches_released_old_to_new_weighting():
    controller = MambaRGBController.__new__(MambaRGBController)
    controller.future_steps = 3
    controller.action_dim = 1
    controller.temporal_decay = 0.01
    controller.execution_horizon_offset = 0
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


def test_temporal_ring_can_execute_the_next_predicted_state():
    controller = MambaRGBController.__new__(MambaRGBController)
    controller.future_steps = 3
    controller.action_dim = 1
    controller.temporal_decay = 0.01
    controller.execution_horizon_offset = 1
    controller.device = torch.device("cpu")
    controller._action_ring = torch.empty(3, 3, 1)
    controller._ring_source_steps = [None, None, None]

    controller.t = 0
    first = controller._ensemble(torch.tensor([[1.0], [2.0], [3.0]]))
    torch.testing.assert_close(first, torch.tensor([2.0]))

    controller.t = 1
    second = controller._ensemble(torch.tensor([[10.0], [20.0], [30.0]]))
    decay = math.exp(-0.01)
    expected = torch.tensor([(3.0 + decay * 20.0) / (1.0 + decay)])
    torch.testing.assert_close(second, expected)


def test_ensembled_action_is_denormalized_with_horizon_zero_statistics():
    controller = MambaRGBController.__new__(MambaRGBController)
    controller.future_steps = 3
    controller.action_dim = 2

    captured = {}

    def fake_denormalize(sequence):
        captured["sequence"] = sequence.clone()
        horizon_bias = torch.tensor([[[100.0, 200.0], [300.0, 400.0], [500.0, 600.0]]])
        return sequence + horizon_bias

    controller._denormalize_sequence = fake_denormalize
    result = controller._denormalize_horizon_zero(torch.tensor([1.0, 2.0]))

    expected_input = torch.tensor([[[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]]])
    torch.testing.assert_close(captured["sequence"], expected_input)
    torch.testing.assert_close(result, torch.tensor([101.0, 202.0]))


def test_get_action_ensembles_normalized_predictions_before_denormalizing():
    controller = MambaRGBController.__new__(MambaRGBController)
    controller.device = torch.device("cpu")
    controller.future_steps = 3
    controller.action_dim = 2
    controller.sample_steps = 5
    controller.temporal_agg = True
    controller.execution_horizon_offset = 0
    controller.hiddens = "hidden-before"
    controller.t = 0
    controller._normalize_qpos = lambda qpos: qpos

    class FakePolicy:
        @staticmethod
        def fusion_engine(image, qpos):
            return torch.zeros(1, 4)

        @staticmethod
        def step(fused, hiddens, sample_steps):
            assert hiddens == "hidden-before"
            assert sample_steps == 5
            prediction = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
            return prediction, "hidden-after"

    controller.policy = FakePolicy()
    captured = {}

    def fake_ensemble(sequence):
        captured["ensemble_input"] = sequence.clone()
        return torch.tensor([7.0, 8.0])

    def fake_denormalize(action):
        captured["denormalize_input"] = action.clone()
        return action + 100.0

    controller._ensemble = fake_ensemble
    controller._denormalize_horizon_zero = fake_denormalize

    result = controller.get_action(
        {"image": torch.zeros(1, 3, 2, 2), "qpos": torch.zeros(1, 2)}
    )

    torch.testing.assert_close(
        captured["ensemble_input"],
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )
    torch.testing.assert_close(captured["denormalize_input"], torch.tensor([7.0, 8.0]))
    torch.testing.assert_close(torch.from_numpy(result), torch.tensor([107.0, 108.0]))
    assert controller.hiddens == "hidden-after"
    assert controller.t == 1


def test_temporal_ring_matches_reference_after_wraparound():
    controller = MambaRGBController.__new__(MambaRGBController)
    controller.future_steps = 3
    controller.action_dim = 1
    controller.temporal_decay = 0.01
    controller.execution_horizon_offset = 0
    controller.device = torch.device("cpu")
    controller._action_ring = torch.empty(3, 3, 1)
    controller._ring_source_steps = [None, None, None]

    history = []
    for step in range(10):
        sequence = torch.tensor(
            [[100.0 * step + horizon] for horizon in range(controller.future_steps)]
        )
        history.append(sequence)
        controller.t = step
        actual = controller._ensemble(sequence)

        candidates = []
        for source in range(max(0, step - controller.future_steps + 1), step + 1):
            candidates.append(history[source][step - source])
        stacked = torch.stack(candidates)
        indices = torch.arange(len(candidates), dtype=stacked.dtype)
        weights = torch.exp(-controller.temporal_decay * indices)
        expected = (stacked * (weights / weights.sum()).unsqueeze(-1)).sum(dim=0)
        torch.testing.assert_close(actual, expected)


def test_reset_clears_recurrent_latent_and_temporal_state():
    controller = MambaRGBController.__new__(MambaRGBController)
    controller.future_steps = 3
    controller.action_dim = 2
    controller.device = torch.device("cpu")
    controller._action_ring = torch.ones(3, 3, 2)
    controller._ring_source_steps = [7, 8, 9]
    controller.t = 10

    class FakePolicy:
        _current_z = torch.ones(1)

        @staticmethod
        def init_hidden_states(batch_size, device):
            assert batch_size == 1
            assert device == torch.device("cpu")
            return "fresh-hidden"

    controller.policy = FakePolicy()
    controller.reset()

    assert controller.hiddens == "fresh-hidden"
    assert controller.t == 0
    assert controller._ring_source_steps == [None, None, None]
    assert torch.count_nonzero(controller._action_ring) == 0
    assert controller.policy._current_z is None
