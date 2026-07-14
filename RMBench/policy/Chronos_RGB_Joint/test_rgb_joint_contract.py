"""CPU contract tests for RMBench 14-D joint layout and normalization."""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

from .M_dataset_robotwinRGB_J import (
    ACTION_KEYS,
    JOINT_KEYS,
    RGBJointTrajectoryDataset,
    make_joint_scaler,
)
from .deploy_policy import _extract_head_rgb, encode_obs, eval as deploy_eval
from .contracts import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES
from .scaler_M import Scaler


def _write_joint_components(root: h5py.File, vector: np.ndarray) -> None:
    group = root.create_group("joint_action")
    group.create_dataset("left_arm", data=vector[:, :6])
    group.create_dataset("left_gripper", data=vector[:, 6])
    group.create_dataset("right_arm", data=vector[:, 7:13])
    group.create_dataset("right_gripper", data=vector[:, 13])
    group.create_dataset("vector", data=vector)


def test_hdf5_layout_and_future_padding_are_exact() -> None:
    vector = np.arange(5 * 14, dtype=np.float32).reshape(5, 14)
    vector[:, 6] = np.linspace(0, 1, 5)
    vector[:, 13] = np.linspace(1, 0, 5)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "episode0.hdf5"
        with h5py.File(path, "w") as root:
            _write_joint_components(root, vector)
        with h5py.File(path, "r") as root:
            loaded = RGBJointTrajectoryDataset._load_lowdim(root)
        np.testing.assert_array_equal(loaded, vector)
        dataset = RGBJointTrajectoryDataset.__new__(RGBJointTrajectoryDataset)
        dataset.future_steps = 16
        future = dataset._make_future_actions(loaded)
        expected_next = np.concatenate([vector[1:], vector[-1:]], axis=0)
        np.testing.assert_array_equal(future[:, 0], expected_next)
        np.testing.assert_array_equal(future[-1], np.repeat(vector[-1][None], 16, axis=0))
        # Every ideal temporal-ensemble candidate available at rollout step t
        # must name the same next target, regardless of its source timestep.
        for timestep in range(len(vector)):
            candidates = [
                future[source, timestep - source]
                for source in range(max(0, timestep - 15), timestep + 1)
            ]
            expected = vector[min(timestep + 1, len(vector) - 1)]
            np.testing.assert_array_equal(
                np.stack(candidates), np.repeat(expected[None], len(candidates), axis=0)
            )


def test_joint_scaler_roundtrip_and_per_horizon_statistics() -> None:
    scaler = make_joint_scaler(16)
    observations = torch.arange(10 * 14, dtype=torch.float32).reshape(10, 14) / 10
    actions = torch.stack([observations + horizon for horizon in range(16)], dim=1)
    data = {key: observations[:, index : index + 1] for index, key in enumerate(JOINT_KEYS)}
    data.update(
        {
            key: actions[..., index : index + 1]
            for index, key in enumerate(ACTION_KEYS)
        }
    )
    scaler.fit(data)
    # The released scaler uses the sample standard deviation (Bessel corrected).
    torch.testing.assert_close(
        scaler.std_dict[JOINT_KEYS[0]],
        observations[:, :1].std(dim=0, unbiased=True),
        rtol=0,
        atol=0,
    )
    normalized = scaler.normalize(data)
    restored = scaler.denormalize(normalized)
    for key in data:
        torch.testing.assert_close(restored[key], data[key], rtol=0, atol=2e-6)
    assert scaler.mean_dict[ACTION_KEYS[0]].shape == (16, 1)
    assert not torch.equal(
        scaler.mean_dict[ACTION_KEYS[0]][0], scaler.mean_dict[ACTION_KEYS[0]][-1]
    )

    # A transposed/missing horizon must never be accepted through broadcasting.
    try:
        scaler.denormalize({ACTION_KEYS[0]: torch.zeros(2, 1, 16)})
    except ValueError as exc:
        assert "must end in (16, 1)" in str(exc)
    else:
        raise AssertionError("A misordered action horizon was silently broadcast")

    try:
        scaler.normalize({JOINT_KEYS[0]: torch.zeros(2, 1, dtype=torch.float64)})
    except TypeError as exc:
        assert "torch.float32" in str(exc)
    else:
        raise AssertionError("A float64 joint input silently changed normalization precision")

    try:
        scaler.normalize({"misspelled_joint": torch.zeros(2, 1)})
    except KeyError as exc:
        assert "unknown key" in str(exc)
    else:
        raise AssertionError("An unknown scaler key was silently passed through")


def test_scaler_fingerprint_and_dark_uint8_rgb_are_unambiguous() -> None:
    scaler = make_joint_scaler(16)
    observations = torch.arange(20 * 14, dtype=torch.float32).reshape(20, 14)
    actions = torch.stack([observations + horizon for horizon in range(16)], dim=1)
    data = {key: observations[:, index:index + 1] for index, key in enumerate(JOINT_KEYS)}
    data.update(
        {key: actions[..., index:index + 1] for index, key in enumerate(ACTION_KEYS)}
    )
    scaler.fit(data)
    fingerprint = scaler.fingerprint()
    changed = {key: value.clone() for key, value in scaler.state_dict().items()}
    first_key = next(iter(changed))
    changed[first_key].view(-1)[0] += 1.0
    assert len(fingerprint) == 64
    assert Scaler.state_fingerprint(changed) != fingerprint

    observation = {
        "observation": {
            "head_camera": {"rgb": np.ones((240, 320, 3), dtype=np.uint8)}
        }
    }
    image = _extract_head_rgb(observation, "head_camera")
    assert image.dtype == np.uint8
    np.testing.assert_array_equal(image, np.ones((240, 320, 3), dtype=np.uint8))

    ambiguous = {
        "observation": {
            "head_camera": {
                "rgb": np.full((240, 320, 3), 1.25, dtype=np.float32)
            }
        }
    }
    try:
        _extract_head_rgb(ambiguous, "head_camera")
    except TypeError as exc:
        assert "uint8" in str(exc)
    else:
        raise AssertionError("Ambiguous float RGB scale was silently accepted")


def test_online_observation_order_and_qpos_execution() -> None:
    vector = np.arange(14, dtype=np.float32)
    vector[6] = 0.25
    vector[13] = 0.75
    observation = {
        "observation": {"head_camera": {"rgb": np.zeros((240, 320, 3), dtype=np.uint8)}},
        "joint_action": {"vector": vector.copy()},
    }
    encoded = encode_obs(observation)
    torch.testing.assert_close(encoded["qpos"], torch.from_numpy(vector).unsqueeze(0))
    assert encoded["image"].shape == (1, 3, 240, 320)
    assert encoded["image"].dtype == torch.uint8

    class FakeModel:
        camera_name = "head_camera"

        def get_action(self, obs):
            return np.asarray(obs["qpos"][0], dtype=np.float32)

    class FakeEnvironment:
        def __init__(self):
            self.received = None
            self.robot = type(
                "FakeRobot",
                (),
                {
                    "left_arm_joints_name": list(LEFT_ARM_JOINT_NAMES),
                    "right_arm_joints_name": list(RIGHT_ARM_JOINT_NAMES),
                },
            )()

        def take_action(self, action, action_type):
            self.received = (np.asarray(action), action_type)

    environment = FakeEnvironment()
    deploy_eval(environment, FakeModel(), observation)
    np.testing.assert_array_equal(environment.received[0], vector)
    assert environment.received[1] == "qpos"

    environment.robot.left_arm_joints_name[0] = "piper_joint1"
    try:
        deploy_eval(environment, FakeModel(), observation)
    except RuntimeError as exc:
        assert "aloha-agilex" in str(exc)
    else:
        raise AssertionError("A same-dimensional wrong robot embodiment was accepted")
