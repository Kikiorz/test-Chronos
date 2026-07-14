"""Focused tests for the official-compatible Chronos RGB data contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

try:
    from .M_dataset_robotwinRGB_E import (
        ACTION_TARGET_OFFSET,
        ACT_KEYS,
        OBS_KEYS,
        RGBTrajectoryDataset,
        make_ee_scaler,
        make_future_action_targets,
    )
    from .scaler_M import Scaler
except ImportError:  # direct focused-test execution without optional Mamba deps
    from M_dataset_robotwinRGB_E import (
        ACTION_TARGET_OFFSET,
        ACT_KEYS,
        OBS_KEYS,
        RGBTrajectoryDataset,
        make_ee_scaler,
        make_future_action_targets,
    )
    from scaler_M import Scaler


class FutureActionWindowTest(unittest.TestCase):
    def test_offset_zero_and_tail_clamp(self) -> None:
        trajectory = np.arange(4, dtype=np.float32)[:, None]
        actual = make_future_action_targets(trajectory, future_steps=3)
        expected = np.array(
            [
                [[0.0], [1.0], [2.0]],
                [[1.0], [2.0], [3.0]],
                [[2.0], [3.0], [3.0]],
                [[3.0], [3.0], [3.0]],
            ],
            dtype=np.float32,
        )
        self.assertEqual(ACTION_TARGET_OFFSET, 0)
        np.testing.assert_array_equal(actual, expected)


class OfficialScalerContractTest(unittest.TestCase):
    def test_default_eps_and_unbiased_standard_deviation(self) -> None:
        scaler = Scaler({"x": 1})
        scaler.fit({"x": torch.tensor([[1.0], [3.0], [5.0]])})

        self.assertEqual(scaler.eps, 1e-8)
        torch.testing.assert_close(scaler.mean_dict["x"], torch.tensor([3.0]))
        # torch.std defaults to Bessel-corrected (unbiased=True): std=2 here.
        torch.testing.assert_close(scaler.std_dict["x"], torch.tensor([2.0]))

    def test_each_action_horizon_has_independent_statistics(self) -> None:
        scaler = Scaler({"a": (3, 1)})
        values = torch.tensor(
            [
                [[0.0], [10.0], [100.0]],
                [[2.0], [14.0], [108.0]],
                [[4.0], [18.0], [116.0]],
            ]
        )
        scaler.fit({"a": values})

        self.assertEqual(tuple(scaler.mean_dict["a"].shape), (3, 1))
        torch.testing.assert_close(
            scaler.mean_dict["a"], torch.tensor([[2.0], [14.0], [108.0]])
        )
        torch.testing.assert_close(
            scaler.std_dict["a"], torch.tensor([[2.0], [4.0], [8.0]])
        )

    def test_ee_scaler_keeps_official_obs_and_horizon_shapes(self) -> None:
        scaler = make_ee_scaler(future_steps=16)
        self.assertEqual(list(scaler.lowdim_dict), OBS_KEYS + ACT_KEYS)
        self.assertTrue(all(tuple(scaler.mean_dict[key].shape) == (1,) for key in OBS_KEYS))
        self.assertTrue(
            all(tuple(scaler.mean_dict[key].shape) == (16, 1) for key in ACT_KEYS)
        )


class RMBenchRGBDatasetContractTest(unittest.TestCase):
    def test_dual_arm_ee_key_order(self) -> None:
        expected_per_arm = ["x", "y", "z", "qw", "qx", "qy", "qz"]
        expected = (
            [f"ee_l_{field}" for field in expected_per_arm]
            + ["gripper_l"]
            + [f"ee_r_{field}" for field in expected_per_arm]
            + ["gripper_r"]
        )
        self.assertEqual(OBS_KEYS, expected)
        self.assertEqual(ACT_KEYS, [f"{key}_act" for key in expected])

    @staticmethod
    def _write_episode(path: Path) -> np.ndarray:
        length = 3
        left_ee = np.stack(
            [np.arange(1, 8, dtype=np.float32) + 20 * t for t in range(length)]
        )
        left_gripper = np.arange(length, dtype=np.float32)[:, None] + 8
        right_ee = np.stack(
            [np.arange(9, 16, dtype=np.float32) + 20 * t for t in range(length)]
        )
        right_gripper = np.arange(length, dtype=np.float32)[:, None] + 16

        # RMBench passes simulator RGB values straight to OpenCV.  Encoding a
        # constant PNG makes channel preservation exact and still exercises
        # the same cv2.imdecode path as its fixed-width JPEG byte strings.
        frame = np.empty((2, 3, 3), dtype=np.uint8)
        frame[...] = np.array([7, 101, 233], dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            raise RuntimeError("cv2.imencode failed in test setup")
        encoded_bytes = encoded.tobytes()

        with h5py.File(path, "w") as root:
            root.create_dataset("endpose/left_endpose", data=left_ee)
            root.create_dataset("endpose/left_gripper", data=left_gripper)
            root.create_dataset("endpose/right_endpose", data=right_ee)
            root.create_dataset("endpose/right_gripper", data=right_gripper)
            root.create_dataset(
                "observation/head_camera/rgb",
                data=np.asarray([encoded_bytes] * length, dtype=f"S{len(encoded_bytes)}"),
            )

        return np.concatenate(
            [left_ee, left_gripper, right_ee, right_gripper], axis=-1
        ).astype(np.float32)

    def test_default_shape_order_uint8_and_rgb_resize(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = Path(tmpdir) / "train"
            train_dir.mkdir()
            expected_obs = self._write_episode(train_dir / "episode_0.hdf5")
            dataset = RGBTrajectoryDataset(train_dir, mode="train", scaler=None)
            sample = dataset[0]

        self.assertEqual(tuple(sample["obs"].shape), (3, 16))
        self.assertEqual(tuple(sample["actions"].shape), (3, 16, 16))
        self.assertEqual(tuple(sample["image"].shape), (3, 3, 480, 640))
        self.assertEqual(sample["image"].dtype, torch.uint8)
        self.assertEqual(sample["image"].device.type, "cpu")
        np.testing.assert_array_equal(sample["obs"].numpy(), expected_obs)

        # Horizon zero is the same timestep, while the final horizon clamps.
        np.testing.assert_array_equal(sample["actions"][0, 0].numpy(), expected_obs[0])
        np.testing.assert_array_equal(sample["actions"][-1].numpy(), np.tile(expected_obs[-1], (16, 1)))

        # No BGR<->RGB swap: CHW values equal the simulator's numeric channels.
        torch.testing.assert_close(
            sample["image"][0, :, 200, 300], torch.tensor([7, 101, 233], dtype=torch.uint8)
        )


if __name__ == "__main__":
    unittest.main()
