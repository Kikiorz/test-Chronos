"""Checkpoint/resume guards that prevent silent data or scaler mixing."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

from .M_dataset_robotwinRGB_J import ACTION_KEYS, JOINT_KEYS, make_joint_scaler
from .contracts import base_policy_contract
from .ema_callback import WarmupPolicyEMACallback
from .train_par_2D_IMLE_Joint import (
    _configure_large_temp_dir,
    _validate_resume_checkpoint,
)


def test_large_temp_dir_is_atomic_and_af_unix_safe() -> None:
    old_tempdir = tempfile.tempdir
    old_tmpdir = os.environ.get("TMPDIR")
    try:
        # A short system-temp sandbox makes this test independent of the
        # repository's path depth; source and destination share one device.
        with tempfile.TemporaryDirectory(prefix="cjt-") as directory:
            root = Path(directory)
            output_dir = root / "out"
            output_dir.mkdir()
            selected = _configure_large_temp_dir(output_dir, root / "t")
            assert selected.stat().st_dev == output_dir.stat().st_dev
            assert os.environ["TMPDIR"] == str(selected)
            assert tempfile.tempdir == str(selected)
            probe = selected / "pymp-00000000" / "listener-00000000"
            assert len(os.fsencode(probe)) + 1 <= 108
    finally:
        tempfile.tempdir = old_tempdir
        if old_tmpdir is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = old_tmpdir


def test_resume_contract_binds_embedded_scaler_and_split() -> None:
    scaler = make_joint_scaler(16)
    observations = torch.arange(20 * 14, dtype=torch.float32).reshape(20, 14)
    actions = torch.stack([observations + horizon for horizon in range(16)], dim=1)
    data = {key: observations[:, index:index + 1] for index, key in enumerate(JOINT_KEYS)}
    data.update(
        {key: actions[..., index:index + 1] for index, key in enumerate(ACTION_KEYS)}
    )
    scaler.fit(data)
    run_contract = {
        "backbone_weights_sha256": "a" * 64,
        "scaler_fingerprint": scaler.fingerprint(),
        "cache_contract_sha256": "b" * 64,
        "cache_dataset_sha256": "c" * 64,
        "split_seed": 42,
        "val_fraction": 0.1,
        "train_episodes": ["episode0.hdf5"],
        "val_episodes": ["episode1.hdf5"],
    }
    contract = {**base_policy_contract(), **run_contract}
    state = {f"scaler.{key}": value for key, value in scaler.state_dict().items()}
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "resume.ckpt"
        torch.save({"policy_contract": contract, "state_dict": state}, checkpoint)
        _validate_resume_checkpoint(str(checkpoint), contract)

        changed = dict(contract)
        changed["split_seed"] = 43
        try:
            _validate_resume_checkpoint(str(checkpoint), changed)
        except RuntimeError:
            pass
        else:  # pragma: no cover - explicit failure message
            raise AssertionError("A changed split contract was not rejected")


def test_ema_reuses_resident_storage_without_changing_update_math() -> None:
    class Holder:
        def __init__(self) -> None:
            self.policy = torch.nn.Linear(3, 2)

    holder = Holder()
    callback = WarmupPolicyEMACallback()
    callback._ensure_ema_on_policy_device(holder)  # type: ignore[arg-type]
    assert callback.ema_state is not None
    pointers = {key: value.data_ptr() for key, value in callback.ema_state.items()}
    callback._ensure_ema_on_policy_device(holder)  # type: ignore[arg-type]
    assert pointers == {key: value.data_ptr() for key, value in callback.ema_state.items()}

    with torch.no_grad():
        holder.policy.weight.fill_(2.0)
        holder.policy.bias.fill_(2.0)
    callback.on_train_batch_end(None, holder, None, None, 0)  # type: ignore[arg-type]
    for value in callback.ema_state.values():
        torch.testing.assert_close(value, torch.full_like(value, 2.0), rtol=0, atol=0)

    decay = callback.get_decay(1)
    with torch.no_grad():
        holder.policy.weight.fill_(4.0)
        holder.policy.bias.fill_(4.0)
    callback.on_train_batch_end(None, holder, None, None, 1)  # type: ignore[arg-type]
    expected = 2.0 * decay + 4.0 * (1.0 - decay)
    for value in callback.ema_state.values():
        torch.testing.assert_close(
            value, torch.full_like(value, expected), rtol=0, atol=1e-7
        )
