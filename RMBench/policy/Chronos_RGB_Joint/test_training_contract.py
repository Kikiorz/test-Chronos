"""Checkpoint/resume guards that prevent silent data or scaler mixing."""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace
from pathlib import Path

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from .M_dataset_robotwinRGB_J import ACTION_KEYS, JOINT_KEYS, make_joint_scaler
from .best_ema_checkpoint import BestEMADeployCheckpoint
from .contracts import base_policy_contract
from .ema_callback import WarmupPolicyEMACallback
from .export_deploy_checkpoint import (
    _validate_complete_ema_policy,
    _validation_metadata,
)
from .scaler_M import Scaler
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


def test_true_best_ema_deploy_checkpoint_is_atomic_and_resume_aware() -> None:
    scaler = make_joint_scaler(16)
    observations = torch.arange(20 * 14, dtype=torch.float32).reshape(20, 14)
    actions = torch.stack([observations + horizon for horizon in range(16)], dim=1)
    data = {key: observations[:, index:index + 1] for index, key in enumerate(JOINT_KEYS)}
    data.update(
        {key: actions[..., index:index + 1] for index, key in enumerate(ACTION_KEYS)}
    )
    scaler.fit(data)

    class Holder:
        def __init__(self) -> None:
            self.policy = torch.nn.Linear(3, 2)
            self.scaler = scaler
            self.run_contract = {"scaler_fingerprint": scaler.fingerprint()}

    class Strategy:
        def barrier(self, name: str) -> None:
            assert name.startswith("best_ema_deploy_")

        def broadcast(self, value, src: int = 0):
            assert src == 0
            return value

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "best.pth"
        holder = Holder()
        ema = WarmupPolicyEMACallback()
        ema._ensure_ema_on_policy_device(holder)  # type: ignore[arg-type]
        callback = BestEMADeployCheckpoint(path)
        trainer = SimpleNamespace(
            callbacks=[ema, callback],
            strategy=Strategy(),
            is_global_zero=True,
            sanity_checking=False,
            callback_metrics={"val_loss": torch.tensor(0.5)},
            current_epoch=3,
            global_step=12,
        )
        callback.on_fit_start(trainer, holder)  # type: ignore[arg-type]
        callback.on_validation_end(trainer, holder)  # type: ignore[arg-type]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["format"] == "chronos_rgb_joint_deploy_v1"
        assert payload["val_loss"] == 0.5
        assert payload["epoch"] == 3
        assert payload["global_step"] == 12
        assert payload["monitor"] == "val_loss"
        assert payload["mode"] == "min"
        assert set(payload["ema_policy_state_dict"]) == set(holder.policy.state_dict())
        assert Scaler.state_fingerprint(payload["scaler_state_dict"]) == scaler.fingerprint()
        first_mtime = path.stat().st_mtime_ns

        trainer.callback_metrics["val_loss"] = torch.tensor(0.6)
        callback.on_validation_end(trainer, holder)  # type: ignore[arg-type]
        assert path.stat().st_mtime_ns == first_mtime

        restored = BestEMADeployCheckpoint(path)
        trainer.callbacks = [ema, restored]
        restored.on_fit_start(trainer, holder)  # type: ignore[arg-type]
        assert restored.best_score == 0.5
        assert restored.best_epoch == 3
        assert restored.best_global_step == 12

        trainer.callback_metrics["val_loss"] = torch.tensor(0.4)
        trainer.current_epoch = 4
        restored.on_validation_end(trainer, holder)  # type: ignore[arg-type]
        improved = torch.load(path, map_location="cpu", weights_only=False)
        assert abs(improved["val_loss"] - 0.4) < 1e-7
        assert improved["epoch"] == 4

        restored_state = restored.state_dict()
        mismatched_payload = dict(improved)
        mismatched_payload["val_loss"] = 0.45
        torch.save(mismatched_payload, path)
        mismatched = BestEMADeployCheckpoint(path)
        mismatched.load_state_dict(restored_state)
        trainer.callbacks = [ema, mismatched]
        try:
            mismatched.on_fit_start(trainer, holder)  # type: ignore[arg-type]
        except RuntimeError as exc:
            assert "state and artifact disagree" in str(exc)
        else:
            raise AssertionError("A worse best-EMA artifact overrode better resume state")

        path.unlink()
        missing = BestEMADeployCheckpoint(path)
        missing.load_state_dict(restored_state)
        trainer.callbacks = [ema, missing]
        try:
            missing.on_fit_start(trainer, holder)  # type: ignore[arg-type]
        except FileNotFoundError as exc:
            assert "artifact is missing" in str(exc)
        else:
            raise AssertionError("Finite best-EMA state accepted a missing artifact")


def test_true_best_ema_uses_real_lightning_validation_and_resume_hooks() -> None:
    scaler = make_joint_scaler(16)
    observations = torch.arange(20 * 14, dtype=torch.float32).reshape(20, 14)
    actions = torch.stack([observations + horizon for horizon in range(16)], dim=1)
    scaler_data = {
        key: observations[:, index:index + 1]
        for index, key in enumerate(JOINT_KEYS)
    }
    scaler_data.update(
        {
            key: actions[..., index:index + 1]
            for index, key in enumerate(ACTION_KEYS)
        }
    )
    scaler.fit(scaler_data)

    class TinyModule(LightningModule):
        def __init__(self, fitted_scaler: Scaler) -> None:
            super().__init__()
            self.policy = torch.nn.Linear(1, 1)
            self.scaler = fitted_scaler
            self.run_contract = {
                "scaler_fingerprint": fitted_scaler.fingerprint()
            }
            self.validation_losses = (0.5, 0.4, 0.45)

        def training_step(self, batch, batch_idx):
            del batch_idx
            return self.policy(batch[0]).square().mean()

        def validation_step(self, batch, batch_idx):
            del batch, batch_idx
            value = torch.tensor(
                self.validation_losses[int(self.current_epoch)], device=self.device
            )
            self.log("val_loss", value, on_step=False, on_epoch=True, batch_size=1)

        def configure_optimizers(self):
            return torch.optim.SGD(self.policy.parameters(), lr=1e-3)

    loader = DataLoader(TensorDataset(torch.ones(1, 1)), batch_size=1)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        artifact = root / "best-ema-deploy.pth"
        ema = WarmupPolicyEMACallback()
        best = BestEMADeployCheckpoint(artifact)
        full = ModelCheckpoint(
            dirpath=root / "full",
            filename="epoch-{epoch}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            every_n_epochs=1,
        )
        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=2,
            callbacks=[ema, best, full],
            logger=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            deterministic=True,
            num_sanity_val_steps=1,
        )
        trainer.fit(TinyModule(scaler), loader, loader)
        payload = torch.load(artifact, map_location="cpu", weights_only=True)
        assert abs(float(payload["val_loss"]) - 0.4) < 1e-7
        assert payload["epoch"] == 1

        checkpoint_path = Path(full.best_model_path)
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        callback_states = [
            state
            for key, state in checkpoint["callbacks"].items()
            if "BestEMADeployCheckpoint" in str(key)
        ]
        assert len(callback_states) == 1
        assert abs(float(callback_states[0]["best_score"]) - 0.4) < 1e-7
        assert callback_states[0]["best_epoch"] == 1

        previous_mtime = artifact.stat().st_mtime_ns
        resumed_ema = WarmupPolicyEMACallback()
        resumed_best = BestEMADeployCheckpoint(artifact)
        resumed_trainer = Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=3,
            callbacks=[resumed_ema, resumed_best],
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            deterministic=True,
            num_sanity_val_steps=0,
        )
        resumed_scaler = make_joint_scaler(16)
        resumed_scaler.load_state_dict(scaler.state_dict(), strict=True)
        resumed_trainer.fit(
            TinyModule(resumed_scaler),
            loader,
            loader,
            ckpt_path=checkpoint_path,
        )
        resumed_payload = torch.load(
            artifact, map_location="cpu", weights_only=True
        )
        assert abs(float(resumed_payload["val_loss"]) - 0.4) < 1e-7
        assert resumed_payload["epoch"] == 1
        assert artifact.stat().st_mtime_ns == previous_mtime


def test_export_metadata_is_bound_to_the_exact_top_k_file() -> None:
    with tempfile.TemporaryDirectory() as directory:
        selected = (Path(directory) / "mamba-best.ckpt").resolve()
        resume = (Path(directory) / "mamba-resume.ckpt").resolve()
        checkpoint = {
            "epoch": 7,
            "global_step": 120,
            "ema_optimization_step": 360,
            "callbacks": {
                "ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min'}": {
                    "monitor": "val_loss",
                    # Deliberately stale: this must never label ``resume``.
                    "current_score": torch.tensor(0.25),
                    "best_model_score": torch.tensor(0.10),
                    "best_k_models": {str(selected): torch.tensor(0.10)},
                }
            },
        }
        assert _validation_metadata(checkpoint, resume) == {}
        metadata = _validation_metadata(checkpoint, selected)
        assert metadata["val_loss"] == float(torch.tensor(0.10))
        assert metadata["epoch"] == 7
        assert metadata["global_step"] == 120
        assert metadata["ema_optimization_step"] == 360

    raw = {
        "policy.weight": torch.zeros(2, 3),
        "policy.counter": torch.zeros((), dtype=torch.long),
        "scaler.mean": torch.zeros(1),
    }
    ema = {
        "weight": torch.ones(2, 3),
        "counter": torch.ones((), dtype=torch.long),
    }
    _validate_complete_ema_policy(ema, raw)
    try:
        _validate_complete_ema_policy({"weight": ema["weight"]}, raw)
    except ValueError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("An incomplete EMA policy was accepted for export")


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
        "dataset_manifest_sha256": "b" * 64,
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
