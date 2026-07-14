"""Keep the true best validation EMA as one compact deployment artifact."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

try:
    from .contracts import base_policy_contract
    from .ema_callback import WarmupPolicyEMACallback
    from .scaler_M import Scaler
except ImportError:  # direct script execution
    from contracts import base_policy_contract
    from ema_callback import WarmupPolicyEMACallback
    from scaler_M import Scaler


class BestEMADeployCheckpoint(Callback):
    """Atomically retain one EMA-only checkpoint at the true best val loss.

    Full optimizer checkpoints are deliberately written only every few epochs
    because each is almost 6 GiB. This callback observes every completed
    validation epoch and writes a single ~1.5 GiB deployment artifact only
    when the deterministic EMA validation loss improves. It never changes the
    model, optimizer, scheduler, RNG, or full-checkpoint cadence.
    """

    format_name = "chronos_rgb_joint_deploy_v1"

    def __init__(self, path: str | Path, monitor: str = "val_loss") -> None:
        super().__init__()
        self.path = Path(path).expanduser().resolve()
        self.monitor = str(monitor)
        self.best_score = math.inf
        self.best_epoch = -1
        self.best_global_step = -1

    @property
    def state_key(self) -> str:
        return self._generate_state_key(monitor=self.monitor, mode="min")

    def state_dict(self) -> dict[str, float | int]:
        return {
            "version": 1,
            "best_score": float(self.best_score),
            "best_epoch": int(self.best_epoch),
            "best_global_step": int(self.best_global_step),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if int(state_dict.get("version", 1)) != 1:
            raise ValueError(f"Unsupported best EMA callback state: {state_dict}")
        score = float(state_dict.get("best_score", math.inf))
        epoch = int(state_dict.get("best_epoch", -1))
        global_step = int(state_dict.get("best_global_step", -1))
        if not math.isfinite(score) and score != math.inf:
            raise ValueError(f"Invalid restored best EMA score: {score}")
        if math.isfinite(score) and (epoch < 0 or global_step < 0):
            raise ValueError(
                "A finite best EMA score requires non-negative epoch/global_step: "
                f"score={score}, epoch={epoch}, global_step={global_step}"
            )
        if score == math.inf and (epoch != -1 or global_step != -1):
            raise ValueError("An empty best EMA state must use epoch=global_step=-1")
        self.best_score = score
        self.best_epoch = epoch
        self.best_global_step = global_step

    @staticmethod
    def _contract(pl_module: LightningModule) -> dict[str, object]:
        run_contract = getattr(pl_module, "run_contract", None)
        if not isinstance(run_contract, Mapping):
            raise TypeError("Best EMA export requires pl_module.run_contract")
        return {**base_policy_contract(), **dict(run_contract)}

    @staticmethod
    def _ema_callback(trainer: Trainer) -> WarmupPolicyEMACallback:
        matches = [
            callback
            for callback in trainer.callbacks
            if isinstance(callback, WarmupPolicyEMACallback)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "Best EMA export requires exactly one WarmupPolicyEMACallback; "
                f"found {len(matches)}"
            )
        return matches[0]

    @staticmethod
    def _cpu_copy(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().to(device="cpu", copy=True)
            for key, value in state.items()
        }

    def _reconcile_existing(self, pl_module: LightningModule) -> None:
        if not self.path.is_file():
            if math.isfinite(self.best_score):
                raise FileNotFoundError(
                    "Resume state records a best EMA checkpoint, but its artifact is missing: "
                    f"score={self.best_score}, epoch={self.best_epoch}, path={self.path}"
                )
            return
        try:
            payload = torch.load(
                self.path, map_location="cpu", weights_only=True, mmap=True
            )
        except TypeError:  # older torch
            payload = torch.load(self.path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise TypeError(f"Existing best EMA artifact is not a mapping: {self.path}")
        if payload.get("format") != self.format_name:
            raise ValueError(f"Existing best EMA artifact has an invalid format: {self.path}")
        existing_contract = payload.get("policy_contract")
        expected_contract = self._contract(pl_module)
        if not isinstance(existing_contract, Mapping) or dict(existing_contract) != expected_contract:
            raise RuntimeError(
                "Refusing to mix an existing best EMA artifact with a different "
                f"policy/data/scaler contract: {self.path}"
            )
        if payload.get("monitor") != self.monitor or payload.get("mode") != "min":
            raise ValueError(f"Existing best EMA artifact has invalid monitor/mode: {self.path}")

        policy = getattr(pl_module, "policy", None)
        if not isinstance(policy, torch.nn.Module):
            raise TypeError("Best EMA preflight requires pl_module.policy")
        policy_state = payload.get("ema_policy_state_dict")
        if not isinstance(policy_state, Mapping) or not policy_state:
            raise ValueError(f"Existing best EMA artifact has no EMA policy: {self.path}")
        if not all(
            isinstance(key, str) and torch.is_tensor(value)
            for key, value in policy_state.items()
        ):
            raise TypeError(f"Existing best EMA policy is not string-to-tensor: {self.path}")
        WarmupPolicyEMACallback._validate_state(
            policy.state_dict(), policy_state, "Existing best EMA artifact"
        )

        scaler_state = payload.get("scaler_state_dict")
        if not isinstance(scaler_state, Mapping) or not scaler_state:
            raise ValueError(f"Existing best EMA artifact has no scaler state: {self.path}")
        scaler_tensors = {
            str(key): value
            for key, value in scaler_state.items()
            if isinstance(key, str) and torch.is_tensor(value)
        }
        if len(scaler_tensors) != len(scaler_state):
            raise TypeError(f"Existing best EMA scaler is not string-to-tensor: {self.path}")
        if Scaler.state_fingerprint(scaler_tensors) != expected_contract.get(
            "scaler_fingerprint"
        ):
            raise RuntimeError(f"Existing best EMA scaler fingerprint is invalid: {self.path}")

        score = float(payload.get("val_loss", math.nan))
        epoch = int(payload.get("epoch", -1))
        global_step = int(payload.get("global_step", -1))
        ema_step = int(payload.get("ema_optimization_step", -1))
        if not math.isfinite(score) or epoch < 0 or global_step < 0 or ema_step < 0:
            raise ValueError(f"Existing best EMA artifact has invalid metadata: {self.path}")
        if not math.isfinite(self.best_score) and self.best_score == math.inf:
            self.best_score = score
            self.best_epoch = epoch
            self.best_global_step = global_step
        elif score < self.best_score:
            # The compact artifact can be newer than the five-epoch full resume
            # checkpoint. Preserve that newer/better result across a crash.
            self.best_score = score
            self.best_epoch = epoch
            self.best_global_step = global_step
        elif (
            score > self.best_score
            or epoch != self.best_epoch
            or global_step != self.best_global_step
        ):
            raise RuntimeError(
                "Best EMA callback state and artifact disagree; refusing to silently "
                f"replace a potentially better result: state=({self.best_score}, "
                f"epoch {self.best_epoch}, step {self.best_global_step}), "
                f"artifact=({score}, epoch {epoch}, step {global_step})"
            )
        del payload

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        local_error: BaseException | None = None
        result: tuple[bool, float | str, int | str, int | str]
        if trainer.is_global_zero:
            try:
                self._reconcile_existing(pl_module)
                result = (
                    True,
                    float(self.best_score),
                    int(self.best_epoch),
                    int(self.best_global_step),
                )
            except BaseException as exc:
                local_error = exc
                result = (False, type(exc).__name__, str(exc), "")
        else:
            result = (False, "uninitialized", "uninitialized", "")
        result = trainer.strategy.broadcast(result, src=0)
        if not result[0]:
            if local_error is not None:
                raise local_error
            raise RuntimeError(f"Best EMA preflight failed on rank zero: {result[1]}: {result[2]}")
        self.best_score = float(result[1])
        self.best_epoch = int(result[2])
        self.best_global_step = int(result[3])

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # In Lightning 2.4 the epoch-reduced metrics are populated after
        # on_validation_epoch_end and before on_validation_end. ModelCheckpoint
        # uses this same hook for monitored validation metrics.
        if trainer.sanity_checking:
            return
        local_metric_error: BaseException | None = None
        metric_result: tuple[bool, float | str, str]
        if trainer.is_global_zero:
            try:
                metric = trainer.callback_metrics.get(self.monitor)
                if metric is None:
                    raise RuntimeError(f"Validation metric {self.monitor!r} is unavailable")
                if torch.is_tensor(metric):
                    score = float(metric.detach().cpu().item())
                else:
                    score = float(metric)
                if not math.isfinite(score):
                    raise FloatingPointError(
                        f"Validation metric {self.monitor!r} is non-finite: {score}"
                    )
                metric_result = (True, score, "")
            except BaseException as exc:
                local_metric_error = exc
                metric_result = (False, type(exc).__name__, str(exc))
        else:
            metric_result = (False, "uninitialized", "uninitialized")
        metric_result = trainer.strategy.broadcast(metric_result, src=0)
        if not metric_result[0]:
            if local_metric_error is not None:
                raise local_metric_error
            raise RuntimeError(
                f"Best EMA metric failed on rank zero: {metric_result[1]}: "
                f"{metric_result[2]}"
            )
        score = float(metric_result[1])
        if score >= self.best_score:
            return

        epoch = int(trainer.current_epoch)
        global_step = int(trainer.global_step)
        local_error: BaseException | None = None
        result: tuple[bool, str, str]
        if trainer.is_global_zero:
            try:
                ema_callback = self._ema_callback(trainer)
                if ema_callback.ema_state is None:
                    raise RuntimeError("EMA state is unavailable at validation end")
                policy = getattr(pl_module, "policy", None)
                if not isinstance(policy, torch.nn.Module):
                    raise TypeError("Best EMA export requires pl_module.policy")
                ema_callback._validate_state(
                    policy.state_dict(), ema_callback.ema_state, "Best EMA export"
                )

                scaler = getattr(pl_module, "scaler", None)
                if not isinstance(scaler, Scaler):
                    raise TypeError("Best EMA export requires pl_module.scaler")
                scaler_state = self._cpu_copy(scaler.state_dict())
                contract = self._contract(pl_module)
                if Scaler.state_fingerprint(scaler_state) != contract.get("scaler_fingerprint"):
                    raise RuntimeError("Best EMA export scaler does not match the run contract")

                policy_state = self._cpu_copy(ema_callback.ema_state)
                payload = {
                    "format": self.format_name,
                    "policy_contract": contract,
                    "ema_policy_state_dict": policy_state,
                    "scaler_state_dict": scaler_state,
                    "val_loss": score,
                    "epoch": epoch,
                    "global_step": global_step,
                    "ema_optimization_step": int(ema_callback.optimization_step),
                    "monitor": self.monitor,
                    "mode": "min",
                }
                partial = self.path.parent / f".{self.path.name}.{os.getpid()}.partial"
                partial.unlink(missing_ok=True)
                try:
                    torch.save(payload, partial)
                    file_fd = os.open(partial, os.O_RDONLY)
                    try:
                        os.fsync(file_fd)
                    finally:
                        os.close(file_fd)
                    os.replace(partial, self.path)
                    directory_fd = os.open(self.path.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                except BaseException:
                    partial.unlink(missing_ok=True)
                    raise
                finally:
                    del payload, policy_state, scaler_state
                print(
                    f"Saved true-best EMA deploy checkpoint: {self.path} "
                    f"(epoch={epoch}, {self.monitor}={score:.8f}, "
                    f"size={self.path.stat().st_size / 2**30:.2f} GiB)"
                )
                result = (True, "", "")
            except BaseException as exc:
                local_error = exc
                result = (False, type(exc).__name__, str(exc))
        else:
            result = (False, "uninitialized", "uninitialized")
        # Broadcast is also the save barrier: nonzero ranks cannot start the
        # next training epoch until rank zero has atomically replaced the file.
        result = trainer.strategy.broadcast(result, src=0)
        if not result[0]:
            if local_error is not None:
                raise local_error
            raise RuntimeError(f"Best EMA save failed on rank zero: {result[1]}: {result[2]}")
        self.best_score = score
        self.best_epoch = epoch
        self.best_global_step = global_step
