"""Strict-checkpoint-safe exponential moving average for Chronos RGB.

Lightning's normal ``state_dict`` remains the raw training state so optimizer
resume is exact.  A complete, deployment-only policy state is stored separately
under ``ema_policy_state_dict``.  In particular, integer buffers are retained;
dropping them would make the controller's strict load fail.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from pytorch_lightning import Callback, LightningModule, Trainer


class WarmupPolicyEMACallback(Callback):
    """Warmup EMA matching the released Chronos update schedule.

    EMA is updated at every training-batch end, including batches inside a
    gradient-accumulation interval.  This intentionally matches the released
    3D callback's timing.  Validation temporarily uses EMA parameters, while
    the raw parameters are restored before training resumes.
    """

    checkpoint_key = "ema_policy_state_dict"
    checkpoint_step_key = "ema_optimization_step"
    checkpoint_version_key = "ema_format_version"

    def __init__(
        self,
        inv_gamma: float = 1.0,
        power: float = 2.0 / 3.0,
        min_value: float = 0.0,
        max_value: float = 0.9999,
    ) -> None:
        super().__init__()
        if inv_gamma <= 0:
            raise ValueError("inv_gamma must be positive")
        if power <= 0:
            raise ValueError("power must be positive")
        if not 0.0 <= min_value <= max_value <= 1.0:
            raise ValueError("EMA values must satisfy 0 <= min_value <= max_value <= 1")
        self.inv_gamma = float(inv_gamma)
        self.power = float(power)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.ema_state: dict[str, torch.Tensor] | None = None
        self.optimization_step = 0
        self._raw_validation_state: dict[str, torch.Tensor] | None = None

    def get_decay(self, step: int) -> float:
        if step <= 0:
            return 0.0
        value = 1.0 - (1.0 + step / self.inv_gamma) ** (-self.power)
        return max(self.min_value, min(value, self.max_value))

    @staticmethod
    def _policy(pl_module: LightningModule) -> torch.nn.Module:
        policy = getattr(pl_module, "policy", None)
        if not isinstance(policy, torch.nn.Module):
            raise TypeError("WarmupPolicyEMACallback requires pl_module.policy to be an nn.Module")
        return policy

    @staticmethod
    def _clone_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.detach().clone() for key, value in state.items()}

    @staticmethod
    def _validate_state(
        reference: Mapping[str, torch.Tensor], candidate: Mapping[str, torch.Tensor], context: str
    ) -> None:
        reference_keys = set(reference)
        candidate_keys = set(candidate)
        if reference_keys != candidate_keys:
            missing = sorted(reference_keys - candidate_keys)
            unexpected = sorted(candidate_keys - reference_keys)
            raise RuntimeError(
                f"{context} policy keys do not match: missing={missing}, unexpected={unexpected}"
            )
        for key, reference_tensor in reference.items():
            candidate_tensor = candidate[key]
            if not torch.is_tensor(candidate_tensor):
                raise TypeError(f"{context} entry {key!r} is not a tensor")
            if candidate_tensor.shape != reference_tensor.shape:
                raise RuntimeError(
                    f"{context} shape mismatch for {key!r}: "
                    f"{tuple(candidate_tensor.shape)} != {tuple(reference_tensor.shape)}"
                )
            if candidate_tensor.dtype != reference_tensor.dtype:
                raise RuntimeError(
                    f"{context} dtype mismatch for {key!r}: "
                    f"{candidate_tensor.dtype} != {reference_tensor.dtype}"
                )

    def _ensure_ema_on_policy_device(self, pl_module: LightningModule) -> None:
        live_state = self._policy(pl_module).state_dict()
        if self.ema_state is None:
            self.ema_state = self._clone_state(live_state)
            return
        self._validate_state(live_state, self.ema_state, "EMA checkpoint")
        # A loaded checkpoint starts on CPU and needs one relocation.  Once
        # resident, retain the same EMA storage: cloning the full ~1.5 GiB
        # policy at every batch is mathematically redundant and very costly.
        relocated: dict[str, torch.Tensor] = {}
        for key, live_tensor in live_state.items():
            ema_tensor = self.ema_state[key].detach()
            if ema_tensor.device != live_tensor.device or ema_tensor.dtype != live_tensor.dtype:
                ema_tensor = ema_tensor.to(
                    device=live_tensor.device, dtype=live_tensor.dtype
                ).clone()
            relocated[key] = ema_tensor
        self.ema_state = relocated

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._ensure_ema_on_policy_device(pl_module)

    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._ensure_ema_on_policy_device(pl_module)
        assert self.ema_state is not None
        decay = self.get_decay(self.optimization_step)
        self.optimization_step += 1
        for key, live_tensor in self._policy(pl_module).state_dict().items():
            ema_tensor = self.ema_state[key]
            live_tensor = live_tensor.detach()
            if live_tensor.is_floating_point() or live_tensor.is_complex():
                ema_tensor.mul_(decay).add_(live_tensor, alpha=1.0 - decay)
            else:
                # Preserve integer/bool buffers required by strict deployment.
                ema_tensor.copy_(live_tensor)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._raw_validation_state is not None:
            raise RuntimeError("EMA validation weights are already active")
        self._ensure_ema_on_policy_device(pl_module)
        assert self.ema_state is not None
        policy = self._policy(pl_module)
        self._raw_validation_state = self._clone_state(policy.state_dict())
        policy.load_state_dict(self.ema_state, strict=True)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._raw_validation_state is None:
            return
        self._policy(pl_module).load_state_dict(self._raw_validation_state, strict=True)
        self._raw_validation_state = None

    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict[str, Any]
    ) -> None:
        self._ensure_ema_on_policy_device(pl_module)
        assert self.ema_state is not None

        # An unusual save during validation must still retain raw weights in the
        # ordinary Lightning state for optimizer-compatible resume.
        if self._raw_validation_state is not None:
            raw_lightning_state = checkpoint.get("state_dict")
            if not isinstance(raw_lightning_state, Mapping):
                raise TypeError("Lightning checkpoint is missing its raw state_dict")
            raw_lightning_state = dict(raw_lightning_state)
            for key, tensor in self._raw_validation_state.items():
                raw_lightning_state[f"policy.{key}"] = tensor.detach().clone()
            checkpoint["state_dict"] = raw_lightning_state

        # Direct policy keys are intentional: deployment can strict-load this
        # mapping without stripping the Lightning ``policy.`` namespace.
        checkpoint[self.checkpoint_key] = self._clone_state(self.ema_state)
        checkpoint[self.checkpoint_step_key] = int(self.optimization_step)
        checkpoint[self.checkpoint_version_key] = 1

    def on_load_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict[str, Any]
    ) -> None:
        loaded = checkpoint.get(self.checkpoint_key)
        if loaded is None:
            # Backward-compatible resume from a checkpoint created before EMA.
            self.ema_state = None
            self.optimization_step = 0
            return
        if not isinstance(loaded, Mapping) or not loaded:
            raise TypeError(f"Checkpoint field {self.checkpoint_key!r} must be a non-empty mapping")
        if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in loaded.items()):
            raise TypeError(f"Checkpoint field {self.checkpoint_key!r} must map strings to tensors")
        self.ema_state = self._clone_state(loaded)
        self.optimization_step = int(checkpoint.get(self.checkpoint_step_key, 0))
