"""Online inference controller for RGB + dual-arm 16-D EE Chronos."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from . import mamba_policy_par_2D_IMLE_EE
from .mamba_policy_par_2D_IMLE_EE import MambaConfig, MambaPolicy
from .scaler_M import Scaler


EE_KEYS = (
    "ee_l_x", "ee_l_y", "ee_l_z", "ee_l_qw", "ee_l_qx", "ee_l_qy", "ee_l_qz", "gripper_l",
    "ee_r_x", "ee_r_y", "ee_r_z", "ee_r_qw", "ee_r_qx", "ee_r_qy", "ee_r_qz", "gripper_r",
)
ACTION_KEYS = tuple(f"{key}_act" for key in EE_KEYS)


def _existing_file(path: os.PathLike[str] | str, description: str) -> str:
    resolved = os.path.abspath(os.path.expandvars(os.path.expanduser(os.fspath(path))))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    return resolved


def _policy_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping) and "ema_policy_state_dict" in checkpoint:
        state = checkpoint["ema_policy_state_dict"]
    else:
        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint must be a state_dict or contain a 'state_dict' mapping")

    # Lightning training checkpoints contain policy.*, scaler.* and metric.*.
    # Selecting the policy namespace is deliberate; loading that namespace
    # into MambaPolicy below is still fully strict.
    policy_items = {
        key[len("policy."):]: value
        for key, value in state.items()
        if isinstance(key, str) and key.startswith("policy.")
    }
    if policy_items:
        return policy_items

    direct = {str(key): value for key, value in state.items()}
    if not direct:
        raise ValueError("Checkpoint state_dict is empty")
    return direct


class MambaRGBController:
    """Stateful single-step controller with bounded temporal ensembling memory."""

    def __init__(self, args: Mapping[str, Any]):
        self.device = torch.device(str(args.get("device", "cuda:0")))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {self.device} was requested, but CUDA is unavailable")

        self.camera_name = str(args.get("camera_name", "head_camera"))
        self.future_steps = int(args.get("future_steps", 16))
        self.action_dim = 16
        self.sample_steps = int(args.get("sample_steps", 5))
        self.temporal_agg = bool(args.get("temporal_agg", True))
        self.temporal_decay = float(args.get("temporal_decay", 0.01))
        self.execution_horizon_offset = int(args.get("execution_horizon_offset", 0))
        self.visual_architecture = str(
            args.get("visual_architecture", "official_realworld")
        ).lower()
        if self.future_steps <= 0 or self.sample_steps <= 0:
            raise ValueError("future_steps and sample_steps must both be positive")
        if self.temporal_decay < 0:
            raise ValueError("temporal_decay must be non-negative")
        if self.execution_horizon_offset != 0:
            raise ValueError("Official RMBench targets require execution_horizon_offset=0")
        if self.visual_architecture != "official_realworld":
            raise ValueError("visual_architecture must be 'official_realworld'")

        config = MambaConfig()
        config.embed_dim = 1024
        config.d_model = 1024
        config.action_dim = self.action_dim
        config.lowdim_dim = self.action_dim
        config.num_blocks = 6
        config.future_steps = self.future_steps
        config.camera_names = [self.camera_name]
        config.visual_architecture = self.visual_architecture

        # The architecture is explicit.  A checkpoint from an older custom
        # RGB variant must fail strict loading instead of being guessed.
        sys.modules.setdefault("mamba_policy_par_2D_IMLE_EE", mamba_policy_par_2D_IMLE_EE)
        self.ckpt_path = _existing_file(args["ckpt_path"], "RGB checkpoint")
        print(f"[MambaRGBController] Reading checkpoint: {self.ckpt_path}")
        checkpoint = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        uses_ema = isinstance(checkpoint, Mapping) and "ema_policy_state_dict" in checkpoint
        state = _policy_state_dict(checkpoint)
        self.config = config

        lowdim_shapes: dict[str, object] = {key: 1 for key in EE_KEYS}
        lowdim_shapes.update({key: (self.future_steps, 1) for key in ACTION_KEYS})
        self.scaler = Scaler(lowdim_dict=lowdim_shapes)
        self.scaler_path = _existing_file(args["scaler_path"], "RGB EE scaler")
        print(f"[MambaRGBController] Loading scaler: {self.scaler_path}")
        self.scaler.load(self.scaler_path)
        self.scaler.to(self.device)
        self.scaler.eval()

        print(
            "[MambaRGBController] Initializing RGB EE MambaPolicy "
            f"architecture={self.visual_architecture}"
        )
        self.policy = MambaPolicy(
            camera_names=config.camera_names,
            embed_dim=config.embed_dim,
            lowdim_dim=config.lowdim_dim,
            d_model=config.d_model,
            action_dim=config.action_dim,
            num_blocks=config.num_blocks,
            mamba_cfg=config,
            future_steps=self.future_steps,
            # A deployment checkpoint contains the complete frozen backbone.
            # Avoid an unnecessary network download before strict loading.
            pretrained_backbone=False,
            freeze_backbone=True,
            visual_architecture=self.visual_architecture,
        ).to(self.device)

        print(f"[MambaRGBController] Loading checkpoint strictly: {self.ckpt_path}")
        self.policy.load_state_dict(state, strict=True)
        num_loaded_tensors = len(state)
        del state, checkpoint
        self.policy.eval()
        self.policy.requires_grad_(False)
        weight_kind = "EMA" if uses_ema else "raw"
        print(
            f"[MambaRGBController] Strictly loaded {num_loaded_tensors} "
            f"{weight_kind} policy tensors"
        )

        # Q x Q x D, where Q=future_steps.  This replaces the original
        # 5000 x (5000+Q) x D allocation while retaining every sequence that
        # can still vote for the current action.
        self._action_ring = torch.empty(
            self.future_steps,
            self.future_steps,
            self.action_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self._ring_source_steps: list[int | None] = [None] * self.future_steps
        self.hiddens: Any = None
        self.t = 0
        self.reset()

    def reset(self) -> None:
        """Clear all recurrent, latent-noise, and temporal-ensemble state."""

        self.hiddens = self.policy.init_hidden_states(batch_size=1, device=self.device)
        self.t = 0
        self._ring_source_steps = [None] * self.future_steps
        self._action_ring.zero_()
        if hasattr(self.policy, "_current_z"):
            self.policy._current_z = None

    def _normalize_qpos(self, qpos: torch.Tensor) -> torch.Tensor:
        values = {key: qpos[:, index:index + 1] for index, key in enumerate(EE_KEYS)}
        normalized = self.scaler.normalize(values)
        return torch.cat([normalized[key] for key in EE_KEYS], dim=-1).float()

    def _denormalize_sequence(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[-2:] != (self.future_steps, self.action_dim):
            raise ValueError(
                "Policy returned an unexpected action shape: "
                f"{tuple(actions.shape)} (expected [B, {self.future_steps}, {self.action_dim}])"
            )
        values = {
            key: actions[..., index:index + 1]
            for index, key in enumerate(ACTION_KEYS)
        }
        denormalized = self.scaler.denormalize(values)
        return torch.cat([denormalized[key] for key in ACTION_KEYS], dim=-1)

    def _ensemble(self, sequence: torch.Tensor) -> torch.Tensor:
        """Temporally ensemble denormalized actions using a bounded ring."""

        row = self.t % self.future_steps
        self._action_ring[row].copy_(sequence)
        self._ring_source_steps[row] = self.t

        candidates = []
        first_source = max(0, self.t - self.future_steps + 1)
        for source_step in range(first_source, self.t + 1):
            source_row = source_step % self.future_steps
            if self._ring_source_steps[source_row] != source_step:
                continue
            horizon = self.t - source_step + self.execution_horizon_offset
            if horizon < self.future_steps:
                candidates.append(self._action_ring[source_row, horizon])

        if not candidates:
            return sequence[self.execution_horizon_offset]
        stacked = torch.stack(candidates, dim=0)
        # Preserve the released controller's exact ordering and weighting:
        # candidates are old -> new, and exp(-k * index) gives the oldest
        # available prediction the largest weight.  Only storage changed from
        # a 5000-squared tensor to this bounded ring.
        candidate_index = torch.arange(len(candidates), device=self.device, dtype=stacked.dtype)
        weights = torch.exp(-self.temporal_decay * candidate_index)
        weights = weights / weights.sum()
        return (stacked * weights.unsqueeze(-1)).sum(dim=0)

    @torch.inference_mode()
    def get_action(self, obs_dict: Mapping[str, torch.Tensor]) -> np.ndarray:
        image = obs_dict["image"].to(self.device, dtype=torch.float32)
        qpos = obs_dict["qpos"].to(self.device, dtype=torch.float32)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError(f"Expected one NCHW RGB image, got {tuple(image.shape)}")
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)
        if qpos.shape != (1, self.action_dim):
            raise ValueError(f"Expected one 16-D EE state, got {tuple(qpos.shape)}")

        qpos_normalized = self._normalize_qpos(qpos)
        fused = self.policy.fusion_engine(image, qpos_normalized)
        predicted_normalized, self.hiddens = self.policy.step(
            fused,
            self.hiddens,
            sample_steps=self.sample_steps,
        )
        predicted = self._denormalize_sequence(predicted_normalized)[0]

        if self.temporal_agg:
            action = self._ensemble(predicted)
        else:
            action = predicted[self.execution_horizon_offset]
        self.t += 1

        if not torch.isfinite(action).all():
            raise FloatingPointError("Chronos_RGB produced a non-finite action")
        return action.detach().cpu().numpy().astype(np.float32, copy=False)
