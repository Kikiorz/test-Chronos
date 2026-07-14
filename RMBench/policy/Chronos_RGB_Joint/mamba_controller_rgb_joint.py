"""Online DINOv3 RGB controller for native 14-D RMBench joint targets."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from . import mamba_policy_par_2D_IMLE_Joint
from .contracts import INFERENCE_SAMPLE_STEPS, JOINT_ORDER, base_policy_contract
from .dinov3_backbone import DINOV3_CACHE_DEVICE_TYPE
from .mamba_policy_par_2D_IMLE_Joint import MambaConfig, MambaPolicy
from .scaler_M import Scaler


JOINT_KEYS = JOINT_ORDER
ACTION_KEYS = tuple(f"{key}_act" for key in JOINT_KEYS)


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


class MambaRGBJointController:
    """Stateful single-step controller with bounded temporal ensembling memory."""

    def __init__(self, args: Mapping[str, Any]):
        self.device = torch.device(str(args.get("device", "cuda:0")))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {self.device} was requested, but CUDA is unavailable")
        if self.device.type != DINOV3_CACHE_DEVICE_TYPE:
            raise ValueError(
                "Formal RGB-Joint deployment requires CUDA so online DINOv3 uses "
                "the cache-validated numerical backend; CPU is not supported."
            )

        self.camera_name = str(args.get("camera_name", "head_camera"))
        self.future_steps = int(args.get("future_steps", 16))
        self.action_dim = 14
        self.sample_steps = int(args.get("sample_steps", INFERENCE_SAMPLE_STEPS))
        self.temporal_agg = bool(args.get("temporal_agg", True))
        self.temporal_decay = float(args.get("temporal_decay", 0.01))
        self.clip_to_training_range = bool(args.get("clip_to_training_range", True))
        self.training_range_margin = float(args.get("training_range_margin", 0.05))
        if self.future_steps <= 0 or self.sample_steps <= 0:
            raise ValueError("future_steps and sample_steps must both be positive")
        if self.sample_steps != INFERENCE_SAMPLE_STEPS:
            raise ValueError(
                f"Checkpoint contract fixes sample_steps={INFERENCE_SAMPLE_STEPS}; "
                f"got {self.sample_steps}"
            )
        if self.temporal_decay < 0:
            raise ValueError("temporal_decay must be non-negative")
        if self.training_range_margin < 0:
            raise ValueError("training_range_margin must be non-negative")

        config = MambaConfig()
        config.embed_dim = 1024
        config.d_model = 1024
        config.action_dim = self.action_dim
        config.lowdim_dim = self.action_dim
        config.num_blocks = 6
        config.future_steps = self.future_steps
        config.camera_names = [self.camera_name]
        self.config = config

        lowdim_shapes: dict[str, object] = {key: 1 for key in JOINT_KEYS}
        lowdim_shapes.update({key: (self.future_steps, 1) for key in ACTION_KEYS})
        self.scaler = Scaler(lowdim_dict=lowdim_shapes)
        self.scaler_path = _existing_file(args["scaler_path"], "RGB Joint scaler")
        print(f"[MambaRGBJointController] Loading scaler: {self.scaler_path}")
        self.scaler.load(self.scaler_path)
        self.scaler_fingerprint = self.scaler.fingerprint()
        self.scaler.to(self.device)
        self.scaler.eval()

        print("[MambaRGBJointController] Initializing DINOv3 RGB Joint MambaPolicy")
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
            backbone_weights=None,
        ).to(self.device)

        # torch.load may need this top-level name to unpickle MambaConfig from
        # a Lightning checkpoint produced by the standalone training script.
        sys.modules.setdefault(
            "mamba_policy_par_2D_IMLE_Joint", mamba_policy_par_2D_IMLE_Joint
        )
        self.ckpt_path = _existing_file(args["ckpt_path"], "RGB checkpoint")
        print(f"[MambaRGBJointController] Loading checkpoint strictly: {self.ckpt_path}")
        # Keep checkpoint tensors on CPU while copying them into the policy;
        # mapping the whole checkpoint to CUDA would temporarily duplicate the
        # model weights in GPU memory.
        try:
            checkpoint = torch.load(
                self.ckpt_path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
        except TypeError:  # torch versions without mmap/weights_only
            checkpoint = torch.load(self.ckpt_path, map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise TypeError("RGB Joint deployment requires a checkpoint mapping")
        contract = checkpoint.get("policy_contract")
        if not isinstance(contract, Mapping):
            raise ValueError("Checkpoint is missing the RGB Joint policy_contract")
        expected_contract = base_policy_contract()
        mismatches = {
            key: (contract.get(key), expected)
            for key, expected in expected_contract.items()
            if contract.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"Incompatible RGB Joint checkpoint contract: {mismatches}")
        for fingerprint_key in (
            "backbone_weights_sha256",
            "scaler_fingerprint",
            "cache_contract_sha256",
            "cache_dataset_sha256",
        ):
            value = contract.get(fingerprint_key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(
                    f"Checkpoint contract has invalid {fingerprint_key}: {value!r}"
                )
        if contract["scaler_fingerprint"] != self.scaler_fingerprint:
            raise ValueError(
                "External scaler does not belong to this checkpoint: "
                f"checkpoint={contract['scaler_fingerprint']}, "
                f"external={self.scaler_fingerprint}"
            )
        compact_scaler = checkpoint.get("scaler_state_dict")
        if isinstance(compact_scaler, Mapping):
            embedded_scaler = {
                str(key): value
                for key, value in compact_scaler.items()
                if isinstance(key, str) and torch.is_tensor(value)
            }
        else:
            lightning_state = checkpoint.get("state_dict")
            if not isinstance(lightning_state, Mapping):
                raise ValueError(
                    "RGB Joint checkpoint has neither scaler_state_dict nor "
                    "Lightning state_dict"
                )
            embedded_scaler = {
                key[len("scaler."):]: value
                for key, value in lightning_state.items()
                if isinstance(key, str)
                and key.startswith("scaler.")
                and torch.is_tensor(value)
            }
        embedded_fingerprint = Scaler.state_fingerprint(embedded_scaler)
        if embedded_fingerprint != self.scaler_fingerprint:
            raise ValueError(
                "Checkpoint's embedded scaler differs from the validated external scaler"
            )
        uses_ema = isinstance(checkpoint, Mapping) and "ema_policy_state_dict" in checkpoint
        state = _policy_state_dict(checkpoint)
        self.policy.load_state_dict(state, strict=True)
        num_loaded_tensors = len(state)
        del state, checkpoint
        self.policy.eval()
        self.policy.requires_grad_(False)
        weight_kind = "EMA" if uses_ema else "raw/backward-compatible"
        print(
            f"[MambaRGBJointController] Strictly loaded {num_loaded_tensors} "
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
        self.clipped_action_steps = 0
        self.clipped_joint_counts = torch.zeros(
            self.action_dim, device=self.device, dtype=torch.long
        )
        self.suspected_topp_failures = {"left": 0, "right": 0}
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
        values = {key: qpos[:, index:index + 1] for index, key in enumerate(JOINT_KEYS)}
        normalized = self.scaler.normalize(values)
        return torch.cat([normalized[key] for key in JOINT_KEYS], dim=-1).float()

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

    def _apply_joint_safety(self, action: torch.Tensor) -> torch.Tensor:
        """Apply a simulation output guard after physical-unit aggregation.

        This is not a real-robot joint-limit, velocity, collision, or E-stop
        safety layer.  It only bounds RMBench outputs using train statistics.
        """

        if not torch.isfinite(action).all():
            raise FloatingPointError("Refusing to clip a non-finite joint action")
        original = action.clone()
        action = action.clone()
        if self.clip_to_training_range:
            lower = []
            upper = []
            for key in ACTION_KEYS:
                minimum = self.scaler.min_dict[key].amin()
                maximum = self.scaler.max_dict[key].amax()
                span = (maximum - minimum).clamp_min(1e-6)
                lower.append(minimum - self.training_range_margin * span)
                upper.append(maximum + self.training_range_margin * span)
            action = torch.maximum(
                torch.minimum(action, torch.stack(upper).to(action)),
                torch.stack(lower).to(action),
            )
        # RMBench maps both grippers to normalized [0,1] independently of TOPP.
        action[6] = action[6].clamp(0.0, 1.0)
        action[13] = action[13].clamp(0.0, 1.0)
        clipped = ~torch.isclose(action, original, rtol=0.0, atol=1e-7)
        if clipped.any() and hasattr(self, "clipped_joint_counts"):
            self.clipped_action_steps += 1
            self.clipped_joint_counts.add_(clipped.long())
            if self.clipped_action_steps <= 10 or self.clipped_action_steps % 100 == 0:
                indices = clipped.nonzero(as_tuple=False).flatten().tolist()
                print(
                    "[Chronos_RGB_Joint] output guard clipped joints "
                    f"{indices} at policy step {self.t} "
                    f"(clipped steps total={self.clipped_action_steps})"
                )
        return action

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
            horizon = self.t - source_step
            if horizon < self.future_steps:
                candidates.append(self._action_ring[source_row, horizon])

        if not candidates:
            return sequence[0]
        stacked = torch.stack(candidates, dim=0)
        # Preserve the released controller's exact ordering and weighting:
        # candidates are old -> new, and exp(-k * index) gives the oldest
        # available prediction the largest weight.  Only storage changed from
        # a 5000-squared tensor to this bounded ring.
        candidate_index = torch.arange(len(candidates), device=self.device, dtype=stacked.dtype)
        weights = torch.exp(-self.temporal_decay * candidate_index)
        weights = weights / weights.sum()
        return (stacked * weights.unsqueeze(-1)).sum(dim=0)

    def record_execution(
        self,
        before_drive: np.ndarray,
        requested: np.ndarray,
        after_drive: np.ndarray,
        after_real: np.ndarray,
    ) -> None:
        """Log likely swallowed RMBench TOPP failures without altering control."""

        arrays = [
            np.asarray(value, dtype=np.float32).reshape(-1)
            for value in (before_drive, requested, after_drive, after_real)
        ]
        if any(value.shape != (self.action_dim,) or not np.isfinite(value).all() for value in arrays):
            raise ValueError("Execution monitor requires four finite 14-D vectors")
        before, target, after, real = arrays
        for arm, indices in (("left", slice(0, 6)), ("right", slice(7, 13))):
            requested_delta = float(np.max(np.abs(target[indices] - before[indices])))
            drive_delta = float(np.max(np.abs(after[indices] - before[indices])))
            if requested_delta > 1e-3 and drive_delta < 1e-5:
                self.suspected_topp_failures[arm] += 1
                real_error = float(np.max(np.abs(real[indices] - target[indices])))
                print(
                    f"[Chronos_RGB_Joint] WARNING: suspected swallowed {arm} TOPP "
                    f"failure at policy step {max(0, self.t - 1)}; "
                    f"requested_delta={requested_delta:.6f}, "
                    f"real_target_error={real_error:.6f}"
                )

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
            raise ValueError(f"Expected one 14-D joint target state, got {tuple(qpos.shape)}")
        if not torch.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
            raise ValueError("Controller RGB input must be finite and scaled to [0,1]")
        if not torch.isfinite(qpos).all():
            raise ValueError("Controller joint input contains NaN or infinity")
        if not (0.0 - 1e-5 <= qpos[0, 6] <= 1.0 + 1e-5):
            raise ValueError("Left normalized gripper state is outside [0,1]")
        if not (0.0 - 1e-5 <= qpos[0, 13] <= 1.0 + 1e-5):
            raise ValueError("Right normalized gripper state is outside [0,1]")

        qpos_normalized = self._normalize_qpos(qpos)
        fused = self.policy.fusion_engine(image, qpos_normalized)
        predicted_normalized, self.hiddens = self.policy.step(
            fused,
            self.hiddens,
            sample_steps=self.sample_steps,
        )
        predicted = self._denormalize_sequence(predicted_normalized)[0]
        if not torch.isfinite(predicted).all():
            raise FloatingPointError(
                "Chronos_RGB_Joint produced a non-finite future action sequence"
            )

        if self.temporal_agg:
            action = self._ensemble(predicted)
        else:
            action = predicted[0]
        if not torch.isfinite(action).all():
            raise FloatingPointError(
                "Chronos_RGB_Joint produced a non-finite aggregated action"
            )
        action = self._apply_joint_safety(action)
        self.t += 1

        if not torch.isfinite(action).all():
            raise FloatingPointError("Chronos_RGB_Joint produced a non-finite action")
        return action.detach().cpu().numpy().astype(np.float32, copy=False)
