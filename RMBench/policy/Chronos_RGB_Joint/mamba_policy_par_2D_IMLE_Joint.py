"""Chronos with the released real-world RGB encoder and RMBench joint control.

The frozen ResNet-18 image path is copied from the released real-world policy.
The Mamba history model, five-sample RMBench IMLE generator, symplectic action
head, and 16-step prediction horizon remain the released RMBench implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .resnet18_backbone import (
        RESNET18_IMAGE_HW,
        OfficialResNet18Trunk,
    )
except ImportError:  # direct script execution
    from resnet18_backbone import (  # type: ignore
        RESNET18_IMAGE_HW,
        OfficialResNet18Trunk,
    )


# Support both ``python -m RMBench.policy...`` and scripts executed directly
# from this directory.  We deliberately reuse the reviewed 3D Chronos temporal
# and action code instead of maintaining a second 1,300-line Mamba copy.
if __package__ and "." in __package__:
    from ..Chronos.mamba_policy_par_3D_IMLE import (
        MambaConfig as _PointCloudMambaConfig,
        MambaPolicy as _PointCloudMambaPolicy,
    )
else:  # pragma: no cover - exercised by the training/deployment entrypoints
    _POLICY_DIR = Path(__file__).resolve().parents[1]
    if str(_POLICY_DIR) not in sys.path:
        sys.path.insert(0, str(_POLICY_DIR))
    from Chronos.mamba_policy_par_3D_IMLE import (  # type: ignore
        MambaConfig as _PointCloudMambaConfig,
        MambaPolicy as _PointCloudMambaPolicy,
    )


class MambaConfig(_PointCloudMambaConfig):
    """One RGB camera, 14-D absolute qpos, and released Chronos capacity."""

    def __init__(self):
        super().__init__()
        self.camera_names = ["head_camera"]
        self.embed_dim = 1024
        self.d_model = 1024
        self.lowdim_dim = 14
        self.action_dim = 14
        self.future_steps = 16
        self.num_blocks = 6
        self.pretrained_backbone = True
        self.freeze_backbone = True
        self.image_hw = RESNET18_IMAGE_HW
        self.image_chunk_size = 256


class ImageMambaFusion(nn.Module):
    """Released frozen ResNet-18 trunk and trainable RGB/proprio fusion."""

    def __init__(
        self,
        embed_dim: int = 1024,
        proprio_dim: int = 14,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        backbone_weights: str | Path | None = None,
        image_hw: tuple[int, int] = RESNET18_IMAGE_HW,
    ):
        super().__init__()
        if embed_dim != 1024:
            raise ValueError(
                "The inherited Chronos action heads expect a 1024-D fused feature; "
                f"got embed_dim={embed_dim}."
            )
        self.embed_dim = int(embed_dim)
        self.proprio_dim = int(proprio_dim)
        self.image_hw = tuple(int(value) for value in image_hw)
        if self.image_hw != RESNET18_IMAGE_HW:
            raise ValueError(
                f"The official image encoder requires {RESNET18_IMAGE_HW}, "
                f"got {self.image_hw}"
            )
        self.freeze_backbone = bool(freeze_backbone)
        self.vision_net = OfficialResNet18Trunk(
            pretrained=pretrained,
            weights_path=backbone_weights,
            freeze=freeze_backbone,
        )
        self.visual_adapter = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(inplace=True),
            nn.Flatten(1),
            nn.Linear(128 * 8 * 10, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
        )
        self.proprio_projector = nn.Sequential(
            nn.Linear(self.proprio_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 512),
            nn.LayerNorm(512),
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.embed_dim + 512, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.vision_net.eval()
        return self

    def encode_images(self, image: torch.Tensor) -> torch.Tensor:
        """Run only the frozen official trunk; inputs are already normalized."""

        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected normalized RGB [N,3,H,W], got {tuple(image.shape)}")
        if tuple(image.shape[-2:]) != self.image_hw:
            raise ValueError(
                f"Expected official RGB size {self.image_hw}, got {tuple(image.shape[-2:])}"
            )
        image = image.to(dtype=self.visual_adapter[0].weight.dtype)
        if not torch.isfinite(image).all():
            raise FloatingPointError("Normalized RGB contains NaN or Inf")
        if self.freeze_backbone:
            self.vision_net.eval()
            with torch.no_grad():
                feature_map = self.vision_net(image)
        else:
            feature_map = self.vision_net(image)
        if tuple(feature_map.shape[1:]) != (512, 15, 20):
            raise RuntimeError(
                "Official ResNet-18 trunk must output [N,512,15,20], got "
                f"{tuple(feature_map.shape)}"
            )
        return feature_map

    def fuse_feature_map(
        self, feature_map: torch.Tensor, proprio_embed: torch.Tensor
    ) -> torch.Tensor:
        if feature_map.ndim != 4 or tuple(feature_map.shape[1:]) != (512, 15, 20):
            raise ValueError(
                f"Expected ResNet map [N,512,15,20], got {tuple(feature_map.shape)}"
            )
        if proprio_embed.ndim != 2 or proprio_embed.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"Expected proprioception [N,{self.proprio_dim}], "
                f"got {tuple(proprio_embed.shape)}"
            )
        visual_feature = self.visual_adapter(feature_map)
        proprio_feature = self.proprio_projector(proprio_embed)
        return self.fusion_proj(torch.cat([visual_feature, proprio_feature], dim=-1))

    def forward(self, vision: torch.Tensor, proprio_embed: torch.Tensor) -> torch.Tensor:
        return self.fuse_feature_map(self.encode_images(vision), proprio_embed)


class MambaPolicy(_PointCloudMambaPolicy):
    """Original Chronos policy with only ``fusion_engine`` replaced by RGB."""

    def __init__(
        self,
        camera_names,
        embed_dim: int = 1024,
        lowdim_dim: int = 14,
        d_model: int = 1024,
        action_dim: int = 14,
        num_blocks: int = 6,
        block_cfg=None,
        mamba_cfg=None,
        future_steps: int = 16,
        pretrained_backbone: bool | None = None,
        freeze_backbone: bool | None = None,
        backbone_weights: str | Path | None = None,
        image_hw: tuple[int, int] = RESNET18_IMAGE_HW,
    ):
        if list(camera_names) != ["head_camera"]:
            raise ValueError(
                "Chronos_RGB_Joint intentionally uses exactly ['head_camera']; "
                f"got {list(camera_names)}"
            )
        if lowdim_dim != 14 or action_dim != 14 or future_steps != 16:
            raise ValueError(
                "Chronos_RGB_Joint requires lowdim_dim=14, action_dim=14, future_steps=16."
            )
        if embed_dim != 1024 or d_model != 1024:
            raise ValueError("Chronos_RGB_Joint requires embed_dim=d_model=1024.")

        cfg = mamba_cfg if mamba_cfg is not None else MambaConfig()
        super().__init__(
            camera_names=camera_names,
            embed_dim=embed_dim,
            lowdim_dim=lowdim_dim,
            d_model=d_model,
            action_dim=action_dim,
            num_blocks=num_blocks,
            block_cfg=block_cfg,
            mamba_cfg=cfg,
            future_steps=future_steps,
        )
        if int(getattr(self, "num_imle_samples", -1)) != 5:
            raise RuntimeError("Chronos_RGB_Joint contract requires exactly 5 IMLE samples")
        if pretrained_backbone is None:
            pretrained_backbone = bool(getattr(cfg, "pretrained_backbone", True))
        if freeze_backbone is None:
            freeze_backbone = bool(getattr(cfg, "freeze_backbone", True))
        self.fusion_engine = ImageMambaFusion(
            embed_dim=embed_dim,
            proprio_dim=lowdim_dim,
            pretrained=pretrained_backbone,
            freeze_backbone=freeze_backbone,
            backbone_weights=backbone_weights,
            image_hw=image_hw,
        )

    def compute_loss_at_indices(
        self,
        x_fused_seq: torch.Tensor,
        gt_actions: torch.Tensor,
        supervision_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the original action loss at selected full-history timesteps.

        Mamba still processes every frame from the start of each episode.  We
        only subsample which valid timesteps instantiate the expensive 5-way
        IMLE U-Net and symplectic-bridge supervision.  Uniform random indices
        therefore approximate the original per-frame objective without
        replacing full episode history with short windows.
        """

        if x_fused_seq.ndim != 3 or gt_actions.ndim != 4:
            raise ValueError(
                f"Expected fused [B,L,D] and actions [B,L,F,A], got "
                f"{tuple(x_fused_seq.shape)} and {tuple(gt_actions.shape)}"
            )
        batch_size, sequence_length, _ = x_fused_seq.shape
        if supervision_indices.ndim != 2 or supervision_indices.shape[0] != batch_size:
            raise ValueError(
                f"Expected supervision_indices [B,N], got {tuple(supervision_indices.shape)}"
            )
        if supervision_indices.dtype != torch.long:
            supervision_indices = supervision_indices.long()
        if supervision_indices.numel() == 0:
            raise ValueError("At least one supervision timestep is required")
        if supervision_indices.min() < 0 or supervision_indices.max() >= sequence_length:
            raise IndexError("A supervision timestep is outside the padded sequence")

        # Full-history Mamba pass happens before any supervision subsampling.
        mamba_full = self.forward_features(x_fused_seq)
        batch_offsets = (
            torch.arange(batch_size, device=x_fused_seq.device, dtype=torch.long)[:, None]
            * sequence_length
        )
        flat_indices = (supervision_indices + batch_offsets).reshape(-1)
        num_supervised = supervision_indices.shape[1]
        num_rows = flat_indices.numel()

        mamba_cond = mamba_full.reshape(batch_size * sequence_length, -1).index_select(
            0, flat_indices
        )
        fused_cond = x_fused_seq.reshape(batch_size * sequence_length, -1).index_select(
            0, flat_indices
        )
        selected_actions = gt_actions.reshape(
            batch_size * sequence_length, self.future_steps, self.action_dim
        ).index_select(0, flat_indices)

        # Stage A: unchanged 5-sample IMLE nearest-mode supervision.
        with torch.no_grad():
            z_samples = torch.randn(
                num_rows,
                self.num_imle_samples,
                self.action_dim,
                self.future_steps,
                device=x_fused_seq.device,
            )
            mamba_expanded = mamba_cond[:, None].expand(
                num_rows, self.num_imle_samples, -1
            ).reshape(num_rows * self.num_imle_samples, -1)
            fused_expanded = fused_cond[:, None].expand(
                num_rows, self.num_imle_samples, -1
            ).reshape(num_rows * self.num_imle_samples, -1)
            z_flat = z_samples.reshape(
                num_rows * self.num_imle_samples, self.action_dim, self.future_steps
            )
            generated = self.imle_generator(mamba_expanded, z_flat, fused_expanded).view(
                num_rows, self.num_imle_samples, self.future_steps, self.action_dim
            )
            distances = F.mse_loss(
                generated,
                selected_actions[:, None].expand_as(generated),
                reduction="none",
            ).mean(dim=(2, 3))
            best_mode = distances.argmin(dim=1)

        best_z = z_samples[torch.arange(num_rows, device=x_fused_seq.device), best_mode]
        q_initial = self.imle_generator(mamba_cond, best_z, fused_cond)
        loss_imle = F.mse_loss(
            q_initial,
            selected_actions.reshape(num_rows, -1),
            reduction="none",
        ).mean(dim=-1)

        # Stage B: unchanged symplectic bridge/force objective.
        q_0 = q_initial.detach()
        q_1 = selected_actions.reshape(num_rows, -1)
        time = torch.rand(num_rows, device=x_fused_seq.device)
        time_column = time[:, None]
        q_target, velocity_target, acceleration_target = self.cubic_spline(
            time_column, q_0, q_1
        )
        sigma_peak = 0.03
        sigma = 16.0 * sigma_peak * ((time_column * (1.0 - time_column)) ** 2)
        sigma_dot = 16.0 * sigma_peak * (
            2.0 * time_column * (1.0 - time_column) * (1.0 - 2.0 * time_column)
        )
        noise = torch.randn_like(q_target)
        q_noisy = q_target + sigma * noise
        velocity_noisy = velocity_target + sigma_dot * noise
        acceleration_pred = self.sb_head(
            q_noisy[:, None], velocity_noisy[:, None], time, fused_cond[:, None]
        ).squeeze(1)
        force_target = (
            acceleration_target
            + 4.0 * (q_target - q_noisy)
            + 4.0 * (velocity_target - velocity_noisy)
        )
        loss_force = F.mse_loss(
            acceleration_pred, force_target, reduction="none"
        ).mean(dim=-1)
        total = loss_imle + 0.1 * loss_force
        return total.view(batch_size, num_supervised)
