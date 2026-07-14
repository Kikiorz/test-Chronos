"""Chronos with a single RGB image encoder and the original EE action model.

Only the observation encoder changes relative to ``policy/Chronos``.  The
Mamba history model, IMLE generator, symplectic action head, 16-D dual-arm EE
representation, and 16-step prediction horizon are inherited unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


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
    """RMBench RGB V1 defaults: one RGB camera and 16-D EE control."""

    def __init__(self):
        super().__init__()
        self.camera_names = ["head_camera"]
        self.embed_dim = 1024
        self.d_model = 1024
        self.lowdim_dim = 16
        self.action_dim = 16
        self.future_steps = 16
        self.num_blocks = 6
        self.pretrained_backbone = True
        self.freeze_backbone = True


class ImageMambaFusion(nn.Module):
    """Frozen ResNet18 plus trainable spatial/proprioceptive fusion.

    The adaptive pool keeps the module independent of a hard-coded camera
    resolution.  RMBench's native 240x320 images are used without an upsample.
    Inputs may be NCHW or NHWC and either uint8/[0,255] or float/[0,1].
    """

    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        embed_dim: int = 1024,
        proprio_dim: int = 16,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        spatial_pool: Sequence[int] = (4, 5),
    ):
        super().__init__()
        if embed_dim != 1024:
            raise ValueError(
                "The inherited Chronos action heads expect a 1024-D fused feature; "
                f"got embed_dim={embed_dim}."
            )
        self.embed_dim = int(embed_dim)
        self.proprio_dim = int(proprio_dim)
        self.freeze_backbone = bool(freeze_backbone)

        if hasattr(models, "ResNet18_Weights"):
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
        else:  # torchvision < 0.13
            resnet = models.resnet18(pretrained=pretrained)
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-2])

        if self.freeze_backbone:
            self.vision_backbone.requires_grad_(False)
            # Keep pretrained BatchNorm running statistics fixed as well.
            self.vision_backbone.eval()

        pool_h, pool_w = (int(spatial_pool[0]), int(spatial_pool[1]))
        self.visual_adapter = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((pool_h, pool_w)),
            nn.Flatten(1),
            nn.Linear(128 * pool_h * pool_w, 512),
            nn.LayerNorm(512),
            nn.SiLU(inplace=True),
        )
        self.proprio_projector = nn.Sequential(
            nn.Linear(self.proprio_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 512),
            nn.LayerNorm(512),
        )
        self.fusion_norm = nn.LayerNorm(1024)

        self.register_buffer(
            "image_mean", torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor(self._IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.vision_backbone.eval()
        return self

    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected a 4-D image batch, got shape={tuple(image.shape)}")
        if image.shape[1] != 3 and image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        if image.shape[1] != 3:
            raise ValueError(f"Expected three RGB channels, got shape={tuple(image.shape)}")

        integer_input = not image.is_floating_point()
        image = image.to(dtype=self.image_mean.dtype)
        # Training uses uint8, which avoids a GPU max/synchronization per
        # chunk.  Deployment may supply either [0,1] or float [0,255].
        if integer_input or (image.numel() and image.detach().amax() > 1.5):
            image = image / 255.0
        return (image - self.image_mean) / self.image_std

    def forward(self, image: torch.Tensor, proprio_embed: torch.Tensor) -> torch.Tensor:
        if proprio_embed.ndim != 2 or proprio_embed.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"Expected proprioception [N,{self.proprio_dim}], "
                f"got {tuple(proprio_embed.shape)}"
            )
        image = self._prepare_image(image)
        if self.freeze_backbone:
            with torch.no_grad():
                feature_map = self.vision_backbone(image)
        else:
            feature_map = self.vision_backbone(image)
        visual_feature = self.visual_adapter(feature_map)
        proprio_feature = self.proprio_projector(proprio_embed)
        return self.fusion_norm(torch.cat([visual_feature, proprio_feature], dim=-1))


class MambaPolicy(_PointCloudMambaPolicy):
    """Original Chronos policy with only ``fusion_engine`` replaced by RGB."""

    def __init__(
        self,
        camera_names,
        embed_dim: int = 1024,
        lowdim_dim: int = 16,
        d_model: int = 1024,
        action_dim: int = 16,
        num_blocks: int = 6,
        block_cfg=None,
        mamba_cfg=None,
        future_steps: int = 16,
        pretrained_backbone: bool | None = None,
        freeze_backbone: bool | None = None,
    ):
        if list(camera_names) != ["head_camera"]:
            raise ValueError(
                "Chronos_RGB V1 intentionally uses exactly one camera: ['head_camera']; "
                f"got {list(camera_names)}"
            )
        if lowdim_dim != 16 or action_dim != 16 or future_steps != 16:
            raise ValueError(
                "Chronos_RGB V1 is the controlled RGB-vs-point-cloud experiment and "
                "therefore requires lowdim_dim=16, action_dim=16, future_steps=16."
            )
        if embed_dim != 1024 or d_model != 1024:
            raise ValueError("Chronos_RGB V1 requires embed_dim=d_model=1024.")

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
        if pretrained_backbone is None:
            pretrained_backbone = bool(getattr(cfg, "pretrained_backbone", True))
        if freeze_backbone is None:
            freeze_backbone = bool(getattr(cfg, "freeze_backbone", True))
        self.fusion_engine = ImageMambaFusion(
            embed_dim=embed_dim,
            proprio_dim=lowdim_dim,
            pretrained=pretrained_backbone,
            freeze_backbone=freeze_backbone,
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
