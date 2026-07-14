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
    """Official real-world RGB encoder with RMBench's 16-D EE interface."""

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
        self.visual_architecture = "official_realworld"
        self.backbone_trainable = "none"


class ImageMambaFusion(nn.Module):
    """The repository's official real-world ``ImageMambaFusion`` contract.

    The train and deployment paths both pass raw RGB to this module.  Keeping
    ImageNet normalization here makes the two paths mathematically identical:
    training may use uint8 ``[0, 255]`` and deployment may use float ``[0, 1]``.
    The official flatten projector fixes the input at 480x640; the ResNet trunk
    then produces 15x20 and the stride-2 adapter produces 8x10.

    ``backbone_trainable``, ``freeze_backbone``, ``visual_architecture`` and
    ``spatial_pool`` are accepted only so older controller/trainer call sites
    can construct a fresh official model.  They cannot enable a legacy V1/V2
    branch or unfreeze any part of the ResNet trunk.
    """

    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        embed_dim: int = 1024,
        proprio_dim: int = 16,
        pretrained: bool = True,
        backbone_trainable: str | None = None,
        freeze_backbone: bool | None = None,
        visual_architecture: str | None = None,
        spatial_pool: Sequence[int] | None = None,
    ):
        super().__init__()
        if embed_dim != 1024:
            raise ValueError(
                "The inherited Chronos action heads expect a 1024-D fused feature; "
                f"got embed_dim={embed_dim}."
            )
        self.embed_dim = int(embed_dim)
        self.proprio_dim = int(proprio_dim)
        if self.proprio_dim != 16:
            raise ValueError(
                "The RMBench RGB policy requires a 16-D dual-arm EE state; "
                f"got proprio_dim={self.proprio_dim}."
            )

        # Backward-compatible constructor surface only.  The fresh checkpoint
        # has one architecture and its ImageNet trunk is always frozen.
        del backbone_trainable, freeze_backbone, visual_architecture, spatial_pool
        self.backbone_trainable = "none"
        self.freeze_backbone = True
        self.visual_architecture = "official_realworld"

        if hasattr(models, "ResNet18_Weights"):
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
        else:  # torchvision < 0.13
            resnet = models.resnet18(pretrained=pretrained)
        self.vision_net = nn.Sequential(*list(resnet.children())[:-2])
        self._replace_bn_with_gn(self.vision_net)
        self.vision_net.requires_grad_(False)
        self.vision_net.eval()

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

        self.register_buffer(
            "image_mean", torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor(self._IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    @property
    def vision_backbone(self) -> nn.Sequential:
        """Read-only alias for older trainer diagnostics (not a state key)."""

        return self.vision_net

    def _replace_bn_with_gn(self, module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(
                    module,
                    name,
                    nn.GroupNorm(num_groups=32, num_channels=child.num_features),
                )
            else:
                self._replace_bn_with_gn(child)

    def train(self, mode: bool = True):
        super().train(mode)
        # The official real-world code extracts ResNet features under no_grad.
        # Explicitly pinning the trunk to eval also makes that guarantee robust
        # when Lightning recursively calls policy.train().
        self.vision_net.eval()
        return self

    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or tuple(image.shape[1:]) != (3, 480, 640):
            raise ValueError(
                "Expected NCHW RGB at the official 480x640 resolution, "
                f"got shape={tuple(image.shape)}"
            )
        if image.shape[0] == 0:
            raise ValueError("The image batch must contain at least one frame")
        if image.dtype == torch.uint8:
            # uint8 already guarantees [0,255].  Avoid a GPU min/max
            # synchronization for every training chunk.
            image = image.to(dtype=self.image_mean.dtype) / 255.0
        elif image.is_floating_point():
            image = image.to(dtype=self.image_mean.dtype)
            if not torch.isfinite(image).all():
                raise ValueError("RGB input contains NaN or infinity")
            if image.amin() < 0 or image.amax() > 1:
                raise ValueError("Floating-point RGB input must already be in [0, 1]")
        else:
            raise TypeError("Integer RGB input must use torch.uint8")
        return (image - self.image_mean) / self.image_std

    def forward(self, image: torch.Tensor, proprio_embed: torch.Tensor) -> torch.Tensor:
        if proprio_embed.ndim != 2 or proprio_embed.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"Expected proprioception [N,{self.proprio_dim}], "
                f"got {tuple(proprio_embed.shape)}"
            )
        image = self._prepare_image(image)
        with torch.no_grad():
            feature_map = self.vision_net(image)
        if tuple(feature_map.shape[1:]) != (512, 15, 20):
            raise RuntimeError(
                "Official RGB trunk must produce [N,512,15,20] before the "
                f"8x10 visual adapter, got {tuple(feature_map.shape)}"
            )
        visual_feature = self.visual_adapter(feature_map)
        proprio_feature = self.proprio_projector(proprio_embed)
        return self.fusion_proj(torch.cat([visual_feature, proprio_feature], dim=-1))


class MambaPolicy(_PointCloudMambaPolicy):
    """Original RMBench Chronos dynamics with the official RGB fusion engine."""

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
        backbone_trainable: str | None = None,
        freeze_backbone: bool | None = None,
        visual_architecture: str | None = None,
    ):
        if list(camera_names) != ["head_camera"]:
            raise ValueError(
                "Chronos_RGB intentionally uses exactly one camera: ['head_camera']; "
                f"got {list(camera_names)}"
            )
        if lowdim_dim != 16 or action_dim != 16 or future_steps != 16:
            raise ValueError(
                "Chronos_RGB is the controlled RGB-vs-point-cloud experiment and "
                "therefore requires lowdim_dim=16, action_dim=16, future_steps=16."
            )
        if embed_dim != 1024 or d_model != 1024:
            raise ValueError("Chronos_RGB requires embed_dim=d_model=1024.")

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
        if backbone_trainable is None:
            backbone_trainable = str(getattr(cfg, "backbone_trainable", "none"))
        if visual_architecture is None:
            visual_architecture = str(
                getattr(cfg, "visual_architecture", "official_realworld")
            )
        self.fusion_engine = ImageMambaFusion(
            embed_dim=embed_dim,
            proprio_dim=lowdim_dim,
            pretrained=pretrained_backbone,
            backbone_trainable=backbone_trainable,
            freeze_backbone=freeze_backbone,
            visual_architecture=visual_architecture,
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
