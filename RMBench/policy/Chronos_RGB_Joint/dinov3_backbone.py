"""DINOv3 ViT-B/16 image encoding shared by training and deployment.

The official DINOv3 weights are gated and are deliberately kept outside Git.
This module accepts either a standalone timm state dict or a larger checkpoint
whose keys use a known backbone prefix, and always validates all 162 backbone
entries strictly before use.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F


DINOV3_MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"
DINOV3_FEATURE_DIM = 768
DINOV3_PATCH_SIZE = 16
# 1.4x the native RMBench 240x320 frame while preserving 4:3 and patch alignment.
DINOV3_IMAGE_HW = (336, 448)
DINOV3_POOL_HW = (4, 5)
DINOV3_CACHE_TOKENS = 1 + DINOV3_POOL_HW[0] * DINOV3_POOL_HW[1]
DINOV3_CACHE_FORMAT_VERSION = 3
DINOV3_CACHE_DTYPE = "float32"
DINOV3_CACHE_AMP_DTYPE = "none"
DINOV3_CACHE_BATCH_IMAGES = 1
DINOV3_CACHE_DEVICE_TYPE = "cuda"
DINOV3_RESIZE_MODE = "bicubic"
RMBENCH_RGB_HW = (240, 320)
DINOV3_RGB_SCALE = "uint8_or_0_255_to_0_1"
DINOV3_NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
DINOV3_NORMALIZATION_STD = (0.229, 0.224, 0.225)
FLOAT32_NUMERICS = {
    "matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
}


def dinov3_cache_contract(
    camera_name: str,
    weights_sha256: str,
    *,
    batch_images: int = DINOV3_CACHE_BATCH_IMAGES,
    amp_dtype: str = DINOV3_CACHE_AMP_DTYPE,
    device_type: str = DINOV3_CACHE_DEVICE_TYPE,
) -> dict[str, object]:
    """Return the single authoritative frozen-feature cache contract."""

    if not camera_name:
        raise ValueError("camera_name must be non-empty")
    if len(weights_sha256) != 64:
        raise ValueError("weights_sha256 must contain 64 hexadecimal characters")
    try:
        int(weights_sha256, 16)
    except ValueError as exc:
        raise ValueError("weights_sha256 is not hexadecimal") from exc
    if batch_images <= 0:
        raise ValueError("batch_images must be positive")
    return {
        "format_version": DINOV3_CACHE_FORMAT_VERSION,
        "model_name": DINOV3_MODEL_NAME,
        "camera_name": camera_name,
        "source_image_hw": list(RMBENCH_RGB_HW),
        "image_hw": list(DINOV3_IMAGE_HW),
        "patch_size": DINOV3_PATCH_SIZE,
        "pool_hw": list(DINOV3_POOL_HW),
        "feature_dim": DINOV3_FEATURE_DIM,
        "cache_tokens": DINOV3_CACHE_TOKENS,
        "token_order": "CLS_then_adaptive_avg_pool_row_major_height_width",
        "dtype": DINOV3_CACHE_DTYPE,
        "amp_dtype": amp_dtype,
        "encoder_batch_images": batch_images,
        "device_type": device_type,
        "resize": {"mode": DINOV3_RESIZE_MODE, "antialias": True},
        "rgb_scale": DINOV3_RGB_SCALE,
        "normalization_mean": list(DINOV3_NORMALIZATION_MEAN),
        "normalization_std": list(DINOV3_NORMALIZATION_STD),
        "float32_numerics": dict(FLOAT32_NUMERICS),
        "weights_sha256": weights_sha256,
    }


def configure_float32_numerics() -> None:
    """Use the same strict FP32 math policy for cache, training, and deploy."""

    torch.set_float32_matmul_precision(str(FLOAT32_NUMERICS["matmul_precision"]))
    torch.backends.cuda.matmul.allow_tf32 = bool(
        FLOAT32_NUMERICS["cuda_matmul_allow_tf32"]
    )
    torch.backends.cudnn.allow_tf32 = bool(FLOAT32_NUMERICS["cudnn_allow_tf32"])


def _load_tensor_mapping(path: str | Path) -> dict[str, torch.Tensor]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"DINOv3 weights do not exist: {resolved}")
    if resolved.suffix == ".safetensors":
        from safetensors.torch import load_file

        loaded: Any = load_file(str(resolved), device="cpu")
    else:
        try:
            loaded = torch.load(resolved, map_location="cpu", weights_only=True)
        except TypeError:  # torch < 2.0
            loaded = torch.load(resolved, map_location="cpu")
    while isinstance(loaded, Mapping):
        nested = next(
            (
                loaded[key]
                for key in ("state_dict", "model", "teacher", "backbone")
                if key in loaded and isinstance(loaded[key], Mapping)
            ),
            None,
        )
        if nested is None:
            break
        loaded = nested
    if not isinstance(loaded, Mapping):
        raise TypeError(f"DINOv3 weight file is not a tensor mapping: {resolved}")
    result = {
        str(key): value
        for key, value in loaded.items()
        if isinstance(key, str) and torch.is_tensor(value)
    }
    if not result:
        raise ValueError(f"DINOv3 weight mapping is empty: {resolved}")
    return result


def _select_strict_backbone_state(
    loaded: Mapping[str, torch.Tensor], reference_keys: set[str]
) -> dict[str, torch.Tensor]:
    prefixes = (
        "",
        "module.",
        "model.",
        "backbone.",
        "backbone.model.",
        "model.backbone.model.",
        "policy.fusion_engine.image_encoder.vision_backbone.",
    )
    for prefix in prefixes:
        candidate = {
            key[len(prefix) :]: value
            for key, value in loaded.items()
            if key.startswith(prefix) and key[len(prefix) :] in reference_keys
        }
        if set(candidate) == reference_keys:
            return candidate
    best_prefix = max(
        prefixes,
        key=lambda prefix: sum(
            key.startswith(prefix) and key[len(prefix) :] in reference_keys
            for key in loaded
        ),
    )
    candidate_keys = {
        key[len(best_prefix) :]
        for key in loaded
        if key.startswith(best_prefix) and key[len(best_prefix) :] in reference_keys
    }
    missing = sorted(reference_keys - candidate_keys)
    raise RuntimeError(
        "DINOv3 weights are incomplete or use an unsupported namespace: "
        f"matched={len(candidate_keys)}/{len(reference_keys)}, "
        f"prefix={best_prefix!r}, missing={missing[:8]}"
    )


def build_dinov3_vitb16(
    *,
    pretrained: bool,
    weights_path: str | Path | None,
) -> nn.Module:
    """Build timm DINOv3-B/16 and strictly load an optional external state."""

    try:
        import timm
    except ImportError as exc:  # pragma: no cover - environment error
        raise ImportError(
            "Chronos_RGB_Joint requires timm>=1.0.20. Install it in the active environment."
        ) from exc

    # Explicit files avoid silently depending on a Hugging Face login.  When
    # no path is supplied, timm's official pretrained route remains available
    # for users whose account has accepted the DINOv3 license.
    model = timm.create_model(
        DINOV3_MODEL_NAME,
        pretrained=bool(pretrained and weights_path is None),
        num_classes=0,
        global_pool="",
    )
    if weights_path is not None:
        loaded = _load_tensor_mapping(weights_path)
        state = _select_strict_backbone_state(loaded, set(model.state_dict()))
        model.load_state_dict(state, strict=True)
    return model


class DINOv3ImageEncoder(nn.Module):
    """Return CLS plus a coarse spatial grid from frozen DINOv3 patch tokens.

    The output is ``[N, 21, 768]``: one CLS token followed by a row-major 4x5
    adaptive-average grid.  Caching these FP32 batch-1 tokens preserves the
    trainable adapter input while avoiding a frozen ViT pass over all 51k
    frames on every epoch.
    """

    def __init__(
        self,
        *,
        image_hw: Sequence[int] = DINOV3_IMAGE_HW,
        pool_hw: Sequence[int] = DINOV3_POOL_HW,
        pretrained: bool = True,
        weights_path: str | Path | None = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        configure_float32_numerics()
        if len(image_hw) != 2 or any(int(value) <= 0 for value in image_hw):
            raise ValueError(f"image_hw must be positive (height,width), got {image_hw}")
        if len(pool_hw) != 2 or any(int(value) <= 0 for value in pool_hw):
            raise ValueError(f"pool_hw must be positive (height,width), got {pool_hw}")
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.pool_hw = (int(pool_hw[0]), int(pool_hw[1]))
        if any(value % DINOV3_PATCH_SIZE for value in self.image_hw):
            raise ValueError(
                f"DINOv3 image dimensions must be multiples of {DINOV3_PATCH_SIZE}: "
                f"{self.image_hw}"
            )
        self.freeze = bool(freeze)
        self.vision_backbone = build_dinov3_vitb16(
            pretrained=pretrained,
            weights_path=weights_path,
        )
        if int(getattr(self.vision_backbone, "num_features", -1)) != DINOV3_FEATURE_DIM:
            raise RuntimeError("Loaded backbone is not DINOv3 ViT-B/16 (expected 768 features)")
        patch_size = tuple(int(value) for value in self.vision_backbone.patch_embed.patch_size)
        if patch_size != (DINOV3_PATCH_SIZE, DINOV3_PATCH_SIZE):
            raise RuntimeError(f"Unexpected DINOv3 patch size: {patch_size}")
        if self.freeze:
            self.vision_backbone.requires_grad_(False)
            self.vision_backbone.eval()

        self.register_buffer(
            "image_mean",
            torch.tensor(DINOV3_NORMALIZATION_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(DINOV3_NORMALIZATION_STD).view(1, 3, 1, 1),
            persistent=False,
        )

    @property
    def cache_tokens(self) -> int:
        return 1 + self.pool_hw[0] * self.pool_hw[1]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.vision_backbone.eval()
        return self

    def prepare_images(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected a 4-D image batch, got {tuple(image.shape)}")
        if image.shape[1] != 3 and image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        if image.shape[1] != 3:
            raise ValueError(f"Expected RGB images, got {tuple(image.shape)}")
        integer_input = not image.is_floating_point()
        image = image.to(dtype=self.image_mean.dtype)
        if integer_input or (image.numel() and image.detach().amax() > 1.5):
            image = image / 255.0
        if tuple(image.shape[-2:]) != self.image_hw:
            image = F.interpolate(
                image,
                size=self.image_hw,
                # Match timm's pretrained DINOv3 preprocessing contract.
                mode=DINOV3_RESIZE_MODE,
                align_corners=False,
                antialias=True,
            )
        return (image - self.image_mean) / self.image_std

    def _tokens_from_output(self, output: Any) -> tuple[torch.Tensor, torch.Tensor]:
        grid_h = self.image_hw[0] // DINOV3_PATCH_SIZE
        grid_w = self.image_hw[1] // DINOV3_PATCH_SIZE
        num_patches = grid_h * grid_w
        if isinstance(output, Mapping):
            patch = output.get("x_norm_patchtokens")
            cls = output.get("x_norm_clstoken")
            if patch is None:
                raise KeyError("DINOv3 output is missing x_norm_patchtokens")
            if cls is None:
                raise KeyError("DINOv3 output is missing x_norm_clstoken")
        elif torch.is_tensor(output):
            if output.ndim != 3 or output.shape[1] < num_patches + 1:
                raise ValueError(f"Unexpected DINOv3 token tensor: {tuple(output.shape)}")
            cls = output[:, 0]
            # This is robust to CLS + four register/storage tokens.
            patch = output[:, -num_patches:]
        else:
            raise TypeError(f"Unsupported DINOv3 output type: {type(output).__name__}")
        if patch.shape[1:] != (num_patches, DINOV3_FEATURE_DIM):
            raise ValueError(f"Unexpected DINOv3 patch tokens: {tuple(patch.shape)}")
        if cls.shape[-1] != DINOV3_FEATURE_DIM:
            raise ValueError(f"Unexpected DINOv3 CLS token: {tuple(cls.shape)}")
        return cls, patch

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = self.prepare_images(image)
        if self.freeze:
            with torch.no_grad():
                output = self.vision_backbone.forward_features(image)
        else:
            output = self.vision_backbone.forward_features(image)
        cls, patch = self._tokens_from_output(output)
        grid_h = self.image_hw[0] // DINOV3_PATCH_SIZE
        grid_w = self.image_hw[1] // DINOV3_PATCH_SIZE
        patch_map = patch.reshape(
            patch.shape[0], grid_h, grid_w, DINOV3_FEATURE_DIM
        ).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(patch_map, self.pool_hw)
        pooled = pooled.flatten(2).transpose(1, 2)
        tokens = torch.cat([cls.unsqueeze(1), pooled], dim=1)
        expected = (self.cache_tokens, DINOV3_FEATURE_DIM)
        if tuple(tokens.shape[1:]) != expected:
            raise RuntimeError(f"DINOv3 cache shape {tuple(tokens.shape[1:])} != {expected}")
        if not torch.isfinite(tokens).all():
            raise FloatingPointError("DINOv3 produced non-finite image features")
        return tokens
