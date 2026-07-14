"""Official Chronos real-world ResNet-18 RGB preprocessing and trunk.

This module is the single RGB contract shared by RMBench training and
deployment.  It mirrors ``real_wolrd`` exactly where modality matters:

* the input is RGB (not OpenCV BGR) at the native RMBench 240x320 size;
* OpenCV ``INTER_AREA`` resizes it to 480x640;
* uint8 pixels are converted to FP32 in [0, 1] and ImageNet-normalized;
* torchvision's pretrained ResNet-18 is truncated before avgpool/fc;
* every BatchNorm2d is replaced by a freshly initialized GroupNorm(32, C).

The official ResNet trunk is frozen.  Its ``train()`` override keeps it in
evaluation mode even when a containing policy is switched to training mode.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models


RMBENCH_RGB_HW = (240, 320)
RESNET18_IMAGE_HW = (480, 640)
RESNET18_FEATURE_HW = (15, 20)
RESNET18_FEATURE_DIM = 512
RESNET18_MODEL_NAME = "torchvision.models.resnet18:ResNet18_Weights.DEFAULT"
# Backward-readable alias for contracts that call this field a model id.
RESNET18_MODEL_ID = RESNET18_MODEL_NAME
RESNET18_RESIZE_MODE = "opencv_inter_area"
RESNET18_NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
RESNET18_NORMALIZATION_STD = (0.229, 0.224, 0.225)
RESNET18_RGB_SCALE = "uint8_to_float32_div255"
RESNET18_WEIGHTS_FILENAME = "resnet18-f37072fd.pth"
RESNET18_WEIGHTS_SHA256 = (
    "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
)
FLOAT32_NUMERICS = {
    "matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    # The released scripts request precision=32 and otherwise retain the
    # PyTorch/cuDNN default. On the supported torch build that means TF32 is
    # enabled for cuDNN convolutions, but disabled for generic CUDA matmul.
    "cudnn_allow_tf32": True,
}

_NORMALIZATION_MEAN_ARRAY = np.asarray(
    RESNET18_NORMALIZATION_MEAN, dtype=np.float32
)[:, None, None]
_NORMALIZATION_STD_ARRAY = np.asarray(
    RESNET18_NORMALIZATION_STD, dtype=np.float32
)[:, None, None]
_NORMALIZED_GLOBAL_MIN = min(
    (0.0 - mean) / std
    for mean, std in zip(
        RESNET18_NORMALIZATION_MEAN, RESNET18_NORMALIZATION_STD, strict=True
    )
)
_NORMALIZED_GLOBAL_MAX = max(
    (1.0 - mean) / std
    for mean, std in zip(
        RESNET18_NORMALIZATION_MEAN, RESNET18_NORMALIZATION_STD, strict=True
    )
)
_RANGE_TOLERANCE = 1.0e-5


def configure_float32_numerics() -> None:
    """Apply the FP32 math policy used by RGB training and deployment."""

    torch.set_float32_matmul_precision(str(FLOAT32_NUMERICS["matmul_precision"]))
    torch.backends.cuda.matmul.allow_tf32 = bool(
        FLOAT32_NUMERICS["cuda_matmul_allow_tf32"]
    )
    torch.backends.cudnn.allow_tf32 = bool(FLOAT32_NUMERICS["cudnn_allow_tf32"])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_resnet18_weights(path: str | Path) -> str:
    """Verify the official torchvision ResNet-18 file and return its SHA256."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"ResNet-18 weights do not exist: {resolved}")
    actual = _sha256_file(resolved)
    if actual != RESNET18_WEIGHTS_SHA256:
        raise ValueError(
            "ResNet-18 weights SHA256 mismatch: "
            f"expected={RESNET18_WEIGHTS_SHA256}, actual={actual}, path={resolved}"
        )
    return actual


def preprocess_rgb_numpy(image: np.ndarray) -> np.ndarray:
    """Convert one native RMBench RGB uint8 frame to normalized CHW FP32.

    Channel order is deliberately not guessed or converted.  Callers must pass
    RGB.  This matters because OpenCV's resize preserves channel positions even
    though its image-file convenience APIs conventionally expose BGR.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy.ndarray, got {type(image).__name__}")
    expected_shape = (*RMBENCH_RGB_HW, 3)
    if image.shape != expected_shape:
        raise ValueError(f"RGB image shape must be {expected_shape}, got {image.shape}")
    if image.dtype != np.uint8:
        raise TypeError(f"RGB image dtype must be uint8, got {image.dtype}")
    # Explicit checks make the raw input contract auditable.  All uint8 values
    # are finite and in range by construction, but keeping the checks here
    # prevents a future dtype relaxation from silently weakening the contract.
    raw_min = int(image.min())
    raw_max = int(image.max())
    if raw_min < 0 or raw_max > 255:
        raise ValueError(f"RGB pixels must be in [0,255], got [{raw_min},{raw_max}]")
    if not np.isfinite(image).all():
        raise FloatingPointError("RGB image contains non-finite pixels")

    contiguous = np.ascontiguousarray(image)
    target_h, target_w = RESNET18_IMAGE_HW
    resized = cv2.resize(
        contiguous,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA,
    )
    scaled_hwc = resized.astype(np.float32) / np.float32(255.0)
    scaled_chw = np.transpose(scaled_hwc, (2, 0, 1))
    normalized = (scaled_chw - _NORMALIZATION_MEAN_ARRAY) / _NORMALIZATION_STD_ARRAY
    normalized = np.ascontiguousarray(normalized, dtype=np.float32)

    expected_output_shape = (3, *RESNET18_IMAGE_HW)
    if normalized.shape != expected_output_shape:
        raise RuntimeError(
            f"Preprocessed RGB shape {normalized.shape} != {expected_output_shape}"
        )
    if normalized.dtype != np.float32:
        raise RuntimeError(f"Preprocessed RGB dtype is not float32: {normalized.dtype}")
    if not np.isfinite(normalized).all():
        raise FloatingPointError("Preprocessed RGB image contains non-finite values")
    output_min = float(normalized.min())
    output_max = float(normalized.max())
    if (
        output_min < _NORMALIZED_GLOBAL_MIN - _RANGE_TOLERANCE
        or output_max > _NORMALIZED_GLOBAL_MAX + _RANGE_TOLERANCE
    ):
        raise RuntimeError(
            "Preprocessed RGB image lies outside the ImageNet-normalized range: "
            f"[{output_min},{output_max}]"
        )
    return normalized


# Descriptive alias retained for callers that prefer the encoder-specific name.
preprocess_resnet18_rgb = preprocess_rgb_numpy


def _load_strict_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        loaded: Any = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0
        loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, Mapping) and "state_dict" in loaded:
        loaded = loaded["state_dict"]
    if not isinstance(loaded, Mapping):
        raise TypeError(f"ResNet-18 weights are not a state dict: {path}")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in loaded.items()):
        raise TypeError(f"ResNet-18 state dict contains non-tensor entries: {path}")
    return loaded


def _replace_batch_norm_with_group_norm(module: nn.Module) -> int:
    replaced = 0
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            # This is intentionally a new GroupNorm.  Copying BN affine values
            # or running statistics would differ from the official real-world
            # implementation.
            setattr(
                module,
                name,
                nn.GroupNorm(num_groups=32, num_channels=child.num_features),
            )
            replaced += 1
        else:
            replaced += _replace_batch_norm_with_group_norm(child)
    return replaced


class OfficialResNet18Trunk(nn.Sequential):
    """Frozen official Chronos ResNet-18 trunk returning ``[N,512,15,20]``."""

    def __init__(
        self,
        pretrained: bool = True,
        weights_path: str | Path | None = None,
        freeze: bool = True,
    ) -> None:
        configure_float32_numerics()

        if weights_path is not None:
            resolved_weights = Path(weights_path).expanduser().resolve()
            weights_sha256 = verify_resnet18_weights(resolved_weights)
            resnet = models.resnet18(weights=None)
            state_dict = _load_strict_state_dict(resolved_weights)
            resnet.load_state_dict(state_dict, strict=True)
        else:
            resolved_weights = None
            weights_sha256 = RESNET18_WEIGHTS_SHA256 if pretrained else None
            try:
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                resnet = models.resnet18(weights=weights)
            except AttributeError:  # torchvision < 0.13
                resnet = models.resnet18(pretrained=pretrained)

        # Sequential inheritance preserves the official checkpoint namespace:
        # ``vision_net.0...vision_net.7`` rather than adding a wrapper level.
        super().__init__(*list(resnet.children())[:-2])
        self.freeze = bool(freeze)
        self.weights_path = str(resolved_weights) if resolved_weights is not None else None
        self.weights_sha256 = weights_sha256
        replaced = _replace_batch_norm_with_group_norm(self)
        if replaced != 20:
            raise RuntimeError(f"Expected to replace 20 ResNet-18 BatchNorm layers, got {replaced}")
        if any(isinstance(module, nn.BatchNorm2d) for module in self.modules()):
            raise RuntimeError("BatchNorm2d remains in the official ResNet-18 trunk")
        if self.freeze:
            self.requires_grad_(False)
            super().train(False)

    def train(self, mode: bool = True):
        if self.freeze:
            # A containing policy's ``train()`` recursively reaches this method;
            # the frozen trunk must nevertheless stay deterministic/eval-only.
            return super().train(False)
        return super().train(mode)

    @staticmethod
    def _validate_input(image: torch.Tensor) -> None:
        if not torch.is_tensor(image):
            raise TypeError(f"image must be a torch.Tensor, got {type(image).__name__}")
        expected_tail = (3, *RESNET18_IMAGE_HW)
        if image.ndim != 4 or tuple(image.shape[1:]) != expected_tail:
            raise ValueError(
                f"Normalized RGB batch shape must be [N,{expected_tail[0]},"
                f"{expected_tail[1]},{expected_tail[2]}], got {tuple(image.shape)}"
            )
        if image.shape[0] <= 0:
            raise ValueError("Normalized RGB batch must contain at least one image")
        if image.dtype != torch.float32:
            raise TypeError(f"Normalized RGB dtype must be torch.float32, got {image.dtype}")
        # aminmax avoids allocating a full-size boolean tensor for large image
        # chunks while still detecting NaN/Inf and double normalization.
        image_min, image_max = torch.aminmax(image.detach())
        if not bool(torch.isfinite(image_min).item() and torch.isfinite(image_max).item()):
            raise FloatingPointError("Normalized RGB batch contains non-finite values")
        min_value = float(image_min.item())
        max_value = float(image_max.item())
        if (
            min_value < _NORMALIZED_GLOBAL_MIN - _RANGE_TOLERANCE
            or max_value > _NORMALIZED_GLOBAL_MAX + _RANGE_TOLERANCE
        ):
            raise ValueError(
                "Normalized RGB batch lies outside the ImageNet range; "
                f"possible missing/double normalization: [{min_value},{max_value}]"
            )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self._validate_input(image)
        context = torch.no_grad() if self.freeze else nullcontext()
        with context:
            features = super().forward(image)
        expected_shape = (image.shape[0], RESNET18_FEATURE_DIM, *RESNET18_FEATURE_HW)
        if tuple(features.shape) != expected_shape:
            raise RuntimeError(
                f"ResNet-18 trunk output shape {tuple(features.shape)} != {expected_shape}"
            )
        if features.dtype != torch.float32:
            raise RuntimeError(f"ResNet-18 trunk output is not FP32: {features.dtype}")
        return features


__all__ = [
    "FLOAT32_NUMERICS",
    "OfficialResNet18Trunk",
    "RESNET18_FEATURE_DIM",
    "RESNET18_FEATURE_HW",
    "RESNET18_IMAGE_HW",
    "RESNET18_MODEL_ID",
    "RESNET18_MODEL_NAME",
    "RESNET18_NORMALIZATION_MEAN",
    "RESNET18_NORMALIZATION_STD",
    "RESNET18_RESIZE_MODE",
    "RESNET18_RGB_SCALE",
    "RESNET18_WEIGHTS_FILENAME",
    "RESNET18_WEIGHTS_SHA256",
    "RMBENCH_RGB_HW",
    "configure_float32_numerics",
    "preprocess_resnet18_rgb",
    "preprocess_rgb_numpy",
    "verify_resnet18_weights",
]
