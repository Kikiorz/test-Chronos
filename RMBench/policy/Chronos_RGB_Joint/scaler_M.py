"""Checkpoint-friendly joint/action statistics for Chronos RGB Joint."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Mapping, Union

import torch
from torch import nn


Shape = Union[int, tuple[int, ...]]


class Scaler(nn.Module):
    """Per-key z-score normalization with statistics stored in ``state_dict``."""

    def __init__(self, lowdim_dict: Mapping[str, Shape], eps: float = 1e-8):
        super().__init__()
        self.lowdim_dict = dict(lowdim_dict)
        self.eps = float(eps)
        self.mean_dict = nn.ParameterDict(
            {
                key: nn.Parameter(torch.zeros(shape, dtype=torch.float32), requires_grad=False)
                for key, shape in self.lowdim_dict.items()
            }
        )
        self.std_dict = nn.ParameterDict(
            {
                key: nn.Parameter(torch.ones(shape, dtype=torch.float32), requires_grad=False)
                for key, shape in self.lowdim_dict.items()
            }
        )
        self.min_dict = nn.ParameterDict(
            {
                key: nn.Parameter(torch.zeros(shape, dtype=torch.float32), requires_grad=False)
                for key, shape in self.lowdim_dict.items()
            }
        )
        self.max_dict = nn.ParameterDict(
            {
                key: nn.Parameter(torch.zeros(shape, dtype=torch.float32), requires_grad=False)
                for key, shape in self.lowdim_dict.items()
            }
        )

    @torch.no_grad()
    def fit(self, data_dict: Mapping[str, torch.Tensor]) -> None:
        missing = set(self.lowdim_dict) - set(data_dict)
        if missing:
            raise KeyError(f"Cannot fit scaler; missing keys: {sorted(missing)}")

        for key in self.lowdim_dict:
            data = torch.as_tensor(data_dict[key], dtype=torch.float32)
            if data.ndim == 0 or data.shape[0] < 2:
                raise ValueError(
                    f"Scaler key {key!r} needs at least two samples for the official "
                    f"sample standard deviation: shape={tuple(data.shape)}"
                )
            if not torch.isfinite(data).all():
                raise ValueError(f"Scaler key {key!r} contains NaN or Inf")
            mean = data.mean(dim=0)
            # Match the released real-world and RMBench scalers exactly:
            # torch.std's default Bessel correction (unbiased=True), eps=1e-8.
            std = data.std(dim=0, unbiased=True).clamp_min(self.eps)
            minimum = data.amin(dim=0)
            maximum = data.amax(dim=0)
            expected_shape = self.mean_dict[key].shape
            if mean.shape != expected_shape:
                raise ValueError(
                    f"Scaler key {key!r}: expected statistic shape {tuple(expected_shape)}, "
                    f"got {tuple(mean.shape)}"
                )
            self.mean_dict[key].copy_(mean)
            self.std_dict[key].copy_(std)
            self.min_dict[key].copy_(minimum)
            self.max_dict[key].copy_(maximum)
        self.validate_fitted()

    @staticmethod
    def state_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
        """Canonical SHA-256 over names, shapes, dtypes, and tensor bytes."""

        digest = hashlib.sha256()
        for key in sorted(state):
            value = state[key]
            if not isinstance(key, str) or not torch.is_tensor(value):
                raise TypeError("Scaler state must map string keys to tensors")
            tensor = value.detach().cpu().contiguous()
            header = f"{key}\0{tensor.dtype}\0{tuple(tensor.shape)}\0".encode("utf-8")
            digest.update(header)
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def fingerprint(self) -> str:
        self.validate_fitted()
        return self.state_fingerprint(self.state_dict())

    @torch.no_grad()
    def validate_fitted(self) -> None:
        for key in self.lowdim_dict:
            mean = self.mean_dict[key]
            std = self.std_dict[key]
            minimum = self.min_dict[key]
            maximum = self.max_dict[key]
            for name, value in (
                ("mean", mean),
                ("std", std),
                ("min", minimum),
                ("max", maximum),
            ):
                if not torch.isfinite(value).all():
                    raise ValueError(f"Scaler {name} for {key!r} contains NaN or Inf")
            if not torch.all(std > 0):
                raise ValueError(f"Scaler std for {key!r} must be strictly positive")
            if not torch.all(minimum <= maximum):
                raise ValueError(f"Scaler min exceeds max for {key!r}")
            tolerance = 1e-5 * (maximum - minimum).abs().clamp_min(1.0)
            if not torch.all((mean >= minimum - tolerance) & (mean <= maximum + tolerance)):
                raise ValueError(f"Scaler mean lies outside min/max for {key!r}")

    def _validate_transform_value(self, key: str, value: torch.Tensor) -> None:
        """Reject silent broadcasting or precision/device changes at the boundary."""

        if key not in self.lowdim_dict:
            raise KeyError(f"Scaler received an unknown key: {key!r}")
        if not torch.is_tensor(value):
            raise TypeError(f"Scaler value for {key!r} must be a tensor")
        statistic = self.mean_dict[key]
        expected_shape = tuple(statistic.shape)
        if (
            value.ndim < len(expected_shape)
            or tuple(value.shape[-len(expected_shape):]) != expected_shape
        ):
            raise ValueError(
                f"Scaler value for {key!r} must end in {expected_shape}; "
                f"got {tuple(value.shape)}"
            )
        if value.dtype != statistic.dtype:
            raise TypeError(
                f"Scaler value for {key!r} must use {statistic.dtype}; got {value.dtype}"
            )
        if value.device != statistic.device:
            raise RuntimeError(
                f"Scaler value for {key!r} is on {value.device}, but its statistics "
                f"are on {statistic.device}"
            )

    def normalize(self, data_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        normalized: Dict[str, torch.Tensor] = {}
        for key, value in data_dict.items():
            self._validate_transform_value(key, value)
            normalized[key] = (value - self.mean_dict[key]) / self.std_dict[key]
        return normalized

    def denormalize(self, data_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        denormalized: Dict[str, torch.Tensor] = {}
        for key, value in data_dict.items():
            self._validate_transform_value(key, value)
            denormalized[key] = value * self.std_dict[key] + self.mean_dict[key]
        return denormalized

    def save(self, filepath: str | Path) -> None:
        self.validate_fitted()
        path = Path(filepath).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, filepath: str | Path) -> None:
        try:
            state = torch.load(Path(filepath).expanduser(), map_location="cpu", weights_only=True)
        except TypeError:  # torch < 2.0
            state = torch.load(Path(filepath).expanduser(), map_location="cpu")
        self.load_state_dict(state, strict=True)
        self.validate_fitted()
