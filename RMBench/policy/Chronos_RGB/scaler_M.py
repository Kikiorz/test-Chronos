"""Small, checkpoint-friendly z-score scaler used by Chronos RGB."""

from __future__ import annotations

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

    @torch.no_grad()
    def fit(self, data_dict: Mapping[str, torch.Tensor]) -> None:
        missing = set(self.lowdim_dict) - set(data_dict)
        if missing:
            raise KeyError(f"Cannot fit scaler; missing keys: {sorted(missing)}")

        for key in self.lowdim_dict:
            data = torch.as_tensor(data_dict[key], dtype=torch.float32)
            if data.ndim == 0 or data.shape[0] < 1:
                raise ValueError(f"Scaler key {key!r} has no samples: shape={tuple(data.shape)}")
            mean = data.mean(dim=0)
            # Keep the official Chronos convention: torch.std's default
            # unbiased=True, with every non-sample dimension (including each
            # action-horizon slot) retaining its own statistic.
            std = data.std(dim=0).clamp_min(self.eps)
            expected_shape = self.mean_dict[key].shape
            if mean.shape != expected_shape:
                raise ValueError(
                    f"Scaler key {key!r}: expected statistic shape {tuple(expected_shape)}, "
                    f"got {tuple(mean.shape)}"
                )
            self.mean_dict[key].copy_(mean)
            self.std_dict[key].copy_(std)

    def normalize(self, data_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            key: (value - self.mean_dict[key]) / self.std_dict[key]
            if key in self.lowdim_dict
            else value
            for key, value in data_dict.items()
        }

    def denormalize(self, data_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            key: value * self.std_dict[key] + self.mean_dict[key]
            if key in self.lowdim_dict
            else value
            for key, value in data_dict.items()
        }

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, filepath: str | Path) -> None:
        state = torch.load(Path(filepath).expanduser(), map_location="cpu", weights_only=True)
        self.load_state_dict(state, strict=True)
