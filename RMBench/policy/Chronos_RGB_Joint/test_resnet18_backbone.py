from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from .resnet18_backbone import (
    OfficialResNet18Trunk,
    RESNET18_NORMALIZATION_MEAN,
    RESNET18_NORMALIZATION_STD,
    RESNET18_WEIGHTS_SHA256,
    preprocess_rgb_numpy,
    verify_resnet18_weights,
)


OFFICIAL_CACHE = Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"


def test_preprocess_rgb_numpy_matches_official_operations():
    rows = np.arange(240, dtype=np.uint16)[:, None, None]
    cols = np.arange(320, dtype=np.uint16)[None, :, None]
    channels = np.arange(3, dtype=np.uint16)[None, None, :]
    image = ((rows * 3 + cols * 5 + channels * 71) % 256).astype(np.uint8)

    actual = preprocess_rgb_numpy(image)
    resized = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
    expected = resized.astype(np.float32) / np.float32(255.0)
    expected = expected.transpose(2, 0, 1)
    expected = (
        expected
        - np.asarray(RESNET18_NORMALIZATION_MEAN, dtype=np.float32)[:, None, None]
    ) / np.asarray(RESNET18_NORMALIZATION_STD, dtype=np.float32)[:, None, None]

    assert actual.shape == (3, 480, 640)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros((240, 320), dtype=np.uint8),
        np.zeros((240, 320, 4), dtype=np.uint8),
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros((240, 320, 3), dtype=np.float32),
    ],
)
def test_preprocess_rgb_numpy_rejects_contract_mismatch(invalid):
    with pytest.raises((TypeError, ValueError)):
        preprocess_rgb_numpy(invalid)


def test_verify_resnet18_weights_rejects_wrong_file(tmp_path):
    wrong = tmp_path / "resnet18.pth"
    wrong.write_bytes(b"not the official torchvision weights")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_resnet18_weights(wrong)


@pytest.mark.skipif(not OFFICIAL_CACHE.is_file(), reason="official cache is not installed")
def test_official_resnet18_weights_frozen_eval_and_output_contract():
    assert verify_resnet18_weights(OFFICIAL_CACHE) == RESNET18_WEIGHTS_SHA256
    trunk = OfficialResNet18Trunk(
        pretrained=True,
        weights_path=OFFICIAL_CACHE,
        freeze=True,
    )

    assert not trunk.training
    assert all(not parameter.requires_grad for parameter in trunk.parameters())
    assert not any(isinstance(module, nn.BatchNorm2d) for module in trunk.modules())
    assert sum(isinstance(module, nn.GroupNorm) for module in trunk.modules()) == 20

    trunk.train(True)
    assert not trunk.training
    assert all(not module.training for module in trunk.modules())

    image = preprocess_rgb_numpy(np.zeros((240, 320, 3), dtype=np.uint8))
    features = trunk(torch.from_numpy(image).unsqueeze(0))
    assert features.shape == (1, 512, 15, 20)
    assert features.dtype == torch.float32
    assert not features.requires_grad
    assert torch.isfinite(features).all()
