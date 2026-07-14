"""Precompute frozen DINOv3-B/16 features for RMBench HDF5 episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

try:
    from .M_dataset_robotwinRGB_J import _decode_robotwin_rgb, discover_episode_files
    from .dinov3_backbone import (
        DINOV3_CACHE_AMP_DTYPE,
        DINOV3_CACHE_BATCH_IMAGES,
        DINOV3_CACHE_DTYPE,
        DINOV3_CACHE_DEVICE_TYPE,
        DINOV3_CACHE_FORMAT_VERSION,
        DINOV3_CACHE_TOKENS,
        DINOV3_FEATURE_DIM,
        DINOV3_IMAGE_HW,
        DINOV3_MODEL_NAME,
        DINOV3_NORMALIZATION_MEAN,
        DINOV3_NORMALIZATION_STD,
        DINOV3_PATCH_SIZE,
        DINOV3_POOL_HW,
        DINOV3_RGB_SCALE,
        DINOV3_RESIZE_MODE,
        FLOAT32_NUMERICS,
        RMBENCH_RGB_HW,
        DINOv3ImageEncoder,
        dinov3_cache_contract,
    )
except ImportError:  # direct script execution
    from M_dataset_robotwinRGB_J import _decode_robotwin_rgb, discover_episode_files
    from dinov3_backbone import (  # type: ignore
        DINOV3_CACHE_AMP_DTYPE,
        DINOV3_CACHE_BATCH_IMAGES,
        DINOV3_CACHE_DTYPE,
        DINOV3_CACHE_DEVICE_TYPE,
        DINOV3_CACHE_FORMAT_VERSION,
        DINOV3_CACHE_TOKENS,
        DINOV3_FEATURE_DIM,
        DINOV3_IMAGE_HW,
        DINOV3_MODEL_NAME,
        DINOV3_NORMALIZATION_MEAN,
        DINOV3_NORMALIZATION_STD,
        DINOV3_PATCH_SIZE,
        DINOV3_POOL_HW,
        DINOV3_RGB_SCALE,
        DINOV3_RESIZE_MODE,
        FLOAT32_NUMERICS,
        RMBENCH_RGB_HW,
        DINOv3ImageEncoder,
        dinov3_cache_contract,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _decode_batch(dataset: h5py.Dataset, start: int, end: int) -> torch.Tensor:
    frames = []
    for index in range(start, end):
        image = _decode_robotwin_rgb(dataset[index])
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Bad RGB frame {index}: {image.shape}")
        if tuple(image.shape[:2]) != RMBENCH_RGB_HW:
            raise ValueError(
                f"RMBench frame {index} has {tuple(image.shape[:2])}, "
                f"expected native {RMBENCH_RGB_HW}"
            )
        frames.append(np.ascontiguousarray(image.transpose(2, 0, 1)))
    return torch.from_numpy(np.stack(frames, axis=0))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cache CLS plus 4x5 DINOv3-B/16 tokens for RMBench RGB Joint"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--weights", required=True, help="External DINOv3/timm weights")
    parser.add_argument("--camera-name", default="head_camera")
    parser.add_argument("--expected-episodes", type=int, default=50)
    parser.add_argument("--batch-images", type=int, default=DINOV3_CACHE_BATCH_IMAGES)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default=DINOV3_CACHE_AMP_DTYPE,
        help="Formal cache contract requires none; lower precision is diagnostic only",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.batch_images <= 0:
        raise ValueError("--batch-images must be positive")
    data_root = Path(args.data_root).expanduser().resolve()
    feature_root = Path(args.feature_root).expanduser().resolve()
    weights_path = Path(args.weights).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"DINOv3 weights do not exist: {weights_path}")
    files = discover_episode_files(data_root, mode="all")
    if args.expected_episodes and len(files) != args.expected_episodes:
        raise RuntimeError(
            f"Expected {args.expected_episodes} episodes, found {len(files)} under {data_root}"
        )
    feature_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {device}")
    if device.type != DINOV3_CACHE_DEVICE_TYPE:
        raise ValueError(
            "The formal cache contract requires CUDA extraction because CPU and "
            "CUDA uint8-to-float division are not bit-identical."
        )
    encoder = DINOv3ImageEncoder(
        image_hw=DINOV3_IMAGE_HW,
        pool_hw=DINOV3_POOL_HW,
        pretrained=True,
        weights_path=weights_path,
        freeze=True,
    ).to(device).eval()
    amp_dtype = {
        "none": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.amp_dtype]

    weights_sha256 = _sha256(weights_path)
    contract = dinov3_cache_contract(
        args.camera_name,
        weights_sha256,
        batch_images=args.batch_images,
        amp_dtype=args.amp_dtype,
        device_type=device.type,
    )
    metadata: dict[str, object] = {
        **contract,
        "cache_contract_sha256": _json_sha256(contract),
        "episodes": {},
    }
    old_metadata: dict[str, object] | None = None
    metadata_path = feature_root / "metadata.json"
    if metadata_path.is_file() and not args.overwrite:
        try:
            loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(loaded_metadata, dict):
                old_metadata = loaded_metadata
        except (OSError, json.JSONDecodeError):
            old_metadata = None
    old_contract_matches = bool(
        old_metadata is not None
        and all(old_metadata.get(key) == value for key, value in contract.items())
        and old_metadata.get("cache_contract_sha256") == _json_sha256(contract)
        and isinstance(old_metadata.get("episodes"), dict)
    )
    total_frames = 0
    for episode_number, path in enumerate(files, start=1):
        output_path = feature_root / f"{path.stem}.npy"
        source_bytes = path.stat().st_size
        source_sha256 = _sha256(path)
        with h5py.File(path, "r") as root:
            image_key = f"observation/{args.camera_name}/rgb"
            if image_key not in root or "joint_action/vector" not in root:
                raise KeyError(f"{path.name} lacks {image_key!r} or joint_action/vector")
            images = root[image_key]
            num_frames = len(root["joint_action/vector"])
            if len(images) != num_frames:
                raise ValueError(
                    f"{path.name}: RGB={len(images)} and joint={num_frames} are not aligned"
                )
            expected_shape = (num_frames, DINOV3_CACHE_TOKENS, DINOV3_FEATURE_DIM)
            reuse = False
            old_entry = None
            if old_contract_matches and old_metadata is not None:
                old_episodes = old_metadata["episodes"]
                assert isinstance(old_episodes, dict)
                candidate_entry = old_episodes.get(path.name)
                if isinstance(candidate_entry, dict):
                    old_entry = candidate_entry
            entry_matches = bool(
                old_entry is not None
                and old_entry.get("feature_file") == output_path.name
                and old_entry.get("frames") == num_frames
                and old_entry.get("source_bytes") == source_bytes
                and old_entry.get("source_sha256") == source_sha256
                and isinstance(old_entry.get("feature_sha256"), str)
                and len(old_entry["feature_sha256"]) == 64
            )
            if output_path.is_file() and not args.overwrite and entry_matches:
                existing = np.load(output_path, mmap_mode="r", allow_pickle=False)
                reuse = existing.shape == expected_shape and existing.dtype == np.float32
                if reuse:
                    # Reading a reduction also detects truncated/corrupt files.
                    reuse = bool(np.isfinite(existing).all())
                if reuse:
                    reuse = _sha256(output_path) == old_entry["feature_sha256"]
            if not reuse:
                partial_path = output_path.with_suffix(".npy.partial")
                partial_path.unlink(missing_ok=True)
                output = np.lib.format.open_memmap(
                    partial_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=expected_shape,
                )
                for start in range(0, num_frames, args.batch_images):
                    end = min(start + args.batch_images, num_frames)
                    batch = _decode_batch(images, start, end).to(device, non_blocking=True)
                    amp_enabled = amp_dtype is not None and device.type == "cuda"
                    with torch.inference_mode(), torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype if amp_dtype is not None else torch.float32,
                        enabled=amp_enabled,
                    ):
                        tokens = encoder(batch)
                    output[start:end] = tokens.float().cpu().numpy()
                output.flush()
                del output
                partial_path.replace(output_path)
            feature_sha256 = _sha256(output_path)
            metadata["episodes"][path.name] = {  # type: ignore[index]
                "feature_file": output_path.name,
                "frames": num_frames,
                "source_bytes": source_bytes,
                "source_sha256": source_sha256,
                "feature_sha256": feature_sha256,
            }
            total_frames += num_frames
        print(
            f"[{episode_number:02d}/{len(files):02d}] "
            f"{'reused' if reuse else 'encoded'} {path.name}: {num_frames} frames",
            flush=True,
        )

    metadata["total_frames"] = total_frames
    metadata["cache_dataset_sha256"] = _json_sha256(metadata["episodes"])
    temporary_metadata = feature_root / "metadata.json.partial"
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary_metadata.replace(metadata_path)
    print(
        f"DINOv3 cache complete: episodes={len(files)}, frames={total_frames}, "
        f"root={feature_root}"
    )


if __name__ == "__main__":
    main()
