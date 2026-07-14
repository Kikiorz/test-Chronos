"""Train DINOv3 RGB Chronos on native 14-D RMBench joint trajectories."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint

try:
    from .M_dataset_robotwinRGB_J import (
        RGBJointTrajectoryDataset,
        discover_episode_files,
        make_joint_scaler,
        parallel_collate_fn_rgb_joint,
    )
    from .dinov3_backbone import (
        DINOV3_IMAGE_HW,
        DINOV3_MODEL_NAME,
        configure_float32_numerics,
    )
    from .contracts import base_policy_contract
    from .mamba_policy_par_2D_IMLE_Joint import MambaConfig, MambaPolicy
    from .ema_callback import WarmupPolicyEMACallback
    from .scaler_M import Scaler
except ImportError:  # direct script execution
    from M_dataset_robotwinRGB_J import (
        RGBJointTrajectoryDataset,
        discover_episode_files,
        make_joint_scaler,
        parallel_collate_fn_rgb_joint,
    )
    from dinov3_backbone import (
        DINOV3_IMAGE_HW,
        DINOV3_MODEL_NAME,
        configure_float32_numerics,
    )
    from contracts import base_policy_contract
    from mamba_policy_par_2D_IMLE_Joint import MambaConfig, MambaPolicy
    from ema_callback import WarmupPolicyEMACallback
    from scaler_M import Scaler


class CadencedResumeCheckpoint(ModelCheckpoint):
    """Keep one cadence checkpoint and atomically point ``last.ckpt`` to it.

    Lightning's built-in ``save_last`` runs every epoch even when
    ``every_n_epochs`` is larger than one.  A dedicated cadence callback avoids
    multi-gigabyte writes between requested save epochs while preserving the
    conventional ``--resume auto`` path.
    """

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        if trainer.is_global_zero:
            destination = Path(filepath)
            last_path = destination.parent / "last.ckpt"
            temporary_link = destination.parent / f".last.ckpt.{os.getpid()}.partial"
            temporary_link.unlink(missing_ok=True)
            temporary_link.symlink_to(destination.name)
            os.replace(temporary_link, last_path)
        trainer.strategy.barrier("cadenced_resume_checkpoint_link")


class LitMambaRGBJoint(pl.LightningModule):
    """Lightning wrapper for raw RGB or fingerprinted DINOv3 feature caches."""

    def __init__(
        self,
        config: MambaConfig,
        scaler: Scaler,
        learning_rate: float = 1.7e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 15,
        eta_min: float = 2e-5,
        vision_chunk_size: int = 32,
        supervision_frames: int = 0,
        validation_seed: int = 42,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = True,
        backbone_weights: str | Path | None = None,
        backbone_weights_sha256: str | None = None,
        run_contract: dict[str, object] | None = None,
    ):
        super().__init__()
        self.config = config
        # Lightning moves registered modules to the accelerator.  The Dataset
        # must retain its own CPU scaler inside worker processes, so never
        # register the same Scaler object in both places.
        self.scaler = copy.deepcopy(scaler)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.warmup_epochs = int(warmup_epochs)
        self.eta_min = float(eta_min)
        self.vision_chunk_size = int(vision_chunk_size)
        self.supervision_frames = int(supervision_frames)
        self.validation_seed = int(validation_seed)
        self.run_contract = dict(run_contract or {})
        if self.vision_chunk_size <= 0:
            raise ValueError("vision_chunk_size must be positive")
        if self.supervision_frames < 0:
            raise ValueError("supervision_frames must be non-negative (0 means all valid frames)")

        self.policy = MambaPolicy(
            camera_names=config.camera_names,
            embed_dim=config.embed_dim,
            lowdim_dim=config.lowdim_dim,
            d_model=config.d_model,
            action_dim=config.action_dim,
            num_blocks=config.num_blocks,
            mamba_cfg=config,
            future_steps=config.future_steps,
            pretrained_backbone=pretrained_backbone,
            freeze_backbone=freeze_backbone,
            backbone_weights=backbone_weights,
            dinov3_image_hw=DINOV3_IMAGE_HW,
        )
        self.save_hyperparameters(
            {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "warmup_epochs": self.warmup_epochs,
                "eta_min": self.eta_min,
                "vision_chunk_size": self.vision_chunk_size,
                "supervision_frames": self.supervision_frames,
                "validation_seed": self.validation_seed,
                "pretrained_backbone": bool(pretrained_backbone),
                "freeze_backbone": bool(freeze_backbone),
                "backbone_name": DINOV3_MODEL_NAME,
                "dinov3_image_hw": list(DINOV3_IMAGE_HW),
                "policy_variant": "chronos_rgb_joint14_dinov3b",
                "backbone_weights_sha256": backbone_weights_sha256,
                "run_contract": self.run_contract,
                "camera_names": list(config.camera_names),
                "embed_dim": config.embed_dim,
                "d_model": config.d_model,
                "lowdim_dim": config.lowdim_dim,
                "action_dim": config.action_dim,
                "future_steps": config.future_steps,
                "num_blocks": config.num_blocks,
            }
        )

    def transfer_batch_to_device(
        self, batch: Dict[str, Any], device: torch.device, dataloader_idx: int
    ) -> Dict[str, Any]:
        # Keep full raw RGB or float32 cached tokens on CPU and move only a
        # bounded frame chunk inside _compute_vision_features.
        transferred = dict(batch)
        for key in ("obs", "actions", "mask", "episode_index", "lengths"):
            if key in transferred and torch.is_tensor(transferred[key]):
                transferred[key] = transferred[key].to(device, non_blocking=True)
        return transferred

    def _compute_vision_features(
        self, vision: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        raw_images = vision.ndim == 5
        cached_tokens = vision.ndim == 4
        if raw_images and vision.shape[2] != 3:
            raise ValueError(f"Expected RGB [B,L,3,H,W], got {tuple(vision.shape)}")
        if not raw_images and not cached_tokens:
            raise ValueError(
                "Expected raw RGB [B,L,3,H,W] or cached tokens [B,L,21,768], "
                f"got {tuple(vision.shape)}"
            )
        batch_size, sequence_length = obs.shape[:2]
        flat_valid = (~mask).reshape(-1)
        valid_indices = flat_valid.nonzero(as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            raise ValueError("Batch contains no valid trajectory frames")

        flat_obs = obs.reshape(-1, obs.shape[-1])
        valid_obs = flat_obs.index_select(0, valid_indices)
        flat_vision = vision.reshape(-1, *vision.shape[2:])
        # Vision data deliberately remains CPU, so use CPU selection indices.
        valid_vision = flat_vision.index_select(0, valid_indices.detach().cpu())

        feature_chunks = []
        for start in range(0, valid_indices.numel(), self.vision_chunk_size):
            end = min(start + self.vision_chunk_size, valid_indices.numel())
            vision_chunk = valid_vision[start:end].to(self.device, non_blocking=True)
            proprio_chunk = valid_obs[start:end]
            # For online raw RGB with a frozen backbone, compute DINOv3 once.
            # Checkpointing the whole fusion module would needlessly run the
            # 86M backbone again during backward.
            if raw_images and self.policy.fusion_engine.image_encoder.freeze:
                vision_chunk = self.policy.fusion_engine.image_encoder(vision_chunk)
            if self.training and torch.is_grad_enabled():
                # The cache removes frozen-DINO compute; checkpoint only the
                # small trainable adapter over each full-trajectory chunk.
                feature = checkpoint(
                    self.policy.fusion_engine,
                    vision_chunk,
                    proprio_chunk,
                    use_reentrant=False,
                )
            else:
                feature = self.policy.fusion_engine(vision_chunk, proprio_chunk)
            feature_chunks.append(feature)
        valid_features = torch.cat(feature_chunks, dim=0)

        flat_features = torch.zeros(
            batch_size * sequence_length,
            self.config.embed_dim,
            device=self.device,
            dtype=valid_features.dtype,
        )
        # Out-of-place index_copy preserves gradients into visual/proprio heads.
        flat_features = flat_features.index_copy(0, valid_indices, valid_features)
        return flat_features.view(batch_size, sequence_length, self.config.embed_dim)

    def _supervision_indices(
        self, lengths: torch.Tensor, stage: str, device: torch.device
    ) -> torch.Tensor:
        shortest_episode = int(lengths.min().item())
        if self.supervision_frames == 0 and not torch.all(lengths == lengths[0]):
            raise ValueError(
                "--supervision-frames 0 means every valid timestep and therefore requires "
                "--batch-size 1 for variable-length episodes. Use a fixed positive "
                "supervision count if batching multiple episodes."
            )
        num_frames = (
            shortest_episode
            if self.supervision_frames == 0
            else min(self.supervision_frames, shortest_episode)
        )
        rows = []
        for length_tensor in lengths:
            length = int(length_tensor.item())
            if num_frames == length:
                # Full supervision should exactly preserve the released
                # per-frame ordering and random-number consumption.
                row = torch.arange(length, device=device)
            elif stage == "train":
                row = torch.randperm(length, device=device)[:num_frames]
            else:
                # Deterministic validation coverage from episode start to end.
                row = torch.linspace(0, length - 1, steps=num_frames, device=device).round().long()
            rows.append(row)
        return torch.stack(rows, dim=0)

    def _validation_batch_seed(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> int:
        episode_indices = batch.get("episode_index")
        stable_offset = int(batch_idx) * 1_000_003
        if torch.is_tensor(episode_indices):
            for position, episode_index in enumerate(episode_indices.detach().cpu().tolist()):
                stable_offset += (position + 1) * (int(episode_index) + 1) * 9_176
        return (self.validation_seed + stable_offset) % (2**63 - 1)

    def _shared_step(
        self, batch: Dict[str, torch.Tensor], stage: str, batch_idx: int
    ) -> torch.Tensor:
        obs = batch["obs"]
        actions = batch["actions"]
        mask = batch["mask"]
        vision_key = "image_features" if "image_features" in batch else "image"
        fused = self._compute_vision_features(batch[vision_key], obs, mask)
        indices = self._supervision_indices(batch["lengths"], stage, obs.device)
        if stage == "val":
            # IMLE mode selection, bridge time and bridge noise are stochastic.
            # Isolate and restore both CPU and this rank's CUDA RNG so validation
            # is repeatable and cannot perturb the next training epoch.
            cuda_devices: list[int] = []
            cuda_index: int | None = None
            if obs.device.type == "cuda":
                cuda_index = obs.device.index
                if cuda_index is None:
                    cuda_index = torch.cuda.current_device()
                cuda_devices = [cuda_index]
            with torch.random.fork_rng(devices=cuda_devices, enabled=True):
                seed = self._validation_batch_seed(batch, batch_idx)
                torch.random.default_generator.manual_seed(seed)
                if cuda_index is not None:
                    with torch.cuda.device(cuda_index):
                        torch.cuda.manual_seed(seed)
                per_frame_loss = self.policy.compute_loss_at_indices(fused, actions, indices)
        else:
            per_frame_loss = self.policy.compute_loss_at_indices(fused, actions, indices)
        loss = per_frame_loss.mean()
        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            sync_dist=(stage != "train"),
            batch_size=obs.shape[0],
        )
        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train", batch_idx)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val", batch_idx)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["policy_contract"] = {
            **base_policy_contract(),
            **self.run_contract,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            (parameter for parameter in self.policy.parameters() if parameter.requires_grad),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        max_epochs = int(self.trainer.max_epochs)
        if self.warmup_epochs <= 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, max_epochs), eta_min=self.eta_min
            )
        elif self.warmup_epochs >= max_epochs:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=max_epochs
            )
        else:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=self.warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epochs - self.warmup_epochs,
                eta_min=self.eta_min,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.warmup_epochs],
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


def _parse_devices(value: str) -> Any:
    value = value.strip()
    if value.isdigit():
        return int(value)
    if "," in value:
        return [int(part) for part in value.split(",")]
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_resume(value: str, output_dir: Path) -> str | None:
    lowered = value.lower()
    if lowered == "none":
        return None
    if lowered == "auto":
        last_checkpoint = output_dir / "last.ckpt"
        return str(last_checkpoint) if last_checkpoint.is_file() else None
    resolved = Path(os.path.expanduser(value)).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {resolved}")
    return str(resolved)


def _configure_large_temp_dir(
    output_dir: Path, requested_dir: str | Path | None = None
) -> Path:
    """Put large atomic checkpoint files on the output filesystem.

    PyTorch DataLoader workers also use :mod:`tempfile` for AF_UNIX resource
    sharing sockets. Linux limits those socket paths to 107 bytes, so a long
    run-specific directory can fail before the first validation batch. Keep a
    deliberately short shared sibling by default and reject paths that cannot
    accommodate multiprocessing's ``pymp-*/listener-*`` suffix.
    """

    temp_dir = (
        Path(requested_dir).expanduser().resolve()
        if requested_dir is not None
        else (output_dir.parent / ".tmp").resolve()
    )
    temp_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if temp_dir.stat().st_dev != output_dir.parent.stat().st_dev:
        raise ValueError(
            "Checkpoint temp directory must be on the same filesystem as --output-dir "
            "so multi-gigabyte atomic renames cannot fail with EXDEV: "
            f"temp={temp_dir}, output={output_dir}"
        )

    # multiprocessing.util.get_temp_dir() creates exactly this shape before
    # Listener adds its random basename. Use eight placeholder characters,
    # matching tempfile's current random-name length, plus one byte for the
    # terminating NUL in sockaddr_un.sun_path[108].
    socket_probe = temp_dir / "pymp-00000000" / "listener-00000000"
    socket_bytes = len(os.fsencode(socket_probe)) + 1
    if socket_bytes > 108:
        raise ValueError(
            "Checkpoint temp directory is too long for Linux DataLoader AF_UNIX "
            f"sockets ({socket_bytes} > 108 bytes with suffix): {temp_dir}. "
            "Pass --temp-dir with a shorter path on the output filesystem."
        )
    if not os.access(temp_dir, os.W_OK | os.X_OK):
        raise PermissionError(f"Checkpoint temp directory is not writable: {temp_dir}")

    # tempfile.tempdir covers the current/forked process; TMPDIR also covers
    # spawned workers and subprocesses. Files live on the large output disk.
    os.environ["TMPDIR"] = str(temp_dir)
    tempfile.tempdir = str(temp_dir)
    return temp_dir


def _validate_resume_checkpoint(
    checkpoint_path: str, expected_contract: Mapping[str, object]
) -> None:
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:  # older torch without mmap/weights_only
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Resume checkpoint must be a mapping")
    contract = checkpoint.get("policy_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Resume checkpoint is missing policy_contract")
    mismatches = {
        key: (contract.get(key), expected)
        for key, expected in expected_contract.items()
        if contract.get(key) != expected
    }
    if mismatches:
        preview = dict(list(mismatches.items())[:12])
        raise RuntimeError(
            "Refusing incompatible resume checkpoint; policy/data/cache/scaler contract "
            f"changed: {preview}"
        )
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("Resume checkpoint has no Lightning state_dict")
    embedded_scaler = {
        key[len("scaler.") :]: value
        for key, value in state.items()
        if isinstance(key, str) and key.startswith("scaler.") and torch.is_tensor(value)
    }
    embedded_fingerprint = Scaler.state_fingerprint(embedded_scaler)
    if embedded_fingerprint != expected_contract.get("scaler_fingerprint"):
        raise RuntimeError(
            "Resume checkpoint's embedded scaler does not match its/current contract"
        )
    del checkpoint


def build_arg_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train DINOv3 Chronos with RMBench RGB and native 14-D joint actions"
    )
    parser.add_argument("--data-root", required=True, help="RMBench .../demo_clean/data")
    parser.add_argument(
        "--expected-episodes",
        type=int,
        default=50,
        help="Refuse to train unless this many episodes are complete (0 disables the guard)",
    )
    parser.add_argument("--task-name", default="cover_blocks")
    parser.add_argument(
        "--output-dir", default=str(here / "checkpoints" / "cover_blocks" / "Joint_14")
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help=(
            "Short temp directory on the output filesystem for atomic checkpoints and "
            "DataLoader sockets (default: <output-parent>/.tmp)"
        ),
    )
    parser.add_argument("--scaler-path", default=str(here / "scaler_cover_blocks_joint_rgb.pth"))
    parser.add_argument(
        "--feature-root",
        default=None,
        help="Optional external [T,21,768] frozen-DINOv3 cache directory",
    )
    parser.add_argument(
        "--backbone-weights",
        default=None,
        help="External DINOv3-B/16 .safetensors/.pth (never commit this file)",
    )
    parser.add_argument("--refit-scaler", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--vision-chunk-size", type=int, default=32)
    parser.add_argument(
        "--supervision-frames",
        type=int,
        default=0,
        help="Full history is retained; IMLE/SB loss uses this many timesteps (0 = all valid frames)",
    )
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--accumulate-grad-batches", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1.7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=15)
    parser.add_argument("--eta-min", type=float, default=2e-5)
    parser.add_argument("--accelerator", default="gpu" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--devices", default="1", help="Lightning devices, e.g. 1 or 0,1")
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--save-top-k", type=int, default=2)
    parser.add_argument(
        "--checkpoint-every-n-epochs",
        type=int,
        default=5,
        help="Write ~5.4 GiB resume/best checkpoints at this cadence (training is unchanged)",
    )
    parser.add_argument(
        "--periodic-every-n-epochs",
        type=int,
        default=0,
        help="Extra full checkpoints; 0 disables them to protect local disk space",
    )
    parser.add_argument(
        "--overfit-batches",
        type=int,
        default=0,
        help="Lightning diagnostic: repeatedly train on this many batches (0 disables)",
    )
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Disable the released Chronos warmup EMA (enabled by default)",
    )
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument(
        "--resume",
        default="auto",
        help="auto, none, or a Lightning checkpoint path",
    )
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    pl.seed_everything(args.seed, workers=True)
    configure_float32_numerics()

    output_dir = Path(args.output_dir).expanduser().resolve()
    scaler_path = Path(args.scaler_path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    # Lightning/FSSpec stages atomic checkpoints through tempfile.mkstemp().
    # /tmp may be much smaller than a full raw+EMA+Adam checkpoint. The helper
    # keeps staging on the output filesystem while reserving AF_UNIX path space
    # for DataLoader workers.
    checkpoint_temp_dir = _configure_large_temp_dir(output_dir, args.temp_dir)
    print(f"Large-file/checkpoint temp directory: {checkpoint_temp_dir}")
    resume = _resolve_resume(args.resume, output_dir)

    if args.expected_episodes < 0:
        raise ValueError("--expected-episodes must be non-negative")
    if args.save_top_k < 1:
        raise ValueError("--save-top-k must be at least 1")
    if args.checkpoint_every_n_epochs < 1:
        raise ValueError("--checkpoint-every-n-epochs must be at least 1")
    if args.epochs % args.checkpoint_every_n_epochs:
        raise ValueError(
            "--epochs must be divisible by --checkpoint-every-n-epochs so the final "
            "optimizer/EMA state is resumable"
        )
    if args.periodic_every_n_epochs < 0:
        raise ValueError("--periodic-every-n-epochs must be non-negative")
    if args.feature_root is not None and args.train_backbone:
        raise ValueError("A frozen feature cache cannot be used with --train-backbone")
    if args.feature_root is None and not args.no_pretrained_backbone:
        raise ValueError(
            "Formal pretrained RGB-Joint training requires --feature-root so the exact "
            "DINOv3 preprocessing/data fingerprint is bound into the checkpoint. "
            "The raw path is reserved for explicit random-backbone diagnostics."
        )
    if args.refit_scaler and resume is not None:
        raise ValueError(
            "--refit-scaler cannot be combined with resume. Use --resume none and a "
            "new run, or keep the checkpoint's verified scaler."
        )
    backbone_weights: Path | None = None
    backbone_sha256: str | None = None
    if args.backbone_weights is not None:
        backbone_weights = Path(args.backbone_weights).expanduser().resolve()
        if not backbone_weights.is_file():
            raise FileNotFoundError(f"DINOv3 weights do not exist: {backbone_weights}")
        backbone_sha256 = _sha256_file(backbone_weights)
    elif not args.no_pretrained_backbone:
        raise ValueError(
            "Pretrained DINOv3 is gated. Pass --backbone-weights with an external "
            "authorized file, or explicitly use --no-pretrained-backbone only for tests."
        )
    all_episode_files = discover_episode_files(args.data_root, mode="all")
    if args.expected_episodes and len(all_episode_files) != args.expected_episodes:
        raise RuntimeError(
            f"Dataset completeness guard: expected exactly {args.expected_episodes} episodes, "
            f"but found {len(all_episode_files)} below {Path(args.data_root).expanduser().resolve()}. "
            "Wait for the download to finish, or explicitly pass --expected-episodes 0 for diagnostics."
        )

    split_kwargs = {
        "root_dir": args.data_root,
        "future_steps": 16,
        "image_hw": (args.image_height, args.image_width),
        "val_fraction": args.val_fraction,
        "split_seed": args.split_seed,
        "feature_root": args.feature_root,
    }
    train_dataset = RGBJointTrajectoryDataset(mode="train", scaler=None, **split_kwargs)
    # Fitting 45 small joint arrays is cheap.  Always recompute the expected
    # train-only statistics so a stale same-shaped scaler cannot be reused.
    fitted_scaler = make_joint_scaler(16)
    train_dataset.scaler = fitted_scaler
    train_dataset.fit_scaler()
    fitted_fingerprint = fitted_scaler.fingerprint()
    save_scaler_after_preflight = False
    if scaler_path.is_file() and not args.refit_scaler:
        persisted_scaler = make_joint_scaler(16)
        persisted_scaler.load(scaler_path)
        if persisted_scaler.fingerprint() != fitted_fingerprint:
            raise RuntimeError(
                "Existing RGB joint scaler does not match statistics recomputed from the "
                "current train episodes. Use a new output directory or explicitly pass "
                "--refit-scaler together with --resume none."
            )
        scaler = persisted_scaler
        print(f"Verified existing train-only RGB joint scaler: {scaler_path}")
    else:
        scaler = fitted_scaler
        save_scaler_after_preflight = True
        print("Fitted a new train-only RGB joint scaler; save is deferred until preflight")
    train_dataset.scaler = scaler
    val_dataset = RGBJointTrajectoryDataset(mode="val", scaler=scaler, **split_kwargs)

    cache_contract_sha256: str | None = None
    cache_dataset_sha256: str | None = None
    if train_dataset.feature_metadata is not None:
        cache_sha256 = train_dataset.feature_metadata["weights_sha256"]
        if backbone_sha256 != cache_sha256:
            raise RuntimeError(
                "DINOv3 cache/backbone mismatch: "
                f"cache={cache_sha256}, weights={backbone_sha256}"
            )
        cache_contract_sha256 = str(
            train_dataset.feature_metadata["cache_contract_sha256"]
        )
        cache_dataset_sha256 = str(
            train_dataset.feature_metadata["cache_dataset_sha256"]
        )

    run_contract: dict[str, object] = {
        "backbone_weights_sha256": backbone_sha256,
        "scaler_fingerprint": scaler.fingerprint(),
        "cache_contract_sha256": cache_contract_sha256,
        "cache_dataset_sha256": cache_dataset_sha256,
        "split_seed": args.split_seed,
        "val_fraction": args.val_fraction,
        "train_episodes": [path.name for path in train_dataset.file_paths],
        "val_episodes": [path.name for path in val_dataset.file_paths],
        "training_contract": {
            "seed": args.seed,
            "validation_seed": args.validation_seed,
            "batch_size": args.batch_size,
            "accumulate_grad_batches": args.accumulate_grad_batches,
            "supervision_frames": args.supervision_frames,
            "vision_chunk_size": args.vision_chunk_size,
            "max_epochs": args.epochs,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "optimizer_betas": [0.9, 0.999],
            "optimizer_eps": 1e-8,
            "optimizer_amsgrad": False,
            "scheduler": "linear_warmup_then_cosine",
            "warmup_epochs": args.warmup_epochs,
            "eta_min": args.eta_min,
            "precision": args.precision,
            "gradient_clip_val": args.gradient_clip_val,
            "ema_enabled": not args.no_ema,
            "ema_inv_gamma": None if args.no_ema else 1.0,
            "ema_power": None if args.no_ema else 2.0 / 3.0,
            "ema_min_decay": None if args.no_ema else 0.0,
            "ema_max_decay": None if args.no_ema else 0.9999,
            "ema_update_timing": "every_training_batch_matching_released_chronos",
            "overfit_batches": args.overfit_batches,
            "pretrained_backbone": not args.no_pretrained_backbone,
            "freeze_backbone": not args.train_backbone,
            "accelerator": args.accelerator,
            "devices": args.devices,
            "deterministic_trainer": False,
            "float32_matmul_precision": "highest",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        },
    }
    expected_policy_contract = {**base_policy_contract(), **run_contract}
    if resume is not None:
        _validate_resume_checkpoint(resume, expected_policy_contract)
        print(f"Verified resume contract: {resume}")
    if save_scaler_after_preflight:
        temporary_scaler = scaler_path.with_suffix(scaler_path.suffix + ".partial")
        temporary_scaler.unlink(missing_ok=True)
        scaler.save(temporary_scaler)
        temporary_scaler.replace(scaler_path)
        print(f"Saved verified train-only RGB joint scaler: {scaler_path}")

    manifest = {
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "expected_episodes": args.expected_episodes,
        "policy_contract": expected_policy_contract,
        "feature_root": None
        if args.feature_root is None
        else str(Path(args.feature_root).expanduser().resolve()),
    }
    manifest_path = output_dir / "split_manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.partial")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    print(f"Wrote deterministic episode split manifest: {manifest_path}")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
        "collate_fn": parallel_collate_fn_rgb_joint,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    config = MambaConfig()
    model = LitMambaRGBJoint(
        config,
        scaler=scaler,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        eta_min=args.eta_min,
        vision_chunk_size=args.vision_chunk_size,
        supervision_frames=args.supervision_frames,
        validation_seed=args.validation_seed,
        pretrained_backbone=not args.no_pretrained_backbone,
        freeze_backbone=not args.train_backbone,
        backbone_weights=backbone_weights,
        backbone_weights_sha256=backbone_sha256,
        run_contract=run_contract,
    )

    callbacks = []
    if not args.no_ema:
        callbacks.append(WarmupPolicyEMACallback(max_value=0.9999))
    callbacks.append(
        ModelCheckpoint(
            dirpath=output_dir,
            filename="mamba-best-{epoch:04d}-{val_loss:.5f}",
            monitor="val_loss",
            mode="min",
            save_top_k=args.save_top_k,
            save_last=False,
            every_n_epochs=args.checkpoint_every_n_epochs,
        )
    )
    callbacks.append(
        CadencedResumeCheckpoint(
            dirpath=output_dir,
            filename="mamba-resume-{epoch:04d}",
            monitor=None,
            save_top_k=1,
            save_last=False,
            every_n_epochs=args.checkpoint_every_n_epochs,
            save_on_train_epoch_end=True,
        )
    )
    if args.periodic_every_n_epochs:
        callbacks.append(
            ModelCheckpoint(
                dirpath=output_dir,
                filename="mamba-periodic-{epoch:04d}",
                every_n_epochs=args.periodic_every_n_epochs,
                save_top_k=-1,
                save_on_train_epoch_end=True,
            )
        )
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    logger = TensorBoardLogger(
        save_dir=output_dir.parent.parent, name=f"{args.task_name}_rgb_joint_dinov3"
    )
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=_parse_devices(args.devices),
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        deterministic=False,
        overfit_batches=args.overfit_batches,
    )

    print(
        f"Training RGB Joint Chronos: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, resume={resume}"
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume)


if __name__ == "__main__":
    main()
