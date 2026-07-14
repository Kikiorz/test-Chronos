"""Train single-head RGB Chronos on RMBench dual-arm EE demonstrations."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint

try:
    from .M_dataset_robotwinRGB_E import (
        RGBTrajectoryDataset,
        discover_episode_files,
        make_ee_scaler,
        parallel_collate_fn_rgb,
    )
    from .mamba_policy_par_2D_IMLE_EE import MambaConfig, MambaPolicy
    from .ema_callback import WarmupPolicyEMACallback
    from .scaler_M import Scaler
except ImportError:  # direct script execution
    from M_dataset_robotwinRGB_E import (
        RGBTrajectoryDataset,
        discover_episode_files,
        make_ee_scaler,
        parallel_collate_fn_rgb,
    )
    from mamba_policy_par_2D_IMLE_EE import MambaConfig, MambaPolicy
    from ema_callback import WarmupPolicyEMACallback
    from scaler_M import Scaler


class LitMambaRGB(pl.LightningModule):
    """Lightning wrapper that keeps full uint8 trajectories in CPU memory."""

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
        # A full 240x320 trajectory is large.  Keep uint8 RGB on CPU and move
        # only a bounded frame chunk inside _compute_image_features.
        transferred = dict(batch)
        for key in ("obs", "actions", "mask", "episode_index", "lengths"):
            if key in transferred and torch.is_tensor(transferred[key]):
                transferred[key] = transferred[key].to(device, non_blocking=True)
        return transferred

    def _compute_image_features(
        self, images: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(f"Expected images [B,L,3,H,W], got {tuple(images.shape)}")
        batch_size, sequence_length = obs.shape[:2]
        flat_valid = (~mask).reshape(-1)
        valid_indices = flat_valid.nonzero(as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            raise ValueError("Batch contains no valid trajectory frames")

        flat_obs = obs.reshape(-1, obs.shape[-1])
        valid_obs = flat_obs.index_select(0, valid_indices)
        flat_images = images.reshape(-1, *images.shape[2:])
        # images deliberately remains CPU, so use a CPU index for selection.
        valid_images = flat_images.index_select(0, valid_indices.detach().cpu())

        feature_chunks = []
        for start in range(0, valid_indices.numel(), self.vision_chunk_size):
            end = min(start + self.vision_chunk_size, valid_indices.numel())
            image_chunk = valid_images[start:end].to(self.device, non_blocking=True)
            proprio_chunk = valid_obs[start:end]
            if self.training and torch.is_grad_enabled():
                # Full episodes contain roughly 1,000 frames.  Recompute each
                # frozen-backbone/trainable-adapter chunk during backward
                # instead of retaining every intermediate feature map.
                feature = checkpoint(
                    self.policy.fusion_engine,
                    image_chunk,
                    proprio_chunk,
                    use_reentrant=False,
                )
            else:
                feature = self.policy.fusion_engine(image_chunk, proprio_chunk)
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
        fused = self._compute_image_features(batch["image"], obs, mask)
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


def build_arg_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train Chronos with one RMBench head-camera RGB stream and 16-D EE actions"
    )
    parser.add_argument("--data-root", required=True, help="RMBench .../demo_clean/data")
    parser.add_argument(
        "--expected-episodes",
        type=int,
        default=50,
        help="Refuse to train unless this many episodes are complete (0 disables the guard)",
    )
    parser.add_argument("--task-name", default="cover_blocks")
    parser.add_argument("--output-dir", default=str(here / "checkpoints" / "cover_blocks" / "EE_16"))
    parser.add_argument("--scaler-path", default=str(here / "scaler_cover_blocks_ee_rgb.pth"))
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
    torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir).expanduser().resolve()
    scaler_path = Path(args.scaler_path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    if args.expected_episodes < 0:
        raise ValueError("--expected-episodes must be non-negative")
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
    }
    train_dataset = RGBTrajectoryDataset(mode="train", scaler=None, **split_kwargs)
    scaler = make_ee_scaler(16)
    if scaler_path.is_file() and not args.refit_scaler:
        scaler.load(scaler_path)
        print(f"Loaded RGB EE scaler: {scaler_path}")
    else:
        train_dataset.scaler = scaler
        train_dataset.fit_scaler()
        scaler.save(scaler_path)
        print(f"Fitted scaler on train episodes only and saved: {scaler_path}")
    train_dataset.scaler = scaler
    val_dataset = RGBTrajectoryDataset(mode="val", scaler=scaler, **split_kwargs)

    manifest = {
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "split_seed": args.split_seed,
        "val_fraction": args.val_fraction,
        "expected_episodes": args.expected_episodes,
        "train_episodes": [path.name for path in train_dataset.file_paths],
        "val_episodes": [path.name for path in val_dataset.file_paths],
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote deterministic episode split manifest: {manifest_path}")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
        "collate_fn": parallel_collate_fn_rgb,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    config = MambaConfig()
    model = LitMambaRGB(
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
    )

    callbacks = []
    if not args.no_ema:
        callbacks.append(WarmupPolicyEMACallback(max_value=0.9999))
    callbacks.extend([
        ModelCheckpoint(
            dirpath=output_dir,
            filename="mamba-best-{epoch:04d}-{val_loss:.5f}",
            monitor="val_loss",
            mode="min",
            save_top_k=5,
            save_last=True,
        ),
        ModelCheckpoint(
            dirpath=output_dir,
            filename="mamba-periodic-{epoch:04d}",
            every_n_epochs=100,
            save_top_k=-1,
            save_on_train_epoch_end=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ])
    logger = TensorBoardLogger(save_dir=output_dir.parent.parent, name=f"{args.task_name}_rgb")
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

    resume: str | None
    if args.resume.lower() == "none":
        resume = None
    elif args.resume.lower() == "auto":
        last_checkpoint = output_dir / "last.ckpt"
        resume = str(last_checkpoint) if last_checkpoint.is_file() else None
    else:
        resume = str(Path(os.path.expanduser(args.resume)).resolve())
    print(f"Training RGB Chronos: train={len(train_dataset)}, val={len(val_dataset)}, resume={resume}")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume)


if __name__ == "__main__":
    main()
