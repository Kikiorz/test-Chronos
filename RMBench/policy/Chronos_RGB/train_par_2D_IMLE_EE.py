"""Train the RMBench Chronos policy with the official real-world RGB encoder.

The experiment deliberately has two sources of truth:

* RMBench supplies the 16-D dual-arm EE interface, temporal/action model and
  released training hyperparameters.
* ``real_wolrd`` supplies the single-camera ResNet18 visual front end and
  image preprocessing contract.

No point-cloud tensor is used by this entry point.
"""

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

try:
    from .M_dataset_robotwinRGB_E import (
        ACTION_TARGET_OFFSET,
        RGBTrajectoryDataset,
        discover_episode_files,
        make_ee_scaler,
        parallel_collate_fn_rgb,
    )
    from .ema_callback import WarmupPolicyEMACallback
    from .mamba_policy_par_2D_IMLE_EE import MambaConfig, MambaPolicy
    from .scaler_M import Scaler
except ImportError:  # direct script execution
    from M_dataset_robotwinRGB_E import (
        ACTION_TARGET_OFFSET,
        RGBTrajectoryDataset,
        discover_episode_files,
        make_ee_scaler,
        parallel_collate_fn_rgb,
    )
    from ema_callback import WarmupPolicyEMACallback
    from mamba_policy_par_2D_IMLE_EE import MambaConfig, MambaPolicy
    from scaler_M import Scaler


class LitMambaRGB(pl.LightningModule):
    """Official RMBench objective with chunked CPU-to-GPU RGB transfer."""

    def __init__(
        self,
        config: MambaConfig,
        scaler: Scaler,
        learning_rate: float = 1.7e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 15,
        eta_min: float = 2e-5,
        vision_chunk_size: int = 256,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        # Dataset workers keep their own CPU scaler.  Lightning may move this
        # copy to the accelerator without mutating the Dataset instance.
        self.scaler = copy.deepcopy(scaler)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.warmup_epochs = int(warmup_epochs)
        self.eta_min = float(eta_min)
        self.vision_chunk_size = int(vision_chunk_size)
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0 or self.eta_min < 0:
            raise ValueError("weight_decay and eta_min must be non-negative")
        if self.vision_chunk_size <= 0:
            raise ValueError("vision_chunk_size must be positive")

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
            freeze_backbone=True,
        )
        self.save_hyperparameters(
            {
                "contract": "rmbench_official_with_realworld_rgb_encoder",
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "warmup_epochs": self.warmup_epochs,
                "eta_min": self.eta_min,
                "vision_chunk_size": self.vision_chunk_size,
                "pretrained_backbone": bool(pretrained_backbone),
                "camera_names": list(config.camera_names),
                "image_hw": [480, 640],
                "embed_dim": config.embed_dim,
                "d_model": config.d_model,
                "lowdim_dim": config.lowdim_dim,
                "action_dim": config.action_dim,
                "future_steps": config.future_steps,
                "num_blocks": config.num_blocks,
                "action_target_offset": ACTION_TARGET_OFFSET,
            }
        )

    def transfer_batch_to_device(
        self, batch: Dict[str, Any], device: torch.device, dataloader_idx: int
    ) -> Dict[str, Any]:
        # A 480x640 trajectory is large.  Keep uint8 images on CPU and transfer
        # only a bounded frame chunk in _compute_image_features.
        transferred = dict(batch)
        for key in ("obs", "actions", "mask", "episode_index", "lengths"):
            value = transferred.get(key)
            if torch.is_tensor(value):
                transferred[key] = value.to(device, non_blocking=False)
        return transferred

    def _compute_image_features(
        self, images: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if images.ndim != 5 or images.shape[2:] != (3, 480, 640):
            raise ValueError(
                "Official real-world RGB encoder requires images [B,L,3,480,640], "
                f"got {tuple(images.shape)}"
            )
        if obs.ndim != 3 or obs.shape[-1] != 16:
            raise ValueError(f"Expected normalized RMBench EE state [B,L,16], got {tuple(obs.shape)}")
        if mask.shape != obs.shape[:2]:
            raise ValueError(f"Mask shape {tuple(mask.shape)} does not match obs {tuple(obs.shape)}")

        batch_size, sequence_length = obs.shape[:2]
        valid_flat = (~mask).reshape(-1)
        valid_indices = valid_flat.nonzero(as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            raise ValueError("Batch contains no valid frames")

        flat_obs = obs.reshape(-1, obs.shape[-1])
        valid_obs = flat_obs.index_select(0, valid_indices)
        flat_images = images.reshape(-1, *images.shape[2:])
        valid_images = flat_images.index_select(0, valid_indices.detach().cpu())

        # The official real-world trunk is frozen.  Its module remains in eval
        # mode while the visual adapter, proprio projector and fusion layer train.
        fusion = self.policy.fusion_engine
        if hasattr(fusion, "vision_net"):
            fusion.vision_net.eval()

        chunks = []
        for start in range(0, int(valid_indices.numel()), self.vision_chunk_size):
            end = min(start + self.vision_chunk_size, int(valid_indices.numel()))
            image_chunk = valid_images[start:end].to(self.device, non_blocking=False)
            chunks.append(fusion(image_chunk, valid_obs[start:end]))
        valid_features = torch.cat(chunks, dim=0)

        flat_features = valid_features.new_zeros(batch_size * sequence_length, self.config.embed_dim)
        flat_features = flat_features.index_copy(0, valid_indices, valid_features)
        return flat_features.view(batch_size, sequence_length, self.config.embed_dim)

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        obs = batch["obs"]
        actions = batch["actions"]
        mask = batch["mask"]
        fused = self._compute_image_features(batch["image"], obs, mask)

        # This is the released RMBench objective: compute every padded row and
        # remove padding only when reducing the returned [B,L] loss.
        loss_per_frame = self.policy.compute_loss(fused, actions)
        valid = (~mask).to(loss_per_frame.dtype)
        loss = (loss_per_frame * valid).sum() / valid.sum().clamp_min(1.0)
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
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        # Both official RMBench and official real-world scripts use one AdamW
        # group at 1.7e-4.  The frozen ResNet trunk is excluded.
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
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train RMBench Chronos with the official real-world ResNet18 RGB encoder"
    )
    parser.add_argument("--data-root", required=True, help="RMBench .../demo_clean/data")
    parser.add_argument(
        "--expected-episodes",
        type=int,
        default=50,
        help="Refuse to train unless exactly this many complete episodes exist (0 disables)",
    )
    parser.add_argument("--task-name", default="cover_blocks")
    parser.add_argument(
        "--output-dir", default=str(here / "checkpoints" / "cover_blocks" / "EE_16_official_rgb")
    )
    parser.add_argument(
        "--scaler-path", default=str(here / "scaler_cover_blocks_ee_official_rgb.pth")
    )
    parser.add_argument("--refit-scaler", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--action-target-offset", type=int, default=ACTION_TARGET_OFFSET)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--vision-chunk-size", type=int, default=256)
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
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    parser.add_argument(
        "--periodic-every",
        type=int,
        default=100,
        help="Save a resumable checkpoint every N epochs (0 disables)",
    )
    parser.add_argument(
        "--resume", default="auto", help="auto, none, or a Lightning checkpoint path"
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run Lightning's one-batch forward/backward validation without checkpoints",
    )
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    pl.seed_everything(args.seed, workers=True)

    if args.expected_episodes < 0:
        raise ValueError("--expected-episodes must be non-negative")
    if args.action_target_offset != ACTION_TARGET_OFFSET or ACTION_TARGET_OFFSET != 0:
        raise ValueError("Official RMBench targets require --action-target-offset 0")
    if (args.image_height, args.image_width) != (480, 640):
        raise ValueError("Official real-world RGB encoder requires --image-height 480 --image-width 640")
    if args.batch_size != 2:
        raise ValueError("Official RMBench training requires --batch-size 2")
    if args.accumulate_grad_batches != 3:
        raise ValueError("Official RMBench training requires --accumulate-grad-batches 3")
    if args.epochs != 600 and not args.fast_dev_run:
        raise ValueError("Official RMBench training requires --epochs 600")
    if args.periodic_every < 0:
        raise ValueError("--periodic-every must be non-negative")

    output_dir = Path(args.output_dir).expanduser().resolve()
    scaler_path = Path(args.scaler_path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume.lower() == "none":
        resume: str | None = None
    elif args.resume.lower() == "auto":
        last_checkpoint = output_dir / "last.ckpt"
        resume = str(last_checkpoint) if last_checkpoint.is_file() else None
    else:
        resume_path = Path(os.path.expanduser(args.resume)).resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        resume = str(resume_path)

    all_episode_files = discover_episode_files(args.data_root, mode="all")
    if args.expected_episodes and len(all_episode_files) != args.expected_episodes:
        raise RuntimeError(
            f"Dataset completeness guard: expected exactly {args.expected_episodes} episodes, "
            f"found {len(all_episode_files)} below {Path(args.data_root).expanduser().resolve()}"
        )

    split_kwargs = {
        "root_dir": args.data_root,
        "future_steps": 16,
        "image_hw": (args.image_height, args.image_width),
        "val_fraction": args.val_fraction,
        "split_seed": args.split_seed,
        "action_target_offset": args.action_target_offset,
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
        print(f"Fitted official RGB EE scaler on train episodes only: {scaler_path}")
    train_dataset.scaler = scaler
    val_dataset = RGBTrajectoryDataset(mode="val", scaler=scaler, **split_kwargs)

    supplied_split_note = (
        "Only 50 flat episodes were supplied; deterministic 45/5 holdout is used."
        if len(all_episode_files) == 50
        and len(train_dataset.file_paths) == 45
        and len(val_dataset.file_paths) == 5
        else "Dataset uses the discovered explicit or deterministic train/validation split."
    )
    manifest = {
        "contract": "RMBench official training + real-world official RGB encoder",
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "available_episodes": len(all_episode_files),
        "official_readme_recommends": {"train": 50, "test": 5},
        "availability_note": supplied_split_note,
        "split_seed": args.split_seed,
        "seed": args.seed,
        "action_target_offset": args.action_target_offset,
        "image_hw": [args.image_height, args.image_width],
        "val_fraction": args.val_fraction,
        "train_episodes": [path.name for path in train_dataset.file_paths],
        "val_episodes": [path.name for path in val_dataset.file_paths],
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "accumulate_grad_batches": args.accumulate_grad_batches,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "eta_min": args.eta_min,
            "precision": args.precision,
            "vision_chunk_size": args.vision_chunk_size,
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote run contract: {manifest_path}")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": False,
        "persistent_workers": args.num_workers > 0,
        "collate_fn": parallel_collate_fn_rgb,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    config = MambaConfig()
    config.camera_names = ["head_camera"]
    config.embed_dim = 1024
    config.d_model = 1024
    config.lowdim_dim = 16
    config.action_dim = 16
    config.num_blocks = 6
    config.future_steps = 16
    model = LitMambaRGB(
        config,
        scaler=scaler,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        eta_min=args.eta_min,
        vision_chunk_size=args.vision_chunk_size,
        pretrained_backbone=not args.no_pretrained_backbone,
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
            save_top_k=5,
            save_last=True,
        )
    )
    if args.periodic_every:
        callbacks.append(
            ModelCheckpoint(
                dirpath=output_dir,
                filename="mamba-periodic-{epoch:04d}",
                every_n_epochs=args.periodic_every,
                save_top_k=-1,
                save_on_train_epoch_end=True,
            )
        )
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    logger = TensorBoardLogger(
        save_dir=output_dir.parent.parent,
        name=f"{args.task_name}_official_rgb",
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
        fast_dev_run=args.fast_dev_run,
    )

    print(
        "Training official-contract RGB Chronos: "
        f"train={len(train_dataset)}, val={len(val_dataset)}, resume={resume}, "
        "input=[RGB 480x640 + EE16], output=[16xEE16]"
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume)


if __name__ == "__main__":
    main()
