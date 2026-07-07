import argparse
import copy
import gc
import os
import sys

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from common.scaler_M import Scaler
from common.metric_UR10D import my_Metric
from common.mamba_policy_par_2D_IMLE import MambaPolicy, MambaConfig
from common.M_dataset_real_UR3 import ImageTrajectoryDataset, parallel_collate_fn_image

torch.serialization.add_safe_globals([MambaConfig])

DEFAULT_TASK_NAME = os.environ.get("CHRONOS_TASK_NAME", "cover_blocks")
DEFAULT_DATA_ROOT = os.environ.get(
    "CHRONOS_DATA_ROOT",
    os.path.join(PACKAGE_ROOT, "datasets", DEFAULT_TASK_NAME),
)
DEFAULT_OUTPUT_ROOT = os.environ.get("CHRONOS_OUTPUT_ROOT", PACKAGE_ROOT)


class WarmupEMACallback(Callback):
    def __init__(self, inv_gamma=1.0, power=2/3, min_value=0.0, max_value=0.9999):
        super().__init__()
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.ema_state = None
        self.original_state = None
        self.optimization_step = 0

    def get_decay(self, step):
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        if step <= 0:
            return 0.0
        return max(self.min_value, min(value, self.max_value))

    def on_fit_start(self, trainer, pl_module):
        self.ema_state = {
            k: v.clone().detach()
            for k, v in pl_module.state_dict().items()
            if v.dtype.is_floating_point
        }

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        decay = self.get_decay(self.optimization_step)
        self.optimization_step += 1
        with torch.no_grad():
            for k, v in pl_module.state_dict().items():
                if k in self.ema_state and v.dtype.is_floating_point:
                    if "bn" in k or "running_mean" in k or "running_var" in k:
                        self.ema_state[k].copy_(v.detach())
                    else:
                        self.ema_state[k].mul_(decay).add_(v.detach(), alpha=1 - decay)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.original_state = {k: v.clone().detach() for k, v in pl_module.state_dict().items()}
        pl_module.load_state_dict(self.ema_state, strict=False)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.original_state is not None:
            pl_module.load_state_dict(self.original_state, strict=False)
            self.original_state = None

    def on_save_checkpoint(self, trainer, pl_module, checkpoint_dict):
        checkpoint_dict["state_dict"] = {k: v.clone() for k, v in self.ema_state.items()}


class LitMambaParallel(pl.LightningModule):
    def __init__(self, config: MambaConfig, scaler: Scaler):
        super().__init__()
        self.save_hyperparameters(ignore=["scaler"])
        self.config = config
        self.scaler = scaler
        self.metric = my_Metric()

        self.policy = MambaPolicy(
            camera_names=config.camera_names,
            embed_dim=config.embed_dim,
            lowdim_dim=config.lowdim_dim,
            d_model=config.d_model,
            action_dim=config.action_dim,
            num_blocks=config.num_blocks,
            future_steps=config.future_steps,
            mamba_cfg=config,
        )

        self.lr = 1.7e-4
        self.weight_decay = 1e-4
        self.train_step_outputs = []
        self.val_step_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.policy.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        max_epochs = self.trainer.max_epochs
        warmup_epochs = 15
        eta_min = 3e-5
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs - warmup_epochs, eta_min=eta_min
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs]
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # 保持旧逻辑：只把小张量交给 Lightning 搬运；image 留在 CPU，由 chunk 流水线逐块搬到 GPU。
        batch["obs"] = batch["obs"].to(device)
        batch["actions"] = batch["actions"].to(device)
        batch["mask"] = batch["mask"].to(device)
        return batch

    def on_train_epoch_start(self):
        if self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            print(f"\n{'='*40}\nEpoch {self.current_epoch} | Real LR: {current_lr:.10f}\n{'='*40}")
            self.log("debug_lr", current_lr, prog_bar=True)

    def denormalize(self, actions: torch.Tensor) -> torch.Tensor:
        # pose10d layout: left [pose9, gripper], right [pose9, gripper]
        pose_dim = 9 if self.config.action_dim == 20 else 6
        arm_dim = pose_dim + 1

        data = {}
        for i in range(pose_dim):
            data[f"pose_l_{i+1}_act"] = actions[..., i:i + 1]
        data["gripper_l_act"] = actions[..., pose_dim:pose_dim + 1]
        r0 = arm_dim
        for i in range(pose_dim):
            data[f"pose_r_{i+1}_act"] = actions[..., r0 + i:r0 + i + 1]
        data["gripper_r_act"] = actions[..., r0 + pose_dim:r0 + pose_dim + 1]

        denorm = self.scaler.denormalize(data)
        out = torch.cat(
            [denorm[f"pose_l_{i+1}_act"] for i in range(pose_dim)] + [denorm["gripper_l_act"]] +
            [denorm[f"pose_r_{i+1}_act"] for i in range(pose_dim)] + [denorm["gripper_r_act"]],
            dim=-1,
        )
        return out

    def compute_image_features(self, image_batch, obs, mask=None):
        """
        安全且简洁的 2D chunk 流水线。

        核心原则：
        1. image_batch 留在 CPU；
        2. 每次只搬一个小 chunk 到 GPU；
        3. 不对 image requires_grad；
        4. 不用 checkpoint；
        5. ResNet18 冻结 no_grad，只训练后面的轻量融合层。
        """
        device = self.device
        B, T = obs.shape[:2]
        chunk_size = int(getattr(self.config, "image_chunk_size", 64))

        if mask is not None:
            lengths = (~mask).sum(dim=1).detach().cpu().tolist()
        else:
            lengths = [(obs[b] != 0).any(dim=-1).sum().item() for b in range(B)]

        fusion = self.policy.fusion_engine

        # 冻结视觉主干时，保持 eval 更干净
        if hasattr(fusion, "vision_net"):
            fusion.vision_net.eval()

        seq_list = []

        for b in range(B):
            L_b = int(lengths[b])
            feat_chunks = []

            for i in range(0, L_b, chunk_size):
                end = min(i + chunk_size, L_b)

                # image_batch 仍在 CPU，这里只搬当前 chunk
                img_chunk = image_batch[b, i:end].contiguous().to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=False,
                )

                # obs 已经在 GPU
                obs_chunk = obs[b, i:end].contiguous()

                feat = fusion(img_chunk, obs_chunk)
                
                feat_chunks.append(feat)

                del img_chunk, obs_chunk, feat

            if L_b > 0:
                x_b = torch.cat(feat_chunks, dim=0)
            else:
                x_b = torch.zeros(0, self.config.embed_dim, device=device)

            # pad 回 batch 内最大长度 T
            if L_b < T:
                pad = x_b.new_zeros(T - L_b, self.config.embed_dim)
                x_b = torch.cat([x_b, pad], dim=0)

            seq_list.append(x_b)

        x_fused_seq = torch.stack(seq_list, dim=0)
        return x_fused_seq

    def training_step(self, batch, batch_idx):
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        mask = batch["mask"].to(self.device)
        image_batch = batch["image"]

        x_fused_seq = self.compute_image_features(image_batch, obs, mask)
        loss_per_sample = self.policy.compute_loss(x_fused_seq, actions)

        loss_mask = (~mask).float()
        loss = (loss_per_sample * loss_mask).sum() / (loss_mask.sum() + 1e-6)

        self.log("train_loss", loss, prog_bar=True, batch_size=obs.shape[0])
        self.train_step_outputs.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        mask = batch["mask"].to(self.device)
        image_batch = batch["image"]

        x_fused_seq = self.compute_image_features(image_batch, obs, mask)
        loss_per_sample = self.policy.compute_loss(x_fused_seq, actions)

        loss_mask = (~mask).float()
        val_loss = (loss_per_sample * loss_mask).sum() / (loss_mask.sum() + 1e-6)

        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True, batch_size=obs.shape[0])
        self.val_step_outputs.append(val_loss.item())

        pred_actions = self.policy.sample_actions(x_fused_seq, steps=5)
        valid_indices = ~mask
        pred_valid = pred_actions[valid_indices]
        gt_valid = actions[valid_indices]

        pred_denorm = self.denormalize(pred_valid)
        gt_denorm = self.denormalize(gt_valid)
        self.metric.update(pred_denorm, gt_denorm)
        return val_loss

    def on_validation_epoch_end(self):
        avg_loss = sum(self.val_step_outputs) / len(self.val_step_outputs) if self.val_step_outputs else 0.0
        self.log("val_epoch_loss", avg_loss, prog_bar=True)
        self.val_step_outputs.clear()

        metric_results = self.metric.compute()
        print("\nValidation Metrics:")
        for k, v in metric_results.items():
            print(f"{k}: {v}")
            self.log(f"val_{k}", v, sync_dist=True)
        self.metric.reset()
        gc.collect()
        torch.cuda.empty_cache()

    def on_train_epoch_end(self):
        avg_loss = sum(self.train_step_outputs) / len(self.train_step_outputs) if self.train_step_outputs else 0.0
        self.log("train_epoch_loss", avg_loss)
        self.train_step_outputs.clear()
        gc.collect()
        torch.cuda.empty_cache()


def _parse_devices(value: str):
    value = str(value).strip()
    if value.lower() == "auto":
        return "auto"
    if "," in value:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(value)]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the real-world Chronos image policy.")
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Dataset root containing train/ and test/.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root for checkpoints, logs, and scalers.")
    parser.add_argument("--scaler-path", default=None, help="Path to a fitted scaler .pth file.")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--accumulate-grad-batches", type=int, default=3)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default="0", help='Lightning devices value, for example "0", "0,1", or "auto".')
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--image-chunk-size", type=int, default=256)
    parser.add_argument("--future-steps", type=int, default=16)
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from checkpoint-dir/last.ckpt.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    seed_everything(42)
    task_name = args.task_name
    use_pose10d = True

    config = MambaConfig()
    config.embed_dim = 1024
    config.d_model = 1024
    config.action_dim = 20 if use_pose10d else 14
    config.lowdim_dim = 20 if use_pose10d else 14
    config.num_blocks = 6
    config.future_steps = args.future_steps
    config.camera_names = ["head_camera"]
    config.image_chunk_size = args.image_chunk_size

    data_root = os.path.abspath(args.data_root)
    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")

    output_root = os.path.abspath(args.output_root)
    checkpoint_dir = args.checkpoint_dir or os.path.join(
        output_root,
        "checkpoints",
        task_name,
        f"S3B_IMAGE_{config.action_dim}D_2",
    )
    log_dir = args.log_dir or os.path.join(output_root, "logs")
    scaler_filename = args.scaler_path or os.path.join(
        output_root,
        "scalers",
        f"scaler_{task_name}_image_{'pose10d' if use_pose10d else 'pose6d'}.pth",
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    full_lowdim_dict = ImageTrajectoryDataset.make_lowdim_dict(
        future_steps=config.future_steps,
        use_pose10d=use_pose10d,
    )
    scaler_cpu = Scaler(lowdim_dict=full_lowdim_dict)
    if os.path.exists(scaler_filename):
        scaler_cpu.load(scaler_filename)
        print(f"[SSD-IO] Loading scaler from {scaler_filename}")
    else:
        raise FileNotFoundError(
            f"Scaler not found: {scaler_filename}. "
            "Run common/M_dataset_real_UR3.py to fit a scaler first."
        )

    scaler_gpu = copy.deepcopy(scaler_cpu)

    train_dataset = ImageTrajectoryDataset(
        root_dir=train_dir,
        mode="train",
        future_steps=config.future_steps,
        scaler=scaler_cpu,
        resize_hw=(640, 480),
        use_pose10d=use_pose10d,
    )
    val_dataset = ImageTrajectoryDataset(
        root_dir=test_dir,
        mode="test",
        future_steps=config.future_steps,
        scaler=scaler_cpu,
        resize_hw=(640, 480),
        use_pose10d=use_pose10d,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=False,
        collate_fn=parallel_collate_fn_image,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=False,
        collate_fn=parallel_collate_fn_image,
    )

    model = LitMambaParallel(config, scaler=scaler_gpu)

    logger = TensorBoardLogger(save_dir=log_dir, name=task_name)
    checkpoint_callback_best = ModelCheckpoint(
        monitor="val_epoch_loss",
        dirpath=checkpoint_dir,
        filename="mamba-best-{epoch:02d}-{val_epoch_loss:.4f}",
        save_top_k=5,
        mode="min",
    )
    checkpoint_callback_last = ModelCheckpoint(dirpath=checkpoint_dir, save_last=True)
    checkpoint_callback_periodic = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="mamba-periodic-{epoch:04d}",
        every_n_epochs=100,
        save_top_k=-1,
        save_on_train_epoch_end=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    ema_callback = WarmupEMACallback(max_value=0.9999)

    trainer = pl.Trainer(
        accumulate_grad_batches=args.accumulate_grad_batches,
        accelerator=args.accelerator,
        devices=_parse_devices(args.devices),
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback_best, checkpoint_callback_last, checkpoint_callback_periodic, lr_monitor, ema_callback],
        precision=args.precision,
        gradient_clip_val=1.0,
        log_every_n_steps=1,
    )

    last_ckpt = os.path.join(checkpoint_dir, "last.ckpt")
    if (not args.no_resume) and os.path.exists(last_ckpt):
        print(f"[SSD-Resume] {task_name} from {last_ckpt}")
        trainer.fit(model, train_loader, val_loader, ckpt_path=last_ckpt)
    else:
        print(f"[SSD-Fresh-Start] {task_name}")
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
