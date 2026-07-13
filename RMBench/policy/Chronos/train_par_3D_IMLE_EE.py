import os
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from scaler_M import Scaler
from metric_E import my_Metric
from mamba_policy_par_3D_IMLE import MambaPolicy, MambaConfig
from M_dataset_robotwin3D_E import PointCloudTrajectoryDataset, parallel_collate_fn_3d
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import cv2
import numpy as np
import io
import threading
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR
import torch.utils.checkpoint as checkpoint
from concurrent.futures import ThreadPoolExecutor
import torch
torch.serialization.add_safe_globals([MambaConfig])
from pytorch_lightning.callbacks import Callback
from torch.nn.modules.batchnorm import _BatchNorm

import gc

# 放在 import 之后
class CPUContainer:
    """
    一个简单的容器，用于欺骗 Lightning，
    防止它自动把 rgb_batch 移动到 GPU 导致显存爆炸。
    """
    def __init__(self, data):
        self.data = data
    
    def to(self, *args, **kwargs):
        # 无论 Lightning 怎么调用 .to()，我们都原地不动
        return self

class WarmupEMACallback(Callback):
    """
    带预热指数滑动平均机制。
    完美兼容 PyTorch Lightning 的验证与保存周期。
    """
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
        # 深度拷贝一份浮点数权重作为 EMA 的初始状态
        self.ema_state = {
            k: v.clone().detach() 
            for k, v in pl_module.state_dict().items() if v.dtype.is_floating_point
        }

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        decay = self.get_decay(self.optimization_step)
        self.optimization_step += 1
        
        with torch.no_grad():
            for k, v in pl_module.state_dict().items():
                if k in self.ema_state and v.dtype.is_floating_point:
                    # 如果是 BatchNorm 这种统计量，直接复制
                    if 'bn' in k or 'running_mean' in k or 'running_var' in k:
                        self.ema_state[k].copy_(v.detach())
                    else:
                        # EMA 核心公式
                        self.ema_state[k].mul_(decay).add_(v.detach(), alpha=1 - decay)

    def on_validation_epoch_start(self, trainer, pl_module):
        # 验证开始前，拦截模型，换上 EMA 权重
        self.original_state = {k: v.clone().detach() for k, v in pl_module.state_dict().items()}
        pl_module.load_state_dict(self.ema_state, strict=False)

    def on_validation_epoch_end(self, trainer, pl_module):
        # 验证结束后，把训练权重还回去
        if self.original_state is not None:
            pl_module.load_state_dict(self.original_state, strict=False)
            self.original_state = None

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # 存入硬盘的检查点，内部状态替换为 EMA 平滑权重
        checkpoint['state_dict'] = {k: v.clone() for k, v in self.ema_state.items()}
# ====================================================================================
class LitMambaParallel(pl.LightningModule):
    def __init__(self, config: MambaConfig, scaler: Scaler):
        super().__init__()
        self.save_hyperparameters(ignore=["scaler"])
        self.config = config
        self.scaler = scaler
        self.metric = my_Metric() 
        
        self.policy = MambaPolicy(
            camera_names = config.camera_names,
            embed_dim = config.embed_dim,
            lowdim_dim = config.lowdim_dim,  # 此处应该传入 16
            d_model = config.d_model,
            action_dim = config.action_dim,  # 此处应该传入 16
            num_blocks = config.num_blocks,
            future_steps = config.future_steps,
            mamba_cfg = config
        )
            
        self.lr = 1.7e-4
        self.weight_decay = 1e-4
        self.train_step_outputs = []
        self.val_step_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.policy.parameters()), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        max_epochs = self.trainer.max_epochs 
        warmup_epochs = 15
        eta_min = 2e-5
        
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
# ====================================================================================

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch['obs'] = batch['obs'].to(device)
        batch['actions'] = batch['actions'].to(device)
        batch['mask'] = batch['mask'].to(device)
        return batch

    def on_train_epoch_start(self):
        if self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            print(f"\n{'='*40}\nEpoch {self.current_epoch} | Real LR: {current_lr:.10f}\n{'='*40}")
            self.log("debug_lr", current_lr, prog_bar=True)
    
    def denormalize(self, actions):
        # 16维 EE 解包
        arm1_dict = {
            'ee_l_x_act': actions[..., 0:1], 'ee_l_y_act': actions[..., 1:2], 'ee_l_z_act': actions[..., 2:3],
            'ee_l_qw_act': actions[..., 3:4], 'ee_l_qx_act': actions[..., 4:5], 'ee_l_qy_act': actions[..., 5:6], 'ee_l_qz_act': actions[..., 6:7],
            'gripper_l_act': actions[..., 7:8]
        }
        arm2_dict = {
            'ee_r_x_act': actions[..., 8:9], 'ee_r_y_act': actions[..., 9:10], 'ee_r_z_act': actions[..., 10:11],
            'ee_r_qw_act': actions[..., 11:12], 'ee_r_qx_act': actions[..., 12:13], 'ee_r_qy_act': actions[..., 13:14], 'ee_r_qz_act': actions[..., 14:15],
            'gripper_r_act': actions[..., 15:16]
        }
        arm1_denorm = self.scaler.denormalize(arm1_dict)
        arm2_denorm = self.scaler.denormalize(arm2_dict)
        out = torch.cat([
            arm1_denorm['ee_l_x_act'], arm1_denorm['ee_l_y_act'], arm1_denorm['ee_l_z_act'], 
            arm1_denorm['ee_l_qw_act'], arm1_denorm['ee_l_qx_act'], arm1_denorm['ee_l_qy_act'], arm1_denorm['ee_l_qz_act'], arm1_denorm['gripper_l_act'],
            arm2_denorm['ee_r_x_act'], arm2_denorm['ee_r_y_act'], arm2_denorm['ee_r_z_act'], 
            arm2_denorm['ee_r_qw_act'], arm2_denorm['ee_r_qx_act'], arm2_denorm['ee_r_qy_act'], arm2_denorm['ee_r_qz_act'], arm2_denorm['gripper_r_act']
        ], dim=2)
        return out
    
    def compute_pointcloud_features(self, pc_batch, obs):
        device = self.device
        B = obs.shape[0]
        lengths = [(obs[b] != 0).any(dim=-1).sum().item() for b in range(B)]
        max_len = max(lengths)
        
        if isinstance(pc_batch, list):
            N, dim = pc_batch[0].shape[1:]
            pc_tensor = torch.zeros(B, max_len, N, dim, device=device)
            for b in range(B):
                L_b = lengths[b]
                pc_tensor[b, :L_b] = pc_batch[b].to(device, non_blocking=True)
        else:
            pc_tensor = pc_batch.to(device, non_blocking=True)
            
        proprio_raw = obs.to(device)
        valid_mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
        for b in range(B):
            valid_mask[b, :lengths[b]] = True
            
        flat_pc = pc_tensor[valid_mask] 
        flat_proprio = proprio_raw[valid_mask]
        
        # 融合网络现在只返回 1024 维的联合特征
        flat_x_fused = self.policy.fusion_engine(flat_pc, flat_proprio)

        x_fused_seq = torch.zeros(B, max_len, self.config.embed_dim, device=device)
        x_fused_seq[valid_mask] = flat_x_fused

        return x_fused_seq

    ## When your GPU OOM, use this version:

    # def compute_pointcloud_features(self, pc_batch, obs):
    #     """
    #     内部硬编码 chunk_size 彻底解决Activation OOM 问题。
    #     """
    #     device = self.device
    #     B = obs.shape[0]
    #
    #     # =========================================================
    #     # [显存阀门] 内部控制切块大小
    #     # RTX 4090 推荐设为 128 或 256。
    #     # =========================================================
    #     chunk_size = 128
    #
    #     # 1. 解析真实序列长度
    #     lengths = [(obs[b] != 0).any(dim=-1).sum().item() for b in range(B)]
    #     max_len = max(lengths)
    #
    #     # 2. 预分配最终的容器
    #     mamba_in_seq = torch.zeros(B, max_len, self.config.embed_dim, device=device)
    #
    #     # 将本体感移至 GPU (14维特征非常小)
    #     obs_gpu = obs.to(device, non_blocking=True)
    #
    #     # 包装 fusion_engine 以便传入 checkpoint
    #     def fusion_wrapper(pc_c, obs_c):
    #         return self.policy.fusion_engine(pc_c, obs_c)
    #
    #     for b in range(B):
    #         L_b = lengths[b]
    #         if L_b == 0: continue
    #
    #         # 3. 获取当前 batch 的完整点云序列 (留在 CPU)
    #         if isinstance(pc_batch, list):
    #             pc_b = pc_batch[b]  # [L_b, N, C]
    #         else:
    #             pc_b = pc_batch[b, :L_b]
    #
    #         # =========================================================
    #         # 4. 时间维度异步切块循环 (Temporal Chunking)
    #         # =========================================================
    #         for i in range(0, L_b, chunk_size):
    #             actual_len = min(chunk_size, L_b - i)
    #
    #             # 从 CPU 切割出当前的 chunk
    #             pc_chunk_cpu = pc_b[i: i + actual_len]
    #             # 获取对应的本体感 chunk
    #             obs_chunk_gpu = obs_gpu[b, i: i + actual_len]
    #
    #             # 只有当前 chunk 会被送入 GPU
    #             # 显存峰值被严格锁死在 [chunk_size, N, C] 的计算量内
    #             pc_chunk_gpu = pc_chunk_cpu.to(device, non_blocking=True)
    #
    #             # [核心魔法]: 欺骗 Checkpoint 触发重计算机制
    #             # 由于输入数据本身不需要求导，我们需要手动加上 requires_grad_()
    #             if self.training:
    #                 pc_chunk_gpu.requires_grad_()
    #
    #             if self.training and pc_chunk_gpu.requires_grad:
    #                 # 使用梯度检查点，use_reentrant=False 是 PyTorch 最新安全规范
    #                 vis_feat = checkpoint.checkpoint(
    #                     fusion_wrapper, pc_chunk_gpu, obs_chunk_gpu, use_reentrant=False
    #                 )
    #             else:
    #                 # 推理模式，直接前向
    #                 vis_feat = self.policy.fusion_engine(pc_chunk_gpu, obs_chunk_gpu)
    #
    #             # 5. 写入预分配的容器中 (只存储高浓缩的 1024 维语义)
    #             mamba_in_seq[b, i: i + actual_len] = vis_feat
    #
    #             # 6. 阅后即焚，立刻释放前向计算产生的激活值碎片
    #             del pc_chunk_gpu, vis_feat
    #
    #     return mamba_in_seq

    def training_step(self, batch, batch_idx):
        obs = batch['obs'].to(self.device)             
        actions = batch['actions'].to(self.device)   
        mask = batch['mask'].to(self.device)         
        pc_batch = batch['point_cloud']          
        
        x_fused_seq = self.compute_pointcloud_features(pc_batch, obs)
        
        loss_per_sample = self.policy.compute_loss(x_fused_seq, actions)
        
        loss_mask = (~mask).float() 
        loss = (loss_per_sample * loss_mask).sum() / (loss_mask.sum() + 1e-6)
        
        self.log("train_loss", loss, prog_bar=True, batch_size=obs.shape[0])
        self.train_step_outputs.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        obs = batch['obs'].to(self.device)
        actions = batch['actions'].to(self.device)
        mask = batch['mask'].to(self.device)
        pc_batch = batch['point_cloud'] 
        
        x_fused_seq = self.compute_pointcloud_features(pc_batch, obs)
        loss_per_sample = self.policy.compute_loss(x_fused_seq, actions)
        
        loss_mask = (~mask).float()
        val_loss = (loss_per_sample * loss_mask).sum() / (loss_mask.sum() + 1e-6)
        
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True, batch_size=obs.shape[0])
        self.val_step_outputs.append(val_loss.item())
        
        pred_actions_flat = self.policy.sample_actions(x_fused_seq, steps=5) 
        
        valid_indices = ~mask
        pred_valid = pred_actions_flat[valid_indices]
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

def main():
    seed_everything(42)
    TASK_NAME = "cover_blocks"
    SSD_ROOT = "/home/sutai/data" ### change to your own path
    CODE_ROOT = "/home/sutai/data2/ZYL/RMBench" 

    config = MambaConfig()
    config.embed_dim = 1024
    config.d_model = 1024
    config.action_dim = 16
    config.lowdim_dim = 16
    config.num_blocks = 6
    config.future_steps = 16

    DATA_ROOT = f"{SSD_ROOT}/{TASK_NAME}/demo_clean/data"
    TRAIN_DIR = f"{DATA_ROOT}/train"
    TEST_DIR  = f"{DATA_ROOT}/test"
    
    CHECKPOINT_DIR = f"{CODE_ROOT}/policy/Chronos/checkpoints/{TASK_NAME}/EE_16"
    LOG_DIR = f"{CODE_ROOT}/policy/Chronos/logs"
    SCALER_FILENAME = f"scaler_{TASK_NAME}_ee_3d.pth" 

    # EE 16维 Keys
    OBS_KEYS = [
        'ee_l_x', 'ee_l_y', 'ee_l_z', 'ee_l_qw', 'ee_l_qx', 'ee_l_qy', 'ee_l_qz', 'gripper_l',
        'ee_r_x', 'ee_r_y', 'ee_r_z', 'ee_r_qw', 'ee_r_qx', 'ee_r_qy', 'ee_r_qz', 'gripper_r'
    ]
    ACT_KEYS = [k + '_act' for k in OBS_KEYS]

    full_lowdim_dict = {k: 1 for k in OBS_KEYS}
    for k in ACT_KEYS: full_lowdim_dict[k] = (config.future_steps, 1)
    
    scaler_cpu = Scaler(lowdim_dict=full_lowdim_dict)
    if os.path.exists(SCALER_FILENAME):
        scaler_cpu.load(SCALER_FILENAME)
        print(f"�� [SSD-IO] Loading EE scaler from {SCALER_FILENAME}")
    else:
        print(f"⚠️ Scaler Error: {SCALER_FILENAME} not found. Ensure fit_scaler is run.")
        
    scaler_gpu = copy.deepcopy(scaler_cpu)

    train_dataset = PointCloudTrajectoryDataset(
        root_dir=TRAIN_DIR, mode="train", future_steps=config.future_steps, scaler=scaler_cpu
    )
    val_dataset = PointCloudTrajectoryDataset(
        root_dir=TEST_DIR, mode="test", future_steps=config.future_steps, scaler=scaler_cpu
    )
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=2, 
                              shuffle=True, 
                            #   num_workers=2,
                            #   persistent_workers=True,
                              collate_fn=parallel_collate_fn_3d)
    val_loader = DataLoader(val_dataset, 
                            batch_size=2, 
                            shuffle=False, 
                            # num_workers=2,
                            # persistent_workers=True,
                            collate_fn=parallel_collate_fn_3d)

    model = LitMambaParallel(config, scaler=scaler_gpu)
    

    logger = TensorBoardLogger(save_dir=LOG_DIR, name=TASK_NAME)
    checkpoint_callback_best = ModelCheckpoint(
        monitor='val_epoch_loss', dirpath=CHECKPOINT_DIR,
        filename='mamba-best-{epoch:02d}-{val_epoch_loss:.4f}', save_top_k=5, mode='min'
    )
    checkpoint_callback_last = ModelCheckpoint(dirpath=CHECKPOINT_DIR, save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # ==================== 每 100 Epoch 定期保存 ====================
    checkpoint_callback_periodic = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename='mamba-periodic-{epoch:04d}', # 命名例如 mamba-periodic-0100.ckpt
        every_n_epochs=100,  # 每 100 个 epoch 触发一次保存
        save_top_k=-1,       # -1 代表不覆盖、不删除之前的周期 ckpt（强制保留每一个百轮节点）
        save_on_train_epoch_end=True # 确保在训练轮次结束时精准触发
    )
    ema_callback = WarmupEMACallback(max_value=0.9999)
    trainer = pl.Trainer(
        accumulate_grad_batches=3,
        accelerator='gpu', 
        devices=[0], 
        max_epochs=600, 
        logger=logger,
        callbacks=[checkpoint_callback_best, checkpoint_callback_last, checkpoint_callback_periodic, lr_monitor, ema_callback],
        precision=32, 
        gradient_clip_val=1.0, 
        log_every_n_steps=1,
        
    )
    
    last_ckpt = f"{CHECKPOINT_DIR}/last.ckpt"
    if os.path.exists(last_ckpt):
        print(f"�� SSD-Resume: {TASK_NAME} from {last_ckpt}")
        trainer.fit(model, train_loader, val_loader, ckpt_path=last_ckpt)
    else:
        print(f"�� SSD-Fresh-Start: {TASK_NAME}")
        trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
