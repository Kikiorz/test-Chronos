import os
import sys
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

DEFAULT_CKPT_DIR = os.environ.get(
    "CHRONOS_CKPT_DIR",
    os.path.join(PACKAGE_ROOT, "checkpoints", "rearrange_cube", "S3B_IMAGE_20D"),
)
DEFAULT_CKPT_PATH = os.environ.get("CHRONOS_CKPT_PATH", os.path.join(DEFAULT_CKPT_DIR, "last.ckpt"))
DEFAULT_SCALER_PATH = os.environ.get(
    "CHRONOS_SCALER_PATH",
    os.path.join(PACKAGE_ROOT, "scalers", "scaler_rearrange_cube_image_pose10d.pth"),
)


try:
    from common import mamba_policy_par_2D_IMLE
    from common.mamba_policy_par_2D_IMLE import MambaConfig, MambaPolicy
except ImportError:
    import mamba_policy_par_2D_IMLE
    from mamba_policy_par_2D_IMLE import MambaConfig, MambaPolicy

from common.scaler_M import Scaler
from common.pose_util import pose6d_to_pose10d


# Checkpoints saved by the training script pickle the config as
# mamba_policy_par_2D_IMLE.MambaConfig. Register the local module before
# torch.load so unpickling works outside the original training tree.
sys.modules.setdefault("mamba_policy_par_2D_IMLE", mamba_policy_par_2D_IMLE)


def resolve_checkpoint_path(path: Optional[str]) -> str:
    if path is None:
        path = DEFAULT_CKPT_PATH

    if os.path.isdir(path):
        last_path = os.path.join(path, "last.ckpt")
        if os.path.exists(last_path):
            return last_path

        ckpts = [
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.endswith(".ckpt")
        ]
        if not ckpts:
            raise FileNotFoundError(f"No .ckpt files found in checkpoint dir: {path}")
        return max(ckpts, key=os.path.getmtime)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def make_pose10d_lowdim_dict(future_steps: int = 16) -> Dict[str, object]:
    obs_keys, act_keys = make_pose10d_keys()
    lowdim_dict: Dict[str, object] = {key: 1 for key in obs_keys}
    for key in act_keys:
        lowdim_dict[key] = (future_steps, 1)
    return lowdim_dict


def make_pose10d_keys() -> Tuple[list[str], list[str]]:
    obs_keys = (
        [f"pose_l_{i + 1}" for i in range(9)] + ["gripper_l"] +
        [f"pose_r_{i + 1}" for i in range(9)] + ["gripper_r"]
    )
    act_keys = [key + "_act" for key in obs_keys]
    return obs_keys, act_keys


def build_default_config() -> MambaConfig:
    config = MambaConfig()
    config.embed_dim = 1024
    config.d_model = 1024
    config.action_dim = 20
    config.lowdim_dim = 20
    config.num_blocks = 6
    config.future_steps = 16
    config.camera_names = ["head_camera"]
    config.image_chunk_size = 256
    return config


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    requested = str(device) if device is not None else "auto"
    if requested.lower() == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"

    torch_device = torch.device(requested)
    if torch_device.type != "cuda":
        return torch_device

    if not torch.cuda.is_available():
        print("[ChronosInference] CUDA requested but torch.cuda.is_available() is False; using CPU.")
        return torch.device("cpu")

    try:
        index = torch_device.index if torch_device.index is not None else torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(index)
        device_arch = f"sm_{capability[0]}{capability[1]}"
        supported_arches = set(torch.cuda.get_arch_list())
        if supported_arches and device_arch not in supported_arches and f"compute_{capability[0]}{capability[1]}" not in supported_arches:
            print(
                "[ChronosInference] CUDA device is "
                f"{torch.cuda.get_device_name(index)} ({device_arch}), but this PyTorch build supports "
                f"{sorted(supported_arches)}. Falling back to CPU."
            )
            return torch.device("cpu")

        probe = torch.empty(1, device=torch_device)
        probe = probe + 1
        torch.cuda.synchronize(torch_device)
        del probe
        return torch_device
    except Exception as exc:
        print(f"[ChronosInference] CUDA probe failed on {torch_device}: {exc}. Falling back to CPU.")
        return torch.device("cpu")


class TorchRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, norm_before_gate=False, group_size=None, **factory_kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, **factory_kwargs))
        self.eps = eps
        self.norm_before_gate = norm_before_gate
        self.group_size = group_size or hidden_size

    def _norm(self, x):
        if self.group_size and self.group_size != x.shape[-1]:
            shape = x.shape
            x_grouped = x.reshape(*shape[:-1], -1, self.group_size)
            rms = torch.rsqrt(x_grouped.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            x = (x_grouped * rms).reshape(shape)
        else:
            x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * self.weight

    def forward(self, x, z=None):
        if z is not None and self.norm_before_gate:
            return self._norm(x) * F.silu(z)
        if z is not None:
            x = x * F.silu(z)
        return self._norm(x)


def patch_cpu_mamba_kernels():
    mamba_policy_par_2D_IMLE.RMSNormGated = TorchRMSNormGated
    mamba_policy_par_2D_IMLE.selective_state_update = None
    mamba_policy_par_2D_IMLE.causal_conv1d_update = None


def configure_cpu_runtime():
    raw_threads = os.environ.get("CHRONOS_CPU_THREADS", "4")
    try:
        num_threads = max(1, int(raw_threads))
    except ValueError:
        num_threads = 4

    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    print(
        "[ChronosInference] CPU runtime: "
        f"torch num_threads={torch.get_num_threads()} "
        "(override with CHRONOS_CPU_THREADS)."
    )


def _patch_resnet18_no_download():
    models = getattr(mamba_policy_par_2D_IMLE, "models", None)
    if models is None or getattr(models, "_chronos_no_download_patch", False):
        return

    original_resnet18 = models.resnet18

    def resnet18_no_download(*args, **kwargs):
        kwargs.pop("pretrained", None)
        kwargs["weights"] = None
        try:
            return original_resnet18(*args, **kwargs)
        except TypeError:
            kwargs.pop("weights", None)
            kwargs["pretrained"] = False
            return original_resnet18(*args, **kwargs)

    models.resnet18 = resnet18_no_download
    models._chronos_no_download_patch = True


class MyInferenceModel(nn.Module):
    """
    Real-world Chronos inference wrapper for the 2D IMAGE 20D model.

    Observation layout is identical to M_dataset_real_UR3.py:
      [left pose9, left gripper, right pose9, right gripper]

    Action layout is the same 20D pose10d-style target:
      [left target pose9, left gripper target, right target pose9, right gripper target]
    """

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CKPT_PATH,
        scaler_path: str = DEFAULT_SCALER_PATH,
        config: Optional[MambaConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
        temporal_agg: bool = False,
        sample_steps: int = 5,
        avoid_resnet_download: bool = True,
    ):
        super().__init__()
        self.device = resolve_device(device)
        print(f"[ChronosInference] Using device: {self.device}")
        if self.device.type == "cpu":
            configure_cpu_runtime()
        self.config = config or build_default_config()
        self.future_steps = int(getattr(self.config, "future_steps", 16))
        self.action_dim = int(getattr(self.config, "action_dim", 20))
        self.sample_steps = sample_steps
        self.temporal_agg = temporal_agg

        if self.action_dim != 20:
            raise ValueError(f"Chronos IMAGE pose10d inference expects action_dim=20, got {self.action_dim}")

        self.obs_keys, self.act_keys = make_pose10d_keys()
        self.lowdim_dict = make_pose10d_lowdim_dict(self.future_steps)

        self.scaler = Scaler(lowdim_dict=self.lowdim_dict)
        if scaler_path is None:
            scaler_path = DEFAULT_SCALER_PATH
        self.scaler_path = scaler_path
        print(f"[ChronosInference] Loading scaler from {self.scaler_path}")
        self.scaler.load(self.scaler_path)
        self.scaler.to(self.device)

        if avoid_resnet_download:
            _patch_resnet18_no_download()
        if self.device.type == "cpu":
            patch_cpu_mamba_kernels()

        print("[ChronosInference] Initializing 2D MambaPolicy...")
        self.policy = MambaPolicy(
            camera_names=self.config.camera_names,
            embed_dim=self.config.embed_dim,
            lowdim_dim=self.config.lowdim_dim,
            d_model=self.config.d_model,
            action_dim=self.config.action_dim,
            num_blocks=self.config.num_blocks,
            future_steps=self.future_steps,
            mamba_cfg=self.config,
        )
        self.policy.to(self.device)
        self.policy.eval()

        ckpt_path = resolve_checkpoint_path(checkpoint_path)
        self.checkpoint_path = ckpt_path
        print(f"[ChronosInference] Loading weights from {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        policy_state = {}
        for key, value in state_dict.items():
            if key.startswith("policy."):
                policy_state[key[len("policy."):]] = value
            elif not key.startswith("scaler.") and not key.startswith("metric."):
                policy_state[key] = value

        missing, unexpected = self.policy.load_state_dict(policy_state, strict=False)
        print(
            "[ChronosInference] Weights loaded "
            f"(missing={len(missing)}, unexpected={len(unexpected)})."
        )
        if missing[:5]:
            print(f"[ChronosInference] Missing sample: {missing[:5]}")
        if unexpected[:5]:
            print(f"[ChronosInference] Unexpected sample: {unexpected[:5]}")

        self.hiddens = None
        self.max_timesteps = 3000
        self.t = 0
        if self.temporal_agg:
            all_time_actions = torch.zeros(
                self.future_steps,
                self.future_steps,
                self.action_dim,
                device=self.device,
            )
            all_time_valid = torch.zeros(
                self.future_steps,
                self.future_steps,
                dtype=torch.bool,
                device=self.device,
            )
        else:
            all_time_actions = torch.empty(0, device=self.device)
            all_time_valid = torch.empty(0, dtype=torch.bool, device=self.device)
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer("all_time_actions", all_time_actions, persistent=False)
        self.register_buffer("all_time_valid", all_time_valid, persistent=False)
        self.reset_hiddens()

    def reset_hiddens(self, verbose: bool = True):
        self.hiddens = self.policy.init_hidden_states(batch_size=1, device=self.device)
        self.t = 0
        if self.temporal_agg:
            self.all_time_actions.zero_()
            self.all_time_valid.zero_()
        if hasattr(self.policy, "_current_z"):
            self.policy._current_z = None
        if verbose:
            print("[ChronosInference] Hidden states reset.")

    def preprocess_image(self, image_bgr: np.ndarray) -> torch.Tensor:
        if image_bgr is None:
            raise ValueError("image_bgr is None")
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        if image_bgr.shape[-1] == 4:
            image_bgr = image_bgr[..., :3]

        image_bgr = cv2.resize(image_bgr, (640, 480), interpolation=cv2.INTER_AREA)
        image_rgb = image_bgr[:, :, [2, 1, 0]].astype(np.float32) / 255.0
        image = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float().unsqueeze(0)
        image = image.to(self.device, non_blocking=False)
        return (image - self.image_mean) / self.image_std

    def build_qpos(
        self,
        pose_l_6d: np.ndarray,
        pose_r_6d: np.ndarray,
        gripper_l: Union[int, float, np.ndarray],
        gripper_r: Union[int, float, np.ndarray],
    ) -> torch.Tensor:
        pose_l = pose6d_to_pose10d(np.asarray(pose_l_6d, dtype=np.float32)[None, :])
        pose_r = pose6d_to_pose10d(np.asarray(pose_r_6d, dtype=np.float32)[None, :])
        grip_l = np.asarray(gripper_l, dtype=np.float32).reshape(1, 1)
        grip_r = np.asarray(gripper_r, dtype=np.float32).reshape(1, 1)
        qpos = np.concatenate([pose_l, grip_l, pose_r, grip_r], axis=-1).astype(np.float32)
        return torch.from_numpy(qpos).to(self.device)

    def normalize_qpos(self, qpos_raw: torch.Tensor) -> torch.Tensor:
        qpos_raw = qpos_raw.to(self.device, dtype=torch.float32)
        if qpos_raw.dim() == 1:
            qpos_raw = qpos_raw.unsqueeze(0)
        lowdim_input = {key: qpos_raw[:, i:i + 1] for i, key in enumerate(self.obs_keys)}
        norm = self.scaler.normalize(lowdim_input)
        return torch.cat([norm[key] for key in self.obs_keys], dim=-1).float()

    def denormalize(self, actions_norm: torch.Tensor) -> torch.Tensor:
        actions_norm = actions_norm.to(self.device, dtype=torch.float32)
        pose_dim = 9
        arm_dim = pose_dim + 1
        data = {}
        for i in range(pose_dim):
            data[f"pose_l_{i + 1}_act"] = actions_norm[..., i:i + 1]
        data["gripper_l_act"] = actions_norm[..., pose_dim:pose_dim + 1]
        r0 = arm_dim
        for i in range(pose_dim):
            data[f"pose_r_{i + 1}_act"] = actions_norm[..., r0 + i:r0 + i + 1]
        data["gripper_r_act"] = actions_norm[..., r0 + pose_dim:r0 + pose_dim + 1]

        denorm = self.scaler.denormalize(data)
        return torch.cat(
            [denorm[f"pose_l_{i + 1}_act"] for i in range(pose_dim)] + [denorm["gripper_l_act"]] +
            [denorm[f"pose_r_{i + 1}_act"] for i in range(pose_dim)] + [denorm["gripper_r_act"]],
            dim=-1,
        )

    @torch.inference_mode()
    def _predict_action_sequence_norm(
        self,
        qpos_raw: Union[np.ndarray, torch.Tensor],
        image_bgr: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(qpos_raw, np.ndarray):
            qpos_raw = torch.from_numpy(qpos_raw).to(self.device)
        qpos_norm = self.normalize_qpos(qpos_raw)

        if isinstance(image_bgr, torch.Tensor):
            image = image_bgr.to(self.device, dtype=torch.float32)
            if image.dim() == 3:
                image = image.unsqueeze(0)
        else:
            image = self.preprocess_image(image_bgr)

        x_fused_step = self.policy.fusion_engine(image, qpos_norm)
        pred_norm, self.hiddens = self.policy.step(
            x_fused_step,
            self.hiddens,
            sample_steps=self.sample_steps,
        )
        return pred_norm

    @torch.inference_mode()
    def predict_action_sequence_norm(
        self,
        qpos_raw: Union[np.ndarray, torch.Tensor],
        image_bgr: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        return self._predict_action_sequence_norm(qpos_raw, image_bgr)[0]

    @torch.inference_mode()
    def select_action_from_sequence_norm(
        self,
        sequence_norm: torch.Tensor,
        execute_step_offset: int = 0,
    ) -> torch.Tensor:
        sequence_norm = sequence_norm.to(self.device, dtype=torch.float32)
        execute_step_offset = int(np.clip(execute_step_offset, 0, self.future_steps - 1))

        if not self.temporal_agg:
            action_norm = sequence_norm[execute_step_offset]
        else:
            row = self.t % self.future_steps
            self.all_time_actions[row].zero_()
            self.all_time_valid[row].zero_()
            self.all_time_actions[row, :self.future_steps] = sequence_norm
            self.all_time_valid[row, :self.future_steps] = True

            execute_t = self.t + execute_step_offset
            candidates = []
            for source_t in range(max(0, execute_t - self.future_steps + 1), self.t + 1):
                horizon_idx = execute_t - source_t
                if horizon_idx < 0 or horizon_idx >= self.future_steps:
                    continue
                source_row = source_t % self.future_steps
                if self.all_time_valid[source_row, horizon_idx]:
                    candidates.append(self.all_time_actions[source_row, horizon_idx])

            if not candidates:
                action_norm = sequence_norm[execute_step_offset]
            else:
                actions_for_exec_step = torch.stack(candidates, dim=0)
                weights = np.exp(-0.01 * np.arange(len(actions_for_exec_step)))
                weights = torch.from_numpy(weights / weights.sum()).to(self.device, dtype=torch.float32)
                action_norm = (actions_for_exec_step * weights.unsqueeze(1)).sum(dim=0)
            self.t += 1

        return self.denormalize(action_norm.unsqueeze(0))[0]

    @torch.inference_mode()
    def predict_action_sequence(
        self,
        qpos_raw: Union[np.ndarray, torch.Tensor],
        image_bgr: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        pred_norm = self._predict_action_sequence_norm(qpos_raw, image_bgr)
        pred = self.denormalize(pred_norm)
        return pred[0].detach().cpu().numpy()

    @torch.inference_mode()
    def get_action(self, obs_dict: Dict[str, Union[np.ndarray, torch.Tensor]]) -> np.ndarray:
        sequence_norm = self.predict_action_sequence_norm(obs_dict["qpos"], obs_dict["image"])
        execute_step_offset = int(obs_dict.get("execute_step_offset", 0))
        action = self.select_action_from_sequence_norm(sequence_norm, execute_step_offset)
        return action.detach().cpu().numpy()

    @torch.inference_mode()
    def predict_from_raw(
        self,
        pose_l_6d: np.ndarray,
        pose_r_6d: np.ndarray,
        gripper_l: Union[int, float, np.ndarray],
        gripper_r: Union[int, float, np.ndarray],
        image_bgr: np.ndarray,
    ) -> np.ndarray:
        qpos = self.build_qpos(pose_l_6d, pose_r_6d, gripper_l, gripper_r)
        return self.predict_action_sequence(qpos, image_bgr)

    def warmup(self):
        image = torch.zeros(1, 3, 480, 640, device=self.device)
        qpos = torch.zeros(1, self.action_dim, device=self.device)
        _ = self.predict_action_sequence(qpos, image)
        self.reset_hiddens()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        print("[ChronosInference] Warmup finished.")
