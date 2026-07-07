import math
import torch
import torch.nn.functional as F
from torchmetrics import Metric


def _normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / torch.clamp(torch.linalg.norm(x, dim=-1, keepdim=True), min=eps)


def rot6d_rows_to_mat(d6: torch.Tensor) -> torch.Tensor:
    """Row-based rot6d -> rotation matrix, consistent with the dataset conversion."""
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = _normalize(a1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = _normalize(b2)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def rot6d_geodesic_deg(pred6: torch.Tensor, gt6: torch.Tensor) -> torch.Tensor:
    pred_R = rot6d_rows_to_mat(pred6)
    gt_R = rot6d_rows_to_mat(gt6)
    rel = pred_R @ gt_R.transpose(-1, -2)
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos = torch.clamp((trace - 1.0) * 0.5, -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos) * (180.0 / math.pi)


class my_Metric(Metric):
    """
    Metric for pose10d action layout:
      left  arm: [pos3, rot6, gripper] => indices 0:10
      right arm: [pos3, rot6, gripper] => indices 10:20

    pred_action / gt_action expected shape: [N, future_steps, 20] or [..., 20].
    """
    def __init__(self):
        super().__init__()
        self.metric_names = [
            "left_pos_mse", "left_pos_mae",
            "left_rot6_mse", "left_rot6_mae", "left_rot_geodesic_deg",
            "left_gripper_mse", "left_gripper_mae",
            "right_pos_mse", "right_pos_mae",
            "right_rot6_mse", "right_rot6_mae", "right_rot_geodesic_deg",
            "right_gripper_mse", "right_gripper_mae",
            "all_mse", "all_mae",
        ]
        for name in self.metric_names:
            self.add_state(name, default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_action: torch.Tensor, gt_action: torch.Tensor):
        with torch.no_grad():
            if pred_action.shape[-1] != 20 or gt_action.shape[-1] != 20:
                raise ValueError(
                    f"metric_pose10d expects action dim 20, got pred={pred_action.shape}, gt={gt_action.shape}. "
                    f"If you keep 14D pose6d, use the old metric_J.py."
                )

            p = pred_action.float()
            g = gt_action.float()

            lp, gp = p[..., 0:3], g[..., 0:3]
            lr, gr = p[..., 3:9], g[..., 3:9]
            lgp, glgp = p[..., 9:10], g[..., 9:10]

            rp, grp = p[..., 10:13], g[..., 10:13]
            rr, grr = p[..., 13:19], g[..., 13:19]
            rgp, grgp = p[..., 19:20], g[..., 19:20]

            self.left_pos_mse += F.mse_loss(lp, gp)
            self.left_pos_mae += F.l1_loss(lp, gp)
            self.left_rot6_mse += F.mse_loss(lr, gr)
            self.left_rot6_mae += F.l1_loss(lr, gr)
            self.left_rot_geodesic_deg += rot6d_geodesic_deg(lr, gr).mean()
            self.left_gripper_mse += F.mse_loss(lgp, glgp)
            self.left_gripper_mae += F.l1_loss(lgp, glgp)

            self.right_pos_mse += F.mse_loss(rp, grp)
            self.right_pos_mae += F.l1_loss(rp, grp)
            self.right_rot6_mse += F.mse_loss(rr, grr)
            self.right_rot6_mae += F.l1_loss(rr, grr)
            self.right_rot_geodesic_deg += rot6d_geodesic_deg(rr, grr).mean()
            self.right_gripper_mse += F.mse_loss(rgp, grgp)
            self.right_gripper_mae += F.l1_loss(rgp, grgp)

            self.all_mse += F.mse_loss(p, g)
            self.all_mae += F.l1_loss(p, g)
            self.total += 1

    def compute(self):
        denom = torch.clamp(self.total.float(), min=1.0)
        return {name: getattr(self, name).float() / denom for name in self.metric_names}
