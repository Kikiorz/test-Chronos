import torch
from torchmetrics import Metric
import torch.nn.functional as F

class my_Metric(Metric):
    def __init__(self):
        super(my_Metric, self).__init__()
        
        # EE 专属指标 (16维)
        self.metric_names = [
            'ee_l_pos_mse', 'ee_l_pos_mae',
            'ee_l_quat_mse', 'ee_l_quat_mae',
            'gripper_l_mse', 'gripper_l_mae',
            'ee_r_pos_mse', 'ee_r_pos_mae',
            'ee_r_quat_mse', 'ee_r_quat_mae',
            'gripper_r_mse', 'gripper_r_mae'
        ]

        for metric_name in self.metric_names:
            self.add_state(metric_name, default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_action, gt_action):
        with torch.no_grad():
            # 左臂 POS (0:3)
            self.ee_l_pos_mse += F.mse_loss(pred_action[..., 0:3], gt_action[..., 0:3])
            self.ee_l_pos_mae += F.l1_loss(pred_action[..., 0:3], gt_action[..., 0:3])
            # 左臂 QUAT (3:7)
            self.ee_l_quat_mse += F.mse_loss(pred_action[..., 3:7], gt_action[..., 3:7])
            self.ee_l_quat_mae += F.l1_loss(pred_action[..., 3:7], gt_action[..., 3:7])
            # 左夹爪 (7:8)
            self.gripper_l_mse += F.mse_loss(pred_action[..., 7:8], gt_action[..., 7:8])
            self.gripper_l_mae += F.l1_loss(pred_action[..., 7:8], gt_action[..., 7:8])

            # 右臂 POS (8:11)
            self.ee_r_pos_mse += F.mse_loss(pred_action[..., 8:11], gt_action[..., 8:11])
            self.ee_r_pos_mae += F.l1_loss(pred_action[..., 8:11], gt_action[..., 8:11])
            # 右臂 QUAT (11:15)
            self.ee_r_quat_mse += F.mse_loss(pred_action[..., 11:15], gt_action[..., 11:15])
            self.ee_r_quat_mae += F.l1_loss(pred_action[..., 11:15], gt_action[..., 11:15])
            # 右夹爪 (15:16)
            self.gripper_r_mse += F.mse_loss(pred_action[..., 15:16], gt_action[..., 15:16])
            self.gripper_r_mae += F.l1_loss(pred_action[..., 15:16], gt_action[..., 15:16])

            self.total += 1

    def compute(self):
        res = {}
        for name in self.metric_names:
            res[name] = getattr(self, name).float() / self.total
        return res