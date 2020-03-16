import torch.nn as nn
import torch

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

def compute_loss_old(poses3d_preds, poses3d_annos, loss_type, device):
    if loss_type == 'MSE':
        criterion = nn.MSELoss(reduction='mean').to(device)
    elif loss_type == 'SmoothL1Loss':
        criterion = nn.SmoothL1Loss(reduction='mean').to(device)
    elif loss_type == 'MAE':
        criterion = nn.L1Loss(reduction='mean').to(device)
    elif loss_type == 'RMSE':
        criterion = RMSELoss().to(device)
    else:
        assert False, "Unknown loss type"

    loss = criterion(poses3d_preds, poses3d_annos)

    return loss

def compute_loss(poses3d_preds, poses3d_annos, b_size):
    pose3d_diff = (poses3d_preds - poses3d_annos) ** 2
    pose3d_diff = pose3d_diff.view(b_size, -1, 3)
    pose3d_dis = torch.sum(pose3d_diff, dim=-1)
    pose3d_dis = torch.mean(torch.sqrt(pose3d_dis))

    return pose3d_dis
