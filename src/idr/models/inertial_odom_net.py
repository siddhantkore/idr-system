# src/idr/models/inertial_odom_net.py
"""
Inertial Odometry Net for vehicle IDR.

Input:  (B, T, 6)  — 6-axis IMU window (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
Output: (B, 5)    — [dx_body, dy_body, dpsi, log_var_x, log_var_y]

Notes:
- dx_body: forward displacement in body frame over the window (metres)
- dy_body: lateral displacement in body frame over the window (should be ~0 for a car, non-zero for turns)
- dpsi:    yaw change over the window (radians), can be positive or negative
- log_var_x, log_var_y: aleatoric log-variances for dx and dy

Architecture: ResNet1D-style causal blocks + GRU + dual output heads.
QAT-friendly: only Conv1d, BatchNorm1d, GELU, GRU, Linear.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class ResBlock(nn.Module):
    """Causal residual block with a bottleneck."""

    def __init__(self, channels, kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(F.gelu(self.bn1(self.conv1(x))))
        h = self.bn2(self.conv2(h))
        return F.gelu(x + h)


class InertialOdomNet(nn.Module):
    """
    Causal inertial odometry estimator.
    Input shape:  (B, T, 6)
    Output shape: (B, 5) -> [dx, dy, dpsi, log_var_x, log_var_y]
    """

    def __init__(self, in_channels=6, hidden=64, gru_hidden=64, dropout=0.1):
        super().__init__()

        self.stem = nn.Sequential(
            CausalConv1d(in_channels, hidden, kernel_size=5, dilation=1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )

        self.res_blocks = nn.Sequential(
            ResBlock(hidden, kernel_size=5, dilation=1, dropout=dropout),
            ResBlock(hidden, kernel_size=5, dilation=2, dropout=dropout),
            ResBlock(hidden, kernel_size=5, dilation=4, dropout=dropout),
            ResBlock(hidden, kernel_size=5, dilation=8, dropout=dropout),
        )

        self.gru = nn.GRU(hidden, gru_hidden, num_layers=1, batch_first=True)

        # Displacement head: 2D body-frame displacement
        self.disp_head = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )
        # Yaw-change head: scalar
        self.dpsi_head = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        # Uncertainty head: log-variances for dx and dy
        self.logvar_head = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T) for Conv1d
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.res_blocks(x)
        x = x.transpose(1, 2)              # (B, T, hidden)

        out, _ = self.gru(x)
        last = out[:, -1, :]               # (B, gru_hidden)

        disp = self.disp_head(last)        # (B, 2) dx, dy
        dpsi = self.dpsi_head(last)        # (B, 1)
        log_var = self.logvar_head(last)   # (B, 2)
        log_var = torch.clamp(log_var, -6.0, 4.0)

        return torch.cat([disp, dpsi, log_var], dim=-1)   # (B, 5)


def odom_nll(pred, target):
    """
    pred:   (B, 5) = [dx, dy, dpsi, log_var_x, log_var_y]
    target: (B, 3) = [dx_true, dy_true, dpsi_true]

    Loss = Gaussian NLL on (dx, dy) + MSE on dpsi.
    The heading change dpsi is trained with plain MSE because its scale is
    small and we don't want the model to game the variance.
    """
    dx_p, dy_p, dpsi_p = pred[:, 0], pred[:, 1], pred[:, 2]
    lvx, lvy = pred[:, 3], pred[:, 4]
    dx_t, dy_t, dpsi_t = target[:, 0], target[:, 1], target[:, 2]

    # Gaussian NLL for dx, dy
    nll_x = 0.5 * (torch.exp(-lvx) * (dx_p - dx_t) ** 2 + lvx)
    nll_y = 0.5 * (torch.exp(-lvy) * (dy_p - dy_t) ** 2 + lvy)

    # MSE for dpsi
    mse_psi = (dpsi_p - dpsi_t) ** 2

    return (nll_x + nll_y).mean() + 5.0 * mse_psi.mean()