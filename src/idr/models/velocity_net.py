# src/idr/models/velocity_net.py
"""
Forward velocity estimator for vehicle IDR.

Input:  (B, T, 6)  — 6-axis IMU window (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
Output: (B, 2)    — [v_hat, log_var]  where v_hat >= 0 and log_var in [-6, 4]

Architecture: causal dilated 1D convolutions + GRU + dual head.
Trained with Gaussian NLL loss so the model learns aleatoric uncertainty.
Designed for QAT: only Conv1d, BatchNorm1d, GELU, GRU, Linear — all int8 friendly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D convolution with left-only padding (causal)."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)

    def forward(self, x):
        # x: (B, C, T)
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        self.conv = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(F.gelu(self.bn(self.conv(x))))


class VelocityNet(nn.Module):
    """
    Causal velocity estimator.
    Input shape:  (B, T, 6)
    Output shape: (B, 2) -> [v_hat, log_var]
    """

    def __init__(self, in_channels=6, hidden=64, gru_hidden=64, dropout=0.1):
        super().__init__()

        self.tcn = nn.Sequential(
            TCNBlock(in_channels, 32, kernel_size=5, dilation=1, dropout=dropout),
            TCNBlock(32, hidden, kernel_size=5, dilation=2, dropout=dropout),
            TCNBlock(hidden, hidden, kernel_size=5, dilation=4, dropout=dropout),
            TCNBlock(hidden, hidden, kernel_size=5, dilation=8, dropout=dropout),
        )

        self.gru = nn.GRU(hidden, gru_hidden, num_layers=1, batch_first=True)

        self.speed_head = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (B, T, C) -> permute to (B, C, T) for conv1d
        x = x.transpose(1, 2)
        x = self.tcn(x)                 # (B, hidden, T)
        x = x.transpose(1, 2)           # (B, T, hidden)
        out, _ = self.gru(x)            # (B, T, gru_hidden)
        last = out[:, -1, :]            # (B, gru_hidden)

        v_raw = self.speed_head(last)   # (B, 1)
        v_hat = F.softplus(v_raw)       # >= 0, smooth

        log_var = self.logvar_head(last)
        log_var = torch.clamp(log_var, -6.0, 4.0)
        return torch.cat([v_hat, log_var], dim=-1)   # (B, 2)


def gaussian_nll(pred, target):
    """
    pred:   (B, 2) = [v_hat, log_var]
    target: (B,)   = v_true
    Returns mean Gaussian NLL.
    """
    v_hat = pred[:, 0]
    log_var = pred[:, 1]
    # 0.5 * ( exp(-logvar) * (v_hat - v)^2 + logvar )
    loss = 0.5 * (torch.exp(-log_var) * (v_hat - target) ** 2 + log_var)
    return loss.mean()