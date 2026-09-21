# src/idr/models/velocity_net.py
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
    Causal velocity estimator with mean pooling + last-state GRU head.

    Output: [v_hat, log_var]  (v_hat >= 0, log_var in [-3, 3])
    """

    def __init__(self, in_channels=6, hidden=96, gru_hidden=96, dropout=0.2):
        super().__init__()

        self.tcn = nn.Sequential(
            TCNBlock(in_channels, 48, kernel_size=5, dilation=1, dropout=dropout),
            TCNBlock(48, hidden, kernel_size=5, dilation=2, dropout=dropout),
            TCNBlock(hidden, hidden, kernel_size=5, dilation=4, dropout=dropout),
            TCNBlock(hidden, hidden, kernel_size=5, dilation=8, dropout=dropout),
        )

        self.gru = nn.GRU(hidden, gru_hidden, num_layers=1, batch_first=True)

        # Feature combination: mean-pooled TCN + last GRU state
        feat_dim = hidden + gru_hidden

        self.speed_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (B, T, C)
        x = x.transpose(1, 2)                     # (B, C, T)
        tcn_out = self.tcn(x)                     # (B, hidden, T)
        pooled = tcn_out.mean(dim=-1)             # (B, hidden)
        seq = tcn_out.transpose(1, 2)             # (B, T, hidden)
        gru_out, _ = self.gru(seq)                # (B, T, gru_hidden)
        last = gru_out[:, -1, :]                  # (B, gru_hidden)
        feat = torch.cat([pooled, last], dim=-1)  # (B, hidden + gru_hidden)

        v_raw = self.speed_head(feat)
        v_hat = F.softplus(v_raw)

        log_var = self.logvar_head(feat)
        log_var = torch.clamp(log_var, -3.0, 3.0)
        return torch.cat([v_hat, log_var], dim=-1)


def gaussian_nll(pred, target, mse_only=False, nll_weight=0.05):
    """
    Stable loss:
      - always compute MSE on the mean
      - optionally add a small-weighted NLL term after warmup
    """
    v_hat = pred[:, 0]
    log_var = pred[:, 1]
    mse = ((v_hat - target) ** 2).mean()

    if mse_only:
        return mse

    nll = 0.5 * (torch.exp(-log_var) * (v_hat - target) ** 2 + log_var)
    return mse + nll_weight * nll.mean()