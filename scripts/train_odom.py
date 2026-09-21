# scripts/train_odom.py
import sys
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from idr.models.inertial_odom_net import InertialOdomNet, odom_nll
from idr.data.dataset import discover_sessions
from idr.utils.exp import (
    load_config, make_experiment_dir, save_config, set_seed, append_summary,
)


class IDROdomDataset(Dataset):
    def __init__(self, X, y_odom, mean, std):
        X = (X - mean) / std
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y_odom.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def load_sessions_odom(processed_dir, sessions):
    X_list, y_list = [], []
    processed_dir = Path(processed_dir)
    for name in sessions:
        p = processed_dir / f"{name}.npz"
        if not p.exists():
            print(f"[warn] missing {name}")
            continue
        d = np.load(p, allow_pickle=True)
        if "y_odom" not in d:
            print(f"[warn] {name} has no y_odom, skipping")
            continue
        X_list.append(d["X"])
        y_list.append(d["y_odom"])
    if not X_list:
        raise RuntimeError("No odometry data found")
    return (np.concatenate(X_list, axis=0),
            np.concatenate(y_list, axis=0).astype(np.float32))


def compute_norm_stats(X):
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def build_loaders(cfg):
    pdir = cfg["data"]["processed_dir"]
    tr_s, va_s, te_s = discover_sessions(pdir)
    print(f"Train {len(tr_s)}  Val {len(va_s)}  Test {len(te_s)}")
    print(f"  val  sessions: {va_s}")
    print(f"  test sessions: {te_s}")
    if not va_s:
        raise RuntimeError("No validation sessions")
    if not te_s:
        raise RuntimeError("No test sessions")

    X_tr, y_tr = load_sessions_odom(pdir, tr_s)
    X_va, y_va = load_sessions_odom(pdir, va_s)
    X_te, y_te = load_sessions_odom(pdir, te_s)

    mean, std = compute_norm_stats(X_tr)
    print(f"Train: {X_tr.shape}  Val: {X_va.shape}  Test: {X_te.shape}")
    print(f"y_odom shape: {y_tr.shape}  "
          f"dx std={y_tr[:,0].std():.3f}  dy std={y_tr[:,1].std():.3f}  "
          f"dpsi std={y_tr[:,2].std():.4f}")

    bs = cfg["data"]["batch_size"]
    nw = cfg["data"]["num_workers"]

    tr = IDROdomDataset(X_tr, y_tr, mean, std)
    va = IDROdomDataset(X_va, y_va, mean, std)
    te = IDROdomDataset(X_te, y_te, mean, std)

    return {
        "train": DataLoader(tr, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True),
        "val":   DataLoader(va, batch_size=bs, shuffle=False, num_workers=nw),
        "test":  DataLoader(te, batch_size=bs, shuffle=False, num_workers=nw),
        "norm":  (mean, std),
    }


def evaluate(model, loader, device):
    model.eval()
    loss_sum = 0.0
    n = 0
    dx_abs = dy_abs = dpsi_abs = 0.0
    dx_sq = dy_sq = dpsi_sq = 0.0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(X)
            loss_sum += odom_nll(pred, y).item() * X.size(0)

            dx_err = pred[:, 0] - y[:, 0]
            dy_err = pred[:, 1] - y[:, 1]
            dpsi_err = pred[:, 2] - y[:, 2]

            dx_abs += dx_err.abs().sum().item()
            dy_abs += dy_err.abs().sum().item()
            dpsi_abs += dpsi_err.abs().sum().item()

            dx_sq += (dx_err ** 2).sum().item()
            dy_sq += (dy_err ** 2).sum().item()
            dpsi_sq += (dpsi_err ** 2).sum().item()
            n += X.size(0)

    return {
        "loss": loss_sum / n,
        "dx_mae": dx_abs / n,
        "dy_mae": dy_abs / n,
        "dpsi_mae": dpsi_abs / n,
        "dx_rmse": (dx_sq / n) ** 0.5,
        "dy_rmse": (dy_sq / n) ** 0.5,
        "dpsi_rmse": (dpsi_sq / n) ** 0.5,
    }


def train(config_path):
    cfg = load_config(config_path)
    exp_dir = make_experiment_dir(cfg["name"])
    save_config(cfg, exp_dir)
    print(f"Experiment dir: {exp_dir}")

    set_seed(cfg.get("seed", 42))

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    loaders = build_loaders(cfg)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]
    mean, std = loaders["norm"]

    m_cfg = cfg["model"]
    model = InertialOdomNet(
        in_channels=m_cfg["in_channels"],
        hidden=m_cfg["hidden"],
        gru_hidden=m_cfg["gru_hidden"],
        dropout=m_cfg["dropout"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    t_cfg = cfg["train"]
    optim = AdamW(model.parameters(), lr=t_cfg["lr"],
                  weight_decay=t_cfg["weight_decay"])

    sched_name = t_cfg.get("scheduler", "cosine")
    if sched_name == "cosine":
        sched = CosineAnnealingLR(optim, T_max=t_cfg["epochs"])
    elif sched_name == "plateau":
        sched = ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=3)
    else:
        sched = None

    best_val = float("inf")
    best_epoch = 0
    patience = t_cfg.get("early_stop_patience", 0)
    bad_epochs = 0
    history = []
    t_start = time.time()

    for epoch in range(1, t_cfg["epochs"] + 1):
        model.train()
        ep_start = time.time()
        loss_sum, n = 0.0, 0
        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            pred = model(X)
            loss = odom_nll(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg["grad_clip"])
            optim.step()
            loss_sum += loss.item() * X.size(0)
            n += X.size(0)
        train_loss = loss_sum / n

        val = evaluate(model, val_loader, device)
        if sched is not None:
            if sched_name == "plateau":
                sched.step(val["loss"])
            else:
                sched.step()

        row = {"epoch": epoch, "train_nll": train_loss, **val,
               "lr": optim.param_groups[0]["lr"],
               "epoch_time_s": time.time() - ep_start}
        history.append(row)
        print(f"[{epoch:02d}] train_nll={train_loss:.4f} "
              f"val_nll={val['loss']:.4f} "
              f"dx_mae={val['dx_mae']:.3f}m dy_mae={val['dy_mae']:.3f}m "
              f"dpsi_mae={val['dpsi_mae']:.4f}rad  ({row['epoch_time_s']:.1f}s)")

        torch.save({
            "model_state": model.state_dict(),
            "norm_mean": mean, "norm_std": std,
            "config": cfg, "epoch": epoch,
        }, exp_dir / "model_last.pt")

        if val["loss"] < best_val:
            best_val = val["loss"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save({
                "model_state": model.state_dict(),
                "norm_mean": mean, "norm_std": std,
                "config": cfg, "epoch": epoch,
                "val_metrics": val,
            }, exp_dir / "model_best.pt")
        else:
            bad_epochs += 1
            if patience and bad_epochs >= patience:
                print(f"Early stop at epoch {epoch}")
                break

    best_ckpt = torch.load(exp_dir / "model_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    test = evaluate(model, test_loader, device)
    print(f"\nBest epoch: {best_epoch}")
    print(f"TEST nll={test['loss']:.4f}")
    print(f"     dx_mae={test['dx_mae']:.3f}m  dy_mae={test['dy_mae']:.3f}m  "
          f"dpsi_mae={test['dpsi_mae']:.4f}rad")
    print(f"     dx_rmse={test['dx_rmse']:.3f}m dy_rmse={test['dy_rmse']:.3f}m "
          f"dpsi_rmse={test['dpsi_rmse']:.4f}rad")

    (exp_dir / "history.json").write_text(json.dumps(history, indent=2))
    (exp_dir / "metrics.json").write_text(json.dumps({
        "best_epoch": best_epoch,
        "best_val": best_ckpt["val_metrics"],
        "test": test,
        "params": n_params,
        "total_time_s": time.time() - t_start,
    }, indent=2))

    append_summary({
        "timestamp": exp_dir.name,
        "name": cfg["name"],
        "epochs": best_epoch,
        "params": n_params,
        "best_val_nll": f"{best_val:.4f}",
        "best_val_mae": f"{best_ckpt['val_metrics']['dx_mae']:.4f}",
        "best_val_rmse": f"{best_ckpt['val_metrics']['dx_rmse']:.4f}",
        "test_nll": f"{test['loss']:.4f}",
        "test_mae": f"{test['dx_mae']:.4f}",
        "test_rmse": f"{test['dx_rmse']:.4f}",
        "train_time_s": f"{time.time() - t_start:.1f}",
        "config_path": str(config_path),
    })

    return model, exp_dir


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/odom_baseline.yaml")
    args = p.parse_args()
    train(args.config)