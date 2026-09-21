# scripts/train_velocity.py
import sys
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from idr.models.velocity_net import VelocityNet, gaussian_nll
from idr.data.dataset import (
    discover_sessions, load_sessions, compute_norm_stats, IDRDataset,
)
from idr.utils.exp import (
    load_config, make_experiment_dir, save_config, set_seed, append_summary,
)

def build_loaders(cfg):
    from torch.utils.data import DataLoader
    pdir = cfg["data"]["processed_dir"]

    # discover_sessions already does the val/test separation based on prefixes.
    tr_s, va_s, te_s = discover_sessions(pdir)

    print(f"Train {len(tr_s)}  Val {len(va_s)}  Test {len(te_s)}")
    print(f"  val  sessions: {va_s}")
    print(f"  test sessions: {te_s}")

    if not va_s:
        raise RuntimeError("No validation sessions discovered. Check prefixes.")
    if not te_s:
        raise RuntimeError("No test sessions discovered. Check prefixes.")

    X_tr, y_tr, _ = load_sessions(pdir, tr_s)
    X_va, y_va, _ = load_sessions(pdir, va_s)
    X_te, y_te, _ = load_sessions(pdir, te_s)

    mean, std = compute_norm_stats(X_tr)
    print(f"Train: {X_tr.shape}  Val: {X_va.shape}  Test: {X_te.shape}")

    tr = IDRDataset(X_tr, y_tr, mean, std)
    va = IDRDataset(X_va, y_va, mean, std)
    te = IDRDataset(X_te, y_te, mean, std)

    bs = cfg["data"]["batch_size"]
    nw = cfg["data"]["num_workers"]
    return {
        "train": DataLoader(tr, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True),
        "val":   DataLoader(va, batch_size=bs, shuffle=False, num_workers=nw),
        "test":  DataLoader(te, batch_size=bs, shuffle=False, num_workers=nw),
        "norm":  (mean, std),
        "sessions": (tr_s, va_s, te_s),
    }


def evaluate(model, loader, device):
    model.eval()
    loss_sum = 0.0
    abs_sum = 0.0
    sq_sum = 0.0
    n = 0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(X)
            loss_sum += gaussian_nll(pred, y).item() * X.size(0)
            err = pred[:, 0] - y
            abs_sum += err.abs().sum().item()
            sq_sum += (err ** 2).sum().item()
            n += X.size(0)
    return {
        "loss": loss_sum / n,
        "mae": abs_sum / n,
        "rmse": (sq_sum / n) ** 0.5,
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
    model = VelocityNet(
        in_channels=m_cfg["in_channels"],
        hidden=m_cfg["hidden"],
        gru_hidden=m_cfg["gru_hidden"],
        dropout=m_cfg["dropout"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    t_cfg = cfg["train"]
    optim = AdamW(model.parameters(), lr=t_cfg["lr"], weight_decay=t_cfg["weight_decay"])

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
            loss = gaussian_nll(pred, y)
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
              f"val_nll={val['loss']:.4f} val_mae={val['mae']:.3f} "
              f"val_rmse={val['rmse']:.3f}  ({row['epoch_time_s']:.1f}s)")

        # save last
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
                print(f"Early stop at epoch {epoch} (patience {patience})")
                break

    # Final test on the best checkpoint
    best_ckpt = torch.load(exp_dir / "model_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    test = evaluate(model, test_loader, device)
    print(f"Best epoch: {best_epoch}")
    print(f"TEST  nll={test['loss']:.4f}  mae={test['mae']:.3f}  rmse={test['rmse']:.3f}")

    # Persist history and metrics
    (exp_dir / "history.json").write_text(json.dumps(history, indent=2))
    (exp_dir / "metrics.json").write_text(json.dumps({
        "best_epoch": best_epoch,
        "best_val": best_ckpt["val_metrics"],
        "test": test,
        "params": n_params,
        "total_time_s": time.time() - t_start,
    }, indent=2))

    # Append to global summary
    append_summary({
        "timestamp": exp_dir.name.split("_")[0] + "_" + exp_dir.name.split("_")[1],
        "name": cfg["name"],
        "epochs": best_epoch,
        "params": n_params,
        "best_val_nll": f"{best_val:.4f}",
        "best_val_mae": f"{best_ckpt['val_metrics']['mae']:.4f}",
        "best_val_rmse": f"{best_ckpt['val_metrics']['rmse']:.4f}",
        "test_nll": f"{test['loss']:.4f}",
        "test_mae": f"{test['mae']:.4f}",
        "test_rmse": f"{test['rmse']:.4f}",
        "train_time_s": f"{time.time() - t_start:.1f}",
        "config_path": str(config_path),
    })

    return model, exp_dir


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/baseline.yaml")
    args = p.parse_args()
    train(args.config)