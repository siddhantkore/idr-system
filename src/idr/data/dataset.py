# src/idr/data/dataset.py
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


# Sessions whose stems start with these prefixes go to val / test.
# Everything else is used for training.
VAL_PREFIXES  = ["Vfa"]    # motorway, high-speed — good validation of generalization
TEST_PREFIXES = ["Y"]      # Driver D — held out entirely


def discover_sessions(processed_dir):
    """
    Return (train_sessions, val_sessions, test_sessions) as lists of stems
    discovered from data/processed/*.npz.
    """
    processed_dir = Path(processed_dir)
    all_stems = sorted(p.stem for p in processed_dir.glob("*.npz"))

    val, test, train = [], [], []
    for stem in all_stems:
        if any(stem.startswith(p) for p in TEST_PREFIXES):
            test.append(stem)
        elif any(stem.startswith(p) for p in VAL_PREFIXES):
            val.append(stem)
        else:
            train.append(stem)
    return train, val, test


def load_sessions(processed_dir, sessions):
    X_list, y_list, sess_list = [], [], []
    processed_dir = Path(processed_dir)
    for name in sessions:
        p = processed_dir / f"{name}.npz"
        if not p.exists():
            print(f"[warn] missing session {name}")
            continue
        d = np.load(p, allow_pickle=True)
        X_list.append(d["X"])
        y_list.append(d["y"])
        sess_list.append(np.full(len(d["y"]), name))
    if not X_list:
        raise RuntimeError(f"No sessions loaded from {processed_dir}")
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0).astype(np.float32)
    sess = np.concatenate(sess_list, axis=0)
    return X, y, sess


def compute_norm_stats(X):
    """Per-channel mean/std across the training set. X shape: (N, T, C)."""
    mean = X.mean(axis=(0, 1), keepdims=True)   # (1, 1, C)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


class IDRDataset(Dataset):
    def __init__(self, X, y, mean, std):
        X = (X - mean) / std
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def build_loaders(processed_dir="data/processed", batch_size=256, num_workers=2):
    train_sessions, val_sessions, test_sessions = discover_sessions(processed_dir)
    print(f"Discovered: {len(train_sessions)} train, "
          f"{len(val_sessions)} val, {len(test_sessions)} test")
    print(f"  val  sessions: {val_sessions}")
    print(f"  test sessions: {test_sessions}")

    X_tr, y_tr, _ = load_sessions(processed_dir, train_sessions)
    X_va, y_va, _ = load_sessions(processed_dir, val_sessions)
    X_te, y_te, _ = load_sessions(processed_dir, test_sessions)

    mean, std = compute_norm_stats(X_tr)
    print(f"Train: {X_tr.shape}  Val: {X_va.shape}  Test: {X_te.shape}")
    print(f"Norm mean: {mean.reshape(-1)}")
    print(f"Norm std:  {std.reshape(-1)}")

    tr = IDRDataset(X_tr, y_tr, mean, std)
    va = IDRDataset(X_va, y_va, mean, std)
    te = IDRDataset(X_te, y_te, mean, std)

    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, drop_last=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers)

    return {
        "train": tr_loader, "val": va_loader, "test": te_loader,
        "norm": (mean, std),
        "sessions": (train_sessions, val_sessions, test_sessions),
    }

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
    X_list, y_list, sess_list = [], [], []
    for name in sessions:
        p = Path(processed_dir) / f"{name}.npz"
        d = np.load(p, allow_pickle=True)
        if "y_odom" not in d:
            print(f"[warn] {name} has no y_odom, skipping")
            continue
        X_list.append(d["X"])
        y_list.append(d["y_odom"])
        sess_list.append(np.full(len(d["y_odom"]), name))
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0).astype(np.float32)
    sess = np.concatenate(sess_list, axis=0)
    return X, y, sess