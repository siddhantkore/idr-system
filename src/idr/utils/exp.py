# src/idr/utils/exp.py
import json
import time
import shutil
from pathlib import Path
import yaml
import torch


EXPERIMENTS_ROOT = Path("experiments")
SUMMARY_CSV = Path("experiments.csv")
SUMMARY_HEADER = (
    "timestamp,name,epochs,params,"
    "best_val_nll,best_val_mae,best_val_rmse,"
    "test_nll,test_mae,test_rmse,"
    "train_time_s,config_path\n"
)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def make_experiment_dir(name):
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = EXPERIMENTS_ROOT / f"{stamp}_{name}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_config(cfg, exp_dir):
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_summary(row: dict):
    """Append one row to experiments.csv, creating the header if needed."""
    if not SUMMARY_CSV.exists():
        SUMMARY_CSV.write_text(SUMMARY_HEADER)
    keys = SUMMARY_HEADER.strip().split(",")
    line = ",".join(str(row.get(k, "")) for k in keys) + "\n"
    with open(SUMMARY_CSV, "a") as f:
        f.write(line)