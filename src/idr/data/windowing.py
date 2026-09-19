# src/idr/data/windowing.py
import numpy as np


def make_windows(imu_data, target_speed, window_len=50, stride=5):
    X, y = [], []
    n = len(imu_data)
    for i in range(0, n - window_len, stride):
        win = imu_data[i:i + window_len]
        tgt = target_speed[i + window_len - 1]
        if not np.isfinite(win).all():
            continue
        if not np.isfinite(tgt):
            continue
        X.append(win)
        y.append(tgt)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)