# scripts/prepare_data.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from tqdm import tqdm

from idr.io.loader import load_s_file, load_v_file
from idr.io.align import align_sv
from idr.data.windowing import make_windows


RAW_DIR = Path("data/raw/Synchronised V abd S datasets/Categorised IOVNB Dataset")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DT = 0.1      # 10 Hz
WINDOW_LEN = 50      # 5 s at 10 Hz
STRIDE = 5


# ---------------------------------------------------------------- discovery

def _clean_session_name(folder_name):
    """Strip driver suffix and leading V- prefix from folder name."""
    name = folder_name.split(" (")[0]     # 'M (Driver B)' -> 'M'
    if name.startswith("V-"):
        name = name[2:]                    # 'V-Vfa01' -> 'Vfa01'
    return name


def find_pairs(raw_dir):
    pairs = []
    for session_dir in raw_dir.rglob("*"):
        if not session_dir.is_dir():
            continue
        s_files = list(session_dir.glob("S-*.csv"))
        v_files = list(session_dir.glob("V-*.csv"))
        if s_files and v_files:
            pairs.append((s_files[0], v_files[0], _clean_session_name(session_dir.name)))
    return pairs


# ---------------------------------------------------------------- resampling

def resample_s_to_grid(s_df, grid_dt=TARGET_DT):
    """
    Resample an S session onto a fixed 10 Hz grid using its own time column.
    Returns a new DataFrame with 'time_s' on the grid plus resampled IMU and aux columns.
    """
    t = s_df["time_s"].to_numpy(dtype=float)
    if t.size < 2:
        raise ValueError("S session too short to resample")

    # Ensure strictly increasing time
    order = np.argsort(t)
    t = t[order]
    s_df = s_df.iloc[order].reset_index(drop=True)

    # Remove duplicate timestamps (keep first)
    keep = np.concatenate([[True], np.diff(t) > 0])
    t = t[keep]
    s_df = s_df.iloc[keep].reset_index(drop=True)

    t0, t1 = t[0], t[-1]
    grid = np.arange(t0, t1, grid_dt)

    out = {"time_s": grid}
    for col in [
        "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z",
        "gps_speed_ms",
        "ori_yaw_rad", "ori_pitch_rad", "ori_roll_rad",
    ]:
        if col in s_df.columns:
            out[col] = np.interp(
                grid, t, s_df[col].to_numpy(dtype=float),
                left=np.nan, right=np.nan,
            )
    return pd.DataFrame(out)


# ---------------------------------------------------------------- alignment helpers

def build_speed_target(s_resampled, v_df):
    """
    Interpolate V speed onto the resampled S time grid (10 Hz).
    v_df must have 'time_s_aligned' from align_sv.
    """
    t_s = s_resampled["time_s"].to_numpy(dtype=float)
    t_v = v_df["time_s_aligned"].to_numpy(dtype=float)
    v_speed = v_df["velocity_ms"].to_numpy(dtype=float)

    order = np.argsort(t_v)
    t_v = t_v[order]
    v_speed = v_speed[order]

    target = np.interp(t_s, t_v, v_speed, left=np.nan, right=np.nan)
    return target


def build_imu_array(s_resampled):
    """Return (imu_array, valid_mask) on the resampled S grid."""
    cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    missing = [c for c in cols if c not in s_resampled.columns]
    if missing:
        raise KeyError(f"Missing IMU columns after resample: {missing}")
    imu = s_resampled[cols].to_numpy(dtype=np.float32)
    mask = np.isfinite(imu).all(axis=1)
    return imu, mask


# ---------------------------------------------------------------- per-pair

def process_pair(s_path, v_path, session_name):
    s_df_raw = load_s_file(s_path)
    v_df_raw = load_v_file(v_path)

    if "gps_speed_ms" not in s_df_raw.columns:
        raise KeyError(f"S file {s_path} missing GPS speed")
    if "velocity_ms" not in v_df_raw.columns:
        raise KeyError(f"V file {v_path} missing velocity")

    # 1. Align on original clocks
    s_df, v_df, lag_s, conf = align_sv(s_df_raw, v_df_raw, target_dt=TARGET_DT, max_lag_s=3.0)

    # 2. Resample S to 10 Hz grid
    s_rs = resample_s_to_grid(s_df, grid_dt=TARGET_DT)

    # 3. Build IMU on the resampled grid
    imu, imu_mask = build_imu_array(s_rs)

    # 4. Build speed target on the same grid
    target = build_speed_target(s_rs, v_df)

    # 5. Combine validity
    valid = imu_mask & np.isfinite(target)
    if valid.sum() < WINDOW_LEN + 1:
        raise ValueError(f"Too few valid samples after masking ({int(valid.sum())})")

    imu = imu[valid]
    target = target[valid]

    # 6. Compute stats BEFORE windowing
    tstd = float(np.nanstd(target))
    is_stationary = tstd < 0.1

    # 7. Window
    X, y = make_windows(imu, target, window_len=WINDOW_LEN, stride=STRIDE)
    if len(y) == 0:
        raise ValueError("No windows produced")

    out_path = OUT_DIR / f"{session_name}.npz"
    np.savez_compressed(
        out_path,
        X=X.astype(np.float32),
        y=y.astype(np.float32),
        session=session_name,
        lag_s=float(lag_s),
        lag_conf=float(conf),
        target_std=tstd,
        is_stationary=is_stationary,
        n_samples=int(valid.sum()),
    )
    return len(y), lag_s, conf, tstd


# ---------------------------------------------------------------- main

def main():
    pairs = find_pairs(RAW_DIR)
    print(f"Found {len(pairs)} S/V pairs")

    successes = []
    failures = []

    for s_path, v_path, session in tqdm(pairs):
        try:
            n, lag, conf, tstd = process_pair(s_path, v_path, session)
            successes.append((session, n, lag, conf, tstd))
        except Exception as e:
            failures.append((session, str(e)))

    print(f"\nSucceeded: {len(successes)}")
    print(f"Failed:    {len(failures)}")

    if successes:
        lags = np.array([s[2] for s in successes])
        confs = np.array([s[3] for s in successes])
        print(f"Lag   (s): min={lags.min():+.2f} max={lags.max():+.2f} "
              f"median={np.median(lags):+.2f} mean={lags.mean():+.2f}")
        print(f"LagConf   : min={confs.min():.2f} max={confs.max():.2f} "
              f"median={np.median(confs):.2f}")
        print(f"|lag| == 0 : {(lags == 0.0).sum()}  (alignment fell back to zero)")
        print(f"|lag|  > 3 : {(np.abs(lags) > 3).sum()}  (should be 0)")

        stationaries = [s for s in successes if s[4] < 0.1]
        print(f"Stationary sessions (target std < 0.1 m/s): {len(stationaries)}")
        for s in stationaries:
            print(f"   {s[0]:10s}  std={s[4]:.4f}")

    if failures:
        print("\nFailures:")
        for name, err in failures:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()