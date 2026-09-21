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

R_EARTH = 6378137.0


# ---------------------------------------------------------------- discovery

def _clean_session_name(folder_name):
    """Strip driver suffix and leading V- prefix from folder name."""
    name = folder_name.split(" (")[0]     # 'M (Driver B)' -> 'M'
    if name.startswith("V-"):
        name = name[2:]                    # 'V-Vfa01' -> 'Vfa01'
    return name


def find_pairs(raw_dir):
    """Return list of (s_path, v_path, session_name) for S/V pairs in the same folder."""
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
    """Resample an S session onto a fixed 10 Hz grid using its own time column."""
    t = s_df["time_s"].to_numpy(dtype=float)
    if t.size < 2:
        raise ValueError("S session too short to resample")

    order = np.argsort(t)
    t = t[order]
    s_df = s_df.iloc[order].reset_index(drop=True)

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


# ---------------------------------------------------------------- speed target

def build_speed_target(s_resampled, v_df):
    """Interpolate V speed onto the resampled S time grid (10 Hz)."""
    t_s = s_resampled["time_s"].to_numpy(dtype=float)
    t_v = v_df["time_s_aligned"].to_numpy(dtype=float)
    v_speed = v_df["velocity_ms"].to_numpy(dtype=float)

    order = np.argsort(t_v)
    t_v = t_v[order]
    v_speed = v_speed[order]

    target = np.interp(t_s, t_v, v_speed, left=np.nan, right=np.nan)
    return target


def build_imu_array(s_resampled):
    cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    missing = [c for c in cols if c not in s_resampled.columns]
    if missing:
        raise KeyError(f"Missing IMU columns after resample: {missing}")
    imu = s_resampled[cols].to_numpy(dtype=np.float32)
    mask = np.isfinite(imu).all(axis=1)
    return imu, mask


# ---------------------------------------------------------------- odom targets

def latlon_to_enu(lat, lon, lat0, lon0):
    """Equirectangular projection to local ENU metres."""
    dlat = np.deg2rad(lat - lat0)
    dlon = np.deg2rad(lon - lon0)
    e = R_EARTH * dlon * np.cos(np.deg2rad(lat0))
    n = R_EARTH * dlat
    return e, n


def compute_odom_targets(v_df, t_grid, window_len, stride):
    """
    Build odometry targets per window.
    Returns array shape (M, 3): [dx_body, dy_body, dpsi]
    where:
        dx_body: forward displacement in vehicle body frame (m)
        dy_body: lateral displacement in body frame     (m)
        dpsi:    yaw change over the window              (rad, wrapped to [-pi, pi])
    """
    t_v = v_df["time_s_aligned"].to_numpy(dtype=float)
    order = np.argsort(t_v)
    t_v = t_v[order]

    lat = v_df["lat"].to_numpy(dtype=float)[order]
    lon = v_df["lon"].to_numpy(dtype=float)[order]
    heading_rad = np.deg2rad(v_df["heading"].to_numpy(dtype=float)[order])
    heading_unwrapped = np.unwrap(heading_rad)

    lat_i = np.interp(t_grid, t_v, lat, left=np.nan, right=np.nan)
    lon_i = np.interp(t_grid, t_v, lon, left=np.nan, right=np.nan)
    psi_i = np.interp(t_grid, t_v, heading_unwrapped, left=np.nan, right=np.nan)

    lat0 = np.nanmean(lat_i)
    lon0 = np.nanmean(lon_i)
    e, n = latlon_to_enu(lat_i, lon_i, lat0, lon0)

    N = len(t_grid)
    out = []
    for i in range(window_len - 1, N, stride):
        j = i - window_len + 1
        if not (np.isfinite(e[j]) and np.isfinite(e[i]) and np.isfinite(psi_i[j])):
            continue
        de = e[i] - e[j]
        dn = n[i] - n[j]
        psi = psi_i[j]
        c, s = np.cos(psi), np.sin(psi)
        dx = c * de + s * dn          # forward
        dy = -s * de + c * dn         # left
        dpsi = np.arctan2(np.sin(psi_i[i] - psi_i[j]),
                          np.cos(psi_i[i] - psi_i[j]))
        out.append([dx, dy, dpsi])
    return np.asarray(out, dtype=np.float32)


# ---------------------------------------------------------------- per pair

def process_pair(s_path, v_path, session_name):
    s_df_raw = load_s_file(s_path)
    v_df_raw = load_v_file(v_path)

    if "gps_speed_ms" not in s_df_raw.columns:
        raise KeyError(f"S file {s_path} missing GPS speed")
    if "velocity_ms" not in v_df_raw.columns:
        raise KeyError(f"V file {v_path} missing velocity")
    for c in ["lat", "lon", "heading"]:
        if c not in v_df_raw.columns:
            raise KeyError(f"V file {v_path} missing {c}")

    # 1. Align on original clocks
    s_df, v_df, lag_s, conf = align_sv(s_df_raw, v_df_raw,
                                       target_dt=TARGET_DT, max_lag_s=3.0)

    # 2. Resample S to 10 Hz
    s_rs = resample_s_to_grid(s_df, grid_dt=TARGET_DT)

    # 3. IMU + mask
    imu, imu_mask = build_imu_array(s_rs)

    # 4. Speed target
    target = build_speed_target(s_rs, v_df)

    # 5. Combined validity (used for both windowing and odom targets)
    valid = imu_mask & np.isfinite(target)
    if valid.sum() < WINDOW_LEN + 1:
        raise ValueError(f"Too few valid samples ({int(valid.sum())})")

    imu_v = imu[valid]
    target_v = target[valid]
    t_grid_v = s_rs["time_s"].to_numpy(dtype=float)[valid]

    tstd = float(np.nanstd(target_v))
    is_stationary = tstd < 0.1

    # 6. Windowing for IMU + speed
    X, y = make_windows(imu_v, target_v, window_len=WINDOW_LEN, stride=STRIDE)
    if len(y) == 0:
        raise ValueError("No windows produced")

    # 7. Odometry targets on the SAME 10 Hz grid, using the SAME stride
    y_odom = compute_odom_targets(v_df, t_grid_v, WINDOW_LEN, STRIDE)

    # 8. Align lengths (guard against off-by-one from NaN edges)
    m = min(len(X), len(y_odom))
    if m == 0:
        raise ValueError("No overlapping windows for odom targets")
    X = X[:m]
    y = y[:m]
    y_odom = y_odom[:m]

    # 9. Filter windows with non-finite odom targets
    finite_mask = np.isfinite(y_odom).all(axis=1)
    if not finite_mask.all():
        X = X[finite_mask]
        y = y[finite_mask]
        y_odom = y_odom[finite_mask]
    if len(y) == 0:
        raise ValueError("All odometry windows were non-finite")

    out_path = OUT_DIR / f"{session_name}.npz"
    np.savez_compressed(
        out_path,
        X=X.astype(np.float32),
        y=y.astype(np.float32),
        y_odom=y_odom.astype(np.float32),
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
        print(f"|lag| == 0 : {(lags == 0.0).sum()}")
        print(f"|lag|  > 3 : {(np.abs(lags) > 3).sum()}")

        stat = [s for s in successes if s[4] < 0.1]
        print(f"Stationary sessions: {len(stat)}")
        for s in stat:
            print(f"   {s[0]:10s}  std={s[4]:.4f}")

    if failures:
        print("\nFailures:")
        for name, err in failures:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()