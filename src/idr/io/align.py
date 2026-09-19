# src/idr/io/align.py
import numpy as np
from scipy.signal import correlate


def _resample(t_src, x_src, t_dst):
    """Linear interpolation with NaN for out-of-range."""
    return np.interp(t_dst, t_src, x_src, left=np.nan, right=np.nan)


def _best_lag(sig_s, sig_v, dt, max_lag_s, min_peak_ratio=1.15):
    """
    Return (lag_s, confidence) where confidence is peak/2nd_peak ratio
    on |correlation|. If confidence < min_peak_ratio, return (0.0, confidence).
    """
    s = np.nan_to_num(sig_s - np.nanmean(sig_s))
    v = np.nan_to_num(sig_v - np.nanmean(sig_v))
    if np.std(s) < 1e-6 or np.std(v) < 1e-6:
        return 0.0, 0.0

    corr = correlate(s, v, mode="full")
    lags = np.arange(-len(v) + 1, len(s)) * dt
    valid = np.abs(lags) <= max_lag_s
    corr_v = np.abs(corr[valid])
    lags_v = lags[valid]
    if corr_v.size == 0:
        return 0.0, 0.0

    idx = np.argmax(corr_v)
    peak = corr_v[idx]
    # Null out a small window around the peak to find the second-highest
    half = max(1, int(0.5 / dt))
    lo = max(0, idx - half)
    hi = min(len(corr_v), idx + half + 1)
    masked = corr_v.copy()
    masked[lo:hi] = 0.0
    second = masked.max() if masked.size else 0.0
    ratio = peak / second if second > 1e-9 else 99.0

    if ratio < min_peak_ratio:
        return 0.0, ratio
    return float(lags_v[idx]), ratio


def align_sv(s_df, v_df, target_dt=0.1, max_lag_s=3.0):
    """
    Robust S/V alignment.
    - Uses both GPS-speed correlation and accelerometer-magnitude correlation.
    - Picks the lag with the higher confidence.
    - Falls back to zero lag if neither is confident.
    Returns (s_df, v_df, lag_s, confidence).
    """
    # Normalize time axes
    t_s_raw = s_df["time_s"].to_numpy(dtype=float)
    t_v_raw = v_df["time_s"].to_numpy(dtype=float)
    t_s_raw = t_s_raw - t_s_raw[0]
    t_v_raw = t_v_raw - t_v_raw[0]

    t_end = min(t_s_raw[-1], t_v_raw[-1])
    if t_end <= 2.0:
        raise ValueError("S/V recordings too short or non-overlapping")

    grid = np.arange(0.0, t_end, target_dt)

    # --- Signal 1: GPS speed vs vehicle speed ---
    s_gps_speed = _resample(t_s_raw, s_df["gps_speed_ms"].to_numpy(dtype=float), grid)
    v_speed = _resample(t_v_raw, v_df["velocity_ms"].to_numpy(dtype=float), grid)
    lag1, conf1 = _best_lag(np.diff(s_gps_speed), np.diff(v_speed),
                            target_dt, max_lag_s)

    # --- Signal 2: horizontal accel magnitude vs vehicle accel ---
    acc_x = s_df["acc_x"].to_numpy(dtype=float)
    acc_y = s_df["acc_y"].to_numpy(dtype=float)
    acc_mag = np.sqrt(acc_x**2 + acc_y**2)
    a_h = _resample(t_s_raw, acc_mag, grid)
    v_acc = np.gradient(v_speed, target_dt)
    lag2, conf2 = _best_lag(a_h - np.nanmean(a_h), v_acc,
                            target_dt, max_lag_s)

    # In align.py, inside align_sv, change threshold:
    if conf2 >= conf1 and conf2 >= 1.5:
        lag_s, conf = lag2, conf2
    elif conf1 >= 1.5:
        lag_s, conf = lag1, conf1
    else:
        lag_s, conf = 0.0, max(conf1, conf2)
        

    # Apply lag to V's clock
    v_df = v_df.copy()
    v_df["time_s_aligned"] = (
        v_df["time_s"].to_numpy(dtype=float) - v_df["time_s"].iloc[0] - lag_s
    )
    s_df = s_df.copy()
    s_df["time_s"] = t_s_raw
    return s_df, v_df, float(lag_s), float(conf)