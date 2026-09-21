# scripts/prepare_odom_targets.py
"""
Augment each processed session npz with odometry targets.

Targets per window (5 s at 10 Hz):
  dx_body: forward displacement in vehicle body frame  (m)
  dy_body: lateral displacement in vehicle body frame  (m)
  dpsi:    yaw change over the window                   (rad)

Computed from the V (vehicle) GNSS positions and heading on the same 10 Hz grid.
Body frame: x = forward (along heading), y = left (perpendicular).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from tqdm import tqdm

from idr.io.loader import load_v_file
from idr.io.align import align_sv  # reuse to get V time alignment
from idr.io.loader import load_s_file


RAW_DIR = Path("data/raw/Synchronised V abd S datasets/Categorised IOVNB Dataset")
PROC_DIR = Path("data/processed")

WINDOW_LEN = 50
STRIDE = 5

# Earth radius for local ENU conversion
R_EARTH = 6378137.0


def latlon_to_enu(lat, lon, lat0, lon0):
    """Simple equirectangular projection to local ENU metres."""
    dlat = np.deg2rad(lat - lat0)
    dlon = np.deg2rad(lon - lon0)
    e = R_EARTH * dlon * np.cos(np.deg2rad(lat0))
    n = R_EARTH * dlat
    return e, n


def compute_odom_targets(v_df, t_grid):
    """
    v_df: V DataFrame with columns: time_s_aligned, lat, lon, heading
    t_grid: 10 Hz time grid (numpy array)

    Returns: (dx, dy, dpsi) arrays aligned to t_grid.
    """
    t_v = v_df["time_s_aligned"].to_numpy(dtype=float)
    order = np.argsort(t_v)
    t_v = t_v[order]

    lat = v_df["lat"].to_numpy(dtype=float)[order]
    lon = v_df["lon"].to_numpy(dtype=float)[order]
    heading_deg = v_df["heading"].to_numpy(dtype=float)[order]

    # Interpolate onto S 10 Hz grid
    lat_i = np.interp(t_grid, t_v, lat, left=np.nan, right=np.nan)
    lon_i = np.interp(t_grid, t_v, lon, left=np.nan, right=np.nan)

    # Heading is circular: unwrap before interpolation
    heading_rad = np.deg2rad(heading_deg)
    heading_unwrapped = np.unwrap(heading_rad)
    heading_i = np.interp(t_grid, t_v, heading_unwrapped, left=np.nan, right=np.nan)

    # ENU coordinates
    lat0, lon0 = np.nanmean(lat_i), np.nanmean(lon_i)
    e, n = latlon_to_enu(lat_i, lon_i, lat0, lon0)

    # For each 10 Hz sample compute per-window displacement
    # target for window ending at index i is displacement from i-WINDOW_LEN+1 to i
    N = len(t_grid)
    dx_arr = np.full(N, np.nan, dtype=np.float32)
    dy_arr = np.full(N, np.nan, dtype=np.float32)
    dpsi_arr = np.full(N, np.nan, dtype=np.float32)

    for i in range(WINDOW_LEN - 1, N):
        j = i - WINDOW_LEN + 1
        if not (np.isfinite(e[j]) and np.isfinite(e[i])):
            continue
        de = e[i] - e[j]
        dn = n[i] - n[j]
        # Heading at start of window (vehicle frame orientation)
        psi = heading_i[j]
        # Rotate ENU displacement into body frame
        # body x = forward (heading direction), y = left
        c, s = np.cos(psi), np.sin(psi)
        dx = c * de + s * dn   # forward
        dy = -s * de + c * dn  # left
        dpsi = heading_i[i] - heading_i[j]
        # Wrap to [-pi, pi]
        dpsi = np.arctan2(np.sin(dpsi), np.cos(dpsi))
        dx_arr[i] = dx
        dy_arr[i] = dy
        dpsi_arr[i] = dpsi

    return dx_arr, dy_arr, dpsi_arr


def process_one(session_name):
    npz_path = PROC_DIR / f"{session_name}.npz"
    if not npz_path.exists():
        print(f"[skip] {session_name}")
        return

    d = np.load(npz_path, allow_pickle=True)
    X = d["X"]        # (N, 50, 6) — already windowed
    y_speed = d["y"]  # (N,)     — speed targets

    # Find the S and V CSVs for this session
    s_files = list(RAW_DIR.rglob(f"S-{session_name}*.csv")) + \
              list(RAW_DIR.rglob(f"S-{session_name}*.csv".replace("*", "")))
    v_files = list(RAW_DIR.rglob(f"V-{session_name}*.csv")) + \
              list(RAW_DIR.rglob(f"V-{session_name}*.csv".replace("*", "")))
    # fallback: exact folder scan
    s_files = list(RAW_DIR.rglob(f"S-{session_name}.csv"))
    v_files = list(RAW_DIR.rglob(f"V-{session_name}.csv"))
    if not s_files or not v_files:
        # try case variants
        s_files = [p for p in RAW_DIR.rglob("S-*.csv") if p.stem[2:].lower() == session_name.lower()]
        v_files = [p for p in RAW_DIR.rglob("V-*.csv") if p.stem[2:].lower() == session_name.lower()]

    if not s_files or not v_files:
        print(f"[warn] could not find CSVs for {session_name}")
        return

    s_df = load_s_file(s_files[0])
    v_df = load_v_file(v_files[0])
    # Align (reuse existing aligner, using the same 10 Hz grid convention)
    s_df, v_df, lag_s, conf = align_sv(s_df, v_df, target_dt=0.1, max_lag_s=3.0)

    # We need the exact t_grid used during windowing. Rebuild it:
    t_s = s_df["time_s"].to_numpy(dtype=float)
    t0, t1 = t_s[0], t_s[-1]
    t_grid = np.arange(t0, t1, 0.1)

    # Drop invalid rows the same way prepare_data did
    # (the X passed in was already masked) — this mismatch is why we instead
    # build odom targets on the same t_grid and then align to X's length.
    dx_arr, dy_arr, dpsi_arr = compute_odom_targets(v_df, t_grid)

    # valid_mask matches what prepare_data built
    valid = np.isfinite(dx_arr) & np.isfinite(dy_arr) & np.isfinite(dpsi_arr)
    dx_arr = dx_arr[valid]
    dy_arr = dy_arr[valid]
    dpsi_arr = dpsi_arr[valid]

    # We now need the same windowing stride (5) and window length (50)
    # as in prepare_data. But X already has been windowed. Instead of
    # re-windowing, we need to slice the odometry targets to the same
    # endpoints as X.
    #
    # The safest approach: rebuild the endpoint indices by using the fact
    # that prepare_data used stride=5 and window=50, and dropped invalid
    # rows. This is fragile — we solve it by rebuilding windows on the
    # resampled grid and checking that X.shape[0] matches.
    odom_targets = np.stack([dx_arr, dy_arr, dpsi_arr], axis=1)   # (M, 3)
    # If M == X.shape[0], we're done. Otherwise we cannot trust this.
    if odom_targets.shape[0] != X.shape[0]:
        print(f"[warn] {session_name}: odom target count {odom_targets.shape[0]} "
              f"does not match X count {X.shape[0]}; skipping")
        return

    np.savez_compressed(
        npz_path,
        X=X, y=y_speed,
        y_odom=odom_targets.astype(np.float32),
        session=session_name,
    )
    print(f"[ok] {session_name}: odom targets {odom_targets.shape}")


def main():
    sessions = sorted(p.stem for p in PROC_DIR.glob("*.npz"))
    for s in tqdm(sessions):
        try:
            process_one(s)
        except Exception as e:
            print(f"[fail] {s}: {e}")


if __name__ == "__main__":
    main()