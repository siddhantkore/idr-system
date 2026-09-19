# src/idr/io/loader.py
import numpy as np
import pandas as pd
from pathlib import Path
from .schema import detect_s_schema, detect_v_schema


def _read_csv_robust(path):
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, encoding="utf-8", encoding_errors="replace")


def _rename(df, mapping):
    """Rename mapped columns to canonical names; drop unmapped."""
    keep = [mapping[k] for k in mapping]
    out = df[keep].rename(columns={v: k for k, v in mapping.items()})
    return out


def load_s_file(path):
    df = _read_csv_robust(path)
    mapping = detect_s_schema(df.columns)
    if "time_ms" not in mapping:
        raise KeyError(f"No time column found in {path}. Columns: {list(df.columns)}")
    df = _rename(df, mapping)

    if "gps_speed" in df.columns:
        df["gps_speed_ms"] = pd.to_numeric(df["gps_speed"], errors="coerce") / 3.6
    if "time_ms" in df.columns:
        df["time_s"] = pd.to_numeric(df["time_ms"], errors="coerce") / 1000.0
    for col in ("ori_yaw", "ori_pitch", "ori_roll"):
        if col in df.columns:
            df[col + "_rad"] = np.deg2rad(pd.to_numeric(df[col], errors="coerce"))
    return df


def load_v_file(path):
    df = _read_csv_robust(path)
    mapping = detect_v_schema(df.columns)
    if "time_s" not in mapping:
        raise KeyError(f"No time column found in {path}. Columns: {list(df.columns)}")
    df = _rename(df, mapping)

    if "velocity" in df.columns:
        df["velocity_ms"] = pd.to_numeric(df["velocity"], errors="coerce") / 3.6
    if "yaw_rate" in df.columns:
        df["yaw_rate_rads"] = np.deg2rad(pd.to_numeric(df["yaw_rate"], errors="coerce"))
    if "ind_long_acc" in df.columns:
        df["ind_long_acc_ms2"] = pd.to_numeric(df["ind_long_acc"], errors="coerce") * 9.80665
    if "ind_lat_acc" in df.columns:
        df["ind_lat_acc_ms2"] = pd.to_numeric(df["ind_lat_acc"], errors="coerce") * 9.80665
    return df