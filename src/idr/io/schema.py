# src/idr/io/schema.py
import re
import unicodedata


def normalize_header(h: str) -> str:
    """
    Normalize a CSV header for matching:
      - strip whitespace
      - replace non-ASCII (Â°, ², Î¼) by transliterating / stripping
      - lowercase
      - collapse internal whitespace
      - remove parenthetical unit suffixes like (m/s²), (degrees), (km/hr)
      - remove trailing punctuation
    """
    if h is None:
        return ""
    # Strip accents / weird bytes by encoding to ascii ignoring errors
    h = unicodedata.normalize("NFKD", h)
    h = h.encode("ascii", "ignore").decode("ascii")
    h = h.strip().lower()
    # Remove parenthetical suffixes (units)
    h = re.sub(r"\(.*?\)", "", h)
    # Collapse whitespace
    h = re.sub(r"\s+", " ", h).strip()
    return h


# Canonical S schema. Keys are our internal names; values are lists of
# normalized header candidates that map to that key.
S_CANONICAL = {
    "gps_lat":         ["gps latitude"],
    "gps_lon":         ["gps longitude"],
    "gps_alt":         ["gps altitude"],
    "gps_speed":       ["gps speed"],
    "gps_accuracy":    ["gps accuracy"],
    "gps_orientation": ["gps orientation"],
    "gps_sats":        ["gps satellites in range"],
    "time_ms":         ["time since start"],
    "date":            ["date"],
    "acc_x":           ["accelerometer x"],
    "acc_y":           ["accelerometer y"],
    "acc_z":           ["accelerometer z"],
    "grav_x":          ["gravity x"],
    "grav_y":          ["gravity y"],
    "grav_z":          ["gravity z"],
    "gyro_x":          ["gyroscope yaw", "gyroscope x"],
    "gyro_y":          ["gyroscope pitch", "gyroscope y"],
    "gyro_z":          ["gyroscope roll", "gyroscope z"],
    "mag_x":           ["magnetic field x"],
    "mag_y":           ["magnetic field y"],
    "mag_z":           ["magnetic field z"],
    "ori_yaw":         ["orientation yaw", "orientation azimuth"],
    "ori_pitch":       ["orientation pitch"],
    "ori_roll":        ["orientation roll"],
}

V_CANONICAL = {
    "gps_sats":        ["no of gps satellites available"],
    "time_s":          ["time since start of day"],
    "lat":             ["latitude"],
    "lon":             ["longitude"],
    "velocity":        ["velocity"],
    "heading":         ["heading"],
    "height":          ["height"],
    "v_vel":           ["vertical velocity"],
    "sample_period":   ["sample period"],
    "steering":        ["steering angle"],
    "wheel_fl":        ["wheel speed front left"],
    "wheel_fr":        ["wheel speed front right"],
    "wheel_rl":        ["wheel speed rear left"],
    "wheel_rr":        ["wheel speed rear right"],
    "yaw_rate":        ["yaw rate"],
    "indicated_speed": ["indicated vehicle speed"],
    "ind_long_acc":    ["indicated longitudinal acceleration"],
    "ind_lat_acc":     ["indicated lateral acceleration"],
    "handbrake":       ["handbrake"],
    "gear":            ["gear"],
    "engine_rpm":      ["engine speed"],
    "brake_pressure":  ["brake pressure"],
    "brake_pos":       ["brake position"],
}


def _map_columns(df_columns, canonical):
    """Return dict {canonical_name: actual_column_name}."""
    normalized = {normalize_header(c): c for c in df_columns}
    mapping = {}
    for canon, candidates in canonical.items():
        for cand in candidates:
            if cand in normalized:
                mapping[canon] = normalized[cand]
                break
    return mapping


def detect_s_schema(columns):
    return _map_columns(columns, S_CANONICAL)


def detect_v_schema(columns):
    return _map_columns(columns, V_CANONICAL)