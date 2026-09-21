## Development Flow


┌─────────────────┐
│ 1. Data Audit   │  Parse headers, units, timestamps, missing values
└────────┬────────┘
         ↓
┌─────────────────┐
│ 2. Ingestion    │  Loader + schema detection + unit normalization
└────────┬────────┘
         ↓
┌─────────────────┐
│ 3. Alignment    │  S/V clock alignment (confidence-gated)
└────────┬────────┘
         ↓
┌─────────────────┐
│ 4. Windowing    │  5 s causal windows at 10 Hz
└────────┬────────┘
         ↓
┌─────────────────┐
│ 5. Target Gen   │  v (speed), [dx, dy, dψ] (odometry) from V GNSS/ECU
└────────┬────────┘
         ↓
┌─────────────────┐
│ 6. Splits       │  Train/val/test by driver/session — no leakage
└────────┬────────┘
         ↓
┌─────────────────┐
│ 7. FP32 Train   │  VelocityNet + InertialOdomNet + BiasNet
└────────┬────────┘
         ↓
┌─────────────────┐
│ 8. QAT Fine-tune│  Insert fake quant, retrain briefly
└────────┬────────┘
         ↓
┌─────────────────┐
│ 9. int8 Export  │  ONNX/TFLite with size + latency verification
└────────┬────────┘
         ↓
┌─────────────────┐
│ 10. ESKF Fusion │  Train-free; tune Q, R via Allan variance + validation
└────────┬────────┘
         ↓
┌─────────────────┐
│ 11. Map Match   │  HMM/Viterbi with real cached OSM
└────────┬────────┘
         ↓
┌─────────────────┐
│ 12. Eval        │  Real held-out sessions with GNSS masked
└────────┬────────┘
         ↓
┌─────────────────┐
│ 13. Mobile Port │  ONNX Runtime + ESKF in Kotlin/C++
└─────────────────┘




## System Flow


┌──────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1 — SENSOR INGESTION                                                    │
│                                                                              │
│  Phone:                    Edge (external IMU):                               │
│    • Accelerometer (m/s²)    • FOG IMU @ 200 Hz                               │
│    • Gyroscope (rad/s)       • Same unified message format                    │
│    • Magnetometer (µT)       • Higher rate, lower noise                       │
│    • GNSS (lat, lon, h, v)                                                    │
│    • 10 Hz default rate                                                       │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ LAYER 2 — PREPROCESSING & CALIBRATION                                         │
│                                                                              │
│  2.1 Timestamp Normalization                                                  │
│      └─ Convert per-source timestamps to seconds from session start           │
│      └─ Handle S (ms) vs V (s from start of day)                              │
│                                                                              │
│  2.2 Unit Normalization                                                       │
│      └─ km/h → m/s, deg → rad, g → m/s²                                      │
│      └─ Centralized per-column metadata                                       │
│                                                                              │
│  2.3 Phone-to-Vehicle Alignment (per session)                                 │
│      └─ Roll/pitch from gravity vector during stationary period               │
│      └─ Yaw from forward-motion PCA of horizontal accel during calibration    │
│      └─ Confidence metric; re-calibrate if phone shifts (detected)            │
│                                                                              │
│  2.4 S/V Clock Alignment (training only)                                      │
│      └─ Cross-correlate speed derivatives of S GPS and V ECU                  │
│      └─ Confidence-gated: falls back to zero lag if peak ratio < 1.5          │
│                                                                              │
│  2.5 Resampling to fixed 10 Hz grid                                           │
│      └─ Uses each source's own timestamps                                     │
│                                                                              │
│  2.6 Causal Windowing                                                         │
│      └─ 5-second (50-sample) sliding windows, stride 5                        │
│      └─ No future context: safe for real-time inference                       │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ LAYER 3 — LEARNED MODELS (QAT-ready)                                          │
│                                                                              │
│  ┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────┐   │
│  │ VelocityNet             │  │ InertialOdomNet         │  │ BiasNet     │   │
│  │                         │  │                         │  │             │   │
│  │ TCN (causal, dilated)   │  │ ResNet1D causal blocks  │  │ 1D-CNN      │   │
│  │   ↓                     │  │   ↓                     │  │   ↓         │   │
│  │ GRU                     │  │ GRU                     │  │             │   │
│  │   ↓                     │  │   ↓                     │  │             │   │
│  │ ┌─ speed head (Softplus)│  │ ┌─ dx, dy, dpsi heads    │  │ Δaccel,     │   │
│  │ └─ logvar head          │  │ └─ logvar head (x,y)     │  │ Δgyro, σ²   │   │
│  │                         │  │                         │  │             │   │
│  │ Output: [v, logσ²]      │  │ Output: [dx,dy,dψ,σ²]   │  │ Residual    │   │
│  │ 82k params (~320 KB)    │  │ ~150k params (~600 KB)  │  │ ~20k params │   │
│  └─────────────────────────┘  └─────────────────────────┘  └─────────────┘   │
│                                                                              │
│  Loss: Gaussian NLL (learns aleatoric uncertainty)                            │
│  Training split: by driver/route — no window leakage                          │
│  All ops int8-friendly (Conv1d, GRU, Linear, BatchNorm)                       │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ LAYER 4 — ERROR-STATE KALMAN FILTER                                           │
│                                                                              │
│  State (16): [p_e, p_n, p_u, v_e, v_n, v_u, q_w, q_x, q_y, q_z,               │
│              b_ax, b_ay, b_az, b_ωx, b_ωy, b_ωz]                              │
│                                                                              │
│  ┌─ PROPAGATION ──────────────────────────────────────────────────┐           │
│  │  Variable-rate IMU integration using actual dt                  │           │
│  │  Attitude: quaternion multiplication with gyro-rate              │           │
│  │  Velocity: rotate accel to ENU, subtract gravity + bias          │           │
│  │  Position: integrate velocity                                    │           │
│  │  Covariance: F·P·Fᵀ + Q (Q from Allan variance)                  │           │
│  └─────────────────────────────────────────────────────────────────┘           │
│                                                                              │
│  ┌─ MEASUREMENT UPDATES ──────────────────────────────────────────┐           │
│  │  GNSS (when available):                                        │           │
│  │    • Position (lat, lon, h) with reported accuracy → R          │           │
│  │    • Velocity (v_e, v_n) with accuracy → R                      │           │
│  │    • Heading from GNSS course                                   │           │
│  │                                                                 │           │
│  │  AI velocity (pseudo-measurement):                              │           │
│  │    • z = v_fwd, H = [body-x direction],                         │           │
│  │    • R = exp(log_var) from VelocityNet                          │           │
│  │                                                                 │           │
│  │  AI displacement (pseudo-measurement, low rate):                │           │
│  │    • z = [dx, dy, dψ], R = diag(exp(logσx²), exp(logσy²), σψ²) │           │
│  │                                                                 │           │
│  │  NHC (adaptive):                                                │           │
│  │    • v_lateral ≈ 0 with σ = 0.1 m/s (straight)                  │           │
│  │    • v_vertical ≈ 0 with σ = 0.1 m/s                            │           │
│  │    • Relax during turns (detected from gyro + map curvature)    │           │
│  │                                                                 │           │
│  │  ZUPT/ZARU (learned gate):                                      │           │
│  │    • Triggered when stationary probability from VelocityNet     │           │
│  │    • Anchors position and resets velocity                       │           │
│  └─────────────────────────────────────────────────────────────────┘           │
│                                                                              │
│  Chi-square gating: reject any measurement whose NIS > threshold              │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ LAYER 5 — PROBABILISTIC MAP MATCHING (HMM/Viterbi)                            │
│                                                                              │
│  Inputs: predicted state + covariance, cached OSM (PBF/GraphML)               │
│                                                                              │
│  For each 1 Hz tick:                                                          │
│    1. Candidate generation: road segments within ~30 m corridor               │
│    2. Emission probability: exp(-d²/2σ_d²) where d = distance to centerline   │
│    3. Transition probability:                                                 │
│         exp(-|route_distance - Euclidean|/β) · heading_agreement              │
│    4. Viterbi decoding: most likely road sequence over last N timestamps      │
│    5. Soft update: project onto matched road with inflated covariance         │
│                                                                              │
│  "No confident road" mode: if max posterior < threshold, skip update          │
│  (handles flyovers, service roads, missing OSM)                               │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ LAYER 6 — MODE MANAGER & OUTPUT SMOOTHING                                     │
│                                                                              │
│  Modes (hysteresis + dwell time):                                             │
│    GNSS_AIDED → GNSS_DEGRADED → INS_HOLDOVER → GNSS_REACQUISITION → GNSS_AIDED│
│                                                                              │
│  Triggers:                                                                    │
│    • Fix age > 1.5 s → GNSS_DEGRADED                                          │
│    • Fix age > 3.0 s OR accuracy > 30 m → INS_HOLDOVER                        │
│    • Fix acquired & NIS passes → GNSS_REACQUISITION                           │
│                                                                              │
│  Reacquisition smoothing: 1.5-second cosine-bell weight                       │
│    w(t) = 0.5·(1 − cos(π·t/T))                                                │
│  Eliminates position jump when GNSS returns                                   │
│                                                                              │
│  Output: {position, velocity, attitude, covariance, mode, confidence}         │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ LAYER 7 — DEPLOYMENT                                                          │
│                                                                              │
│  Android app:                                                                 │
│    • Sensor streaming (SensorManager)                                         │
│    • On-device ONNX Runtime / TFLite inference                                │
│    • ESKF in C++/Kotlin native                                                │
│    • OSM tiles + cached GraphML                                               │
│    • Map UI (Google Maps SDK or MapLibre)                                     │
│    • Mode indicator (green/yellow/red)                                        │
│                                                                              │
│  Edge engine:                                                                 │
│    • Same estimator in C++/Rust                                               │
│    • External FOG IMU ingestion at 200 Hz                                     │
│    • Configurable map region                                                  │
└──────────────────────────────────────────────────────────────────────────────┘