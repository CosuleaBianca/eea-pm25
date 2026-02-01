import os
import json
import time
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Iterable, Tuple

import tensorflow as tf
from tensorflow.keras import layers, Model

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'train_models'))

# ==========================================================
# CONFIG
# ==========================================================
CSV_PATH = "ml_ready_dataset_full_realistic.csv"

HORIZONS = [1, 3, 6, 12, 24]

TRAIN_END = pd.Timestamp("2022-12-31 23:00:00", tz="UTC")
TEST_START = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
TEST_END = pd.Timestamp("2024-12-31 23:00:00", tz="UTC")

LOOKBACK_H = 168
GAP_BREAK_HOURS = 2

BATCH_SIZE = 512
EPOCHS = 50
LR = 5e-4
PATIENCE = 6

# Output (single-output)
OUTPUT_METRICS_PATH = "results/lstm_residual_metrics.csv"
OUTPUT_PREDS_PATH = "results/lstm_residual_predictions_sample.csv"
OUTPUT_RUN_INFO_JSON = "train_models/global/runs_info/lstm_residual_run_info.json"
MODELS_DIR = "models/lstm_residual_models"
HIST_DIR = "train_models/history/lstm_residual_histories"
SCALERS_JSON = "train_models/scalers/lstm_residual_station_scalers.json"

STATIC_FEATURE_NAMES = {"Latitude", "Longitude", "Altitude"}
STATIC_FEATURE_PREFIXES = ("StationType_", "StationArea_")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# Create all output directories upfront to prevent data loss
print("Creating output directories...")
os.makedirs("train_models/global/runs_info", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SCALERS_JSON), exist_ok=True)  # train_models/scalers/
os.makedirs(HIST_DIR, exist_ok=True)
print("All directories created successfully.")

# ==========================================================
# GPU safety
# ==========================================================
def configure_gpu_memory_growth():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            print("No GPU detected by TensorFlow. Running on CPU.")
            return
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s).")
    except Exception as e:
        print("WARNING: Could not set GPU memory growth:", e)

# ==========================================================
# METRICS (LR-like)
# ==========================================================
def r2_score_np(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

def mae_np(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse_np(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def bias_np(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(y_pred - y_true))

def calculate_metrics_df(sub_df: pd.DataFrame) -> pd.Series:
    y_true = sub_df["y_true"].to_numpy()
    y_pred = sub_df["y_pred"].to_numpy()
    if len(y_true) == 0:
        return pd.Series({"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "Bias": np.nan, "Count": 0})
    return pd.Series({
        "MAE": mae_np(y_true, y_pred),
        "RMSE": rmse_np(y_true, y_pred),
        "R2": r2_score_np(y_true, y_pred),
        "Bias": bias_np(y_true, y_pred),
        "Count": float(len(y_true))
    })

# ==========================================================
# DATA HELPERS
# ==========================================================
def make_station_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SiteNumber"] = df["SiteNumber"].astype(str)
    df["station_key"] = df["Country"].astype(str) + "_" + df["SiteNumber"].astype(str)
    return df

def recover_station_area_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    area_cols = [c for c in df.columns if c.startswith("StationArea_")]
    if not area_cols:
        df["StationArea_Label"] = "Unknown"
        return df
    df["StationArea_Label"] = df[area_cols].idxmax(axis=1).apply(lambda x: x.replace("StationArea_", ""))
    return df

def build_multi_horizon_targets_like_lr(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    df = df.copy()
    df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True, errors="coerce")
    df = df.sort_values(["Country", "SiteNumber", "dt_utc"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["Country", "SiteNumber", "dt_utc"], keep="first")

    df["_row_id"] = np.arange(len(df), dtype=np.int64)
    out_rows = []

    for (country, site), g in df.groupby(["Country", "SiteNumber"], sort=False):
        g = g.sort_values("dt_utc")
        idx_orig = g["dt_utc"]
        row_ids = g["_row_id"].to_numpy()

        g_idx = g.set_index("dt_utc")
        g_hourly = g_idx.asfreq("h")

        targets = {}
        for h in horizons:
            targets[f"y_h{h}"] = g_hourly["PM2.5"].shift(-h)

        targets_df = pd.DataFrame(targets, index=g_hourly.index)
        mapped = targets_df.loc[idx_orig].copy()
        mapped["_row_id"] = row_ids
        out_rows.append(mapped)

    targets_all = pd.concat(out_rows, axis=0, ignore_index=True)
    df = df.merge(targets_all, on="_row_id", how="left").drop(columns=["_row_id"])
    return df

def select_feature_columns_like_lstm_script(df: pd.DataFrame) -> List[str]:
    features_from_file = [
        "Latitude", "Longitude", "Altitude",
        "StationType_background", "StationType_industrial", "StationType_traffic",
        "StationArea_rural", "StationArea_rural-nearcity", "StationArea_suburban", "StationArea_urban",
        "temperature_2m", "relative_humidity_2m", "dew_point_2m", "wind_u", "wind_v",
        "precipitation", "surface_pressure",
        "PM2.5", "NO2", "PM10",
        "hour", "day_of_week", "day_of_month", "month", "year", "is_weekend",
        "season", "hour_sin", "hour_cos", "month_sin", "month_cos",
        "NO2_lag_12h", "NO2_lag_168h", "NO2_lag_1h", "NO2_lag_24h", "NO2_lag_2h", "NO2_lag_3h", "NO2_lag_6h",
        "PM10_lag_12h", "PM10_lag_168h", "PM10_lag_1h", "PM10_lag_24h", "PM10_lag_2h", "PM10_lag_3h", "PM10_lag_6h",
        "PM2.5_lag_12h", "PM2.5_lag_168h", "PM2.5_lag_1h", "PM2.5_lag_24h", "PM2.5_lag_2h", "PM2.5_lag_3h", "PM2.5_lag_6h",
        "NO2_rolling_mean_12h", "NO2_rolling_mean_24h", "NO2_rolling_mean_3h", "NO2_rolling_mean_6h",
        "PM10_rolling_mean_12h", "PM10_rolling_mean_24h", "PM10_rolling_mean_3h", "PM10_rolling_mean_6h",
        "PM2.5_rolling_mean_12h", "PM2.5_rolling_mean_24h", "PM2.5_rolling_mean_3h", "PM2.5_rolling_mean_6h",
        "NO2_rolling_std_12h", "NO2_rolling_std_24h", "NO2_rolling_std_3h", "NO2_rolling_std_6h",
        "PM10_rolling_std_12h", "PM10_rolling_std_24h", "PM10_rolling_std_3h", "PM10_rolling_std_6h",
        "PM2.5_rolling_std_12h", "PM2.5_rolling_std_24h", "PM2.5_rolling_std_3h", "PM2.5_rolling_std_6h",
    ]
    non_feature_cols = {"Country", "SiteNumber", "dt_utc", "dt_local", "Start", "PM2.5_next", "station_key", "StationArea_Label"}
    feature_cols = [c for c in features_from_file if c in df.columns and c not in non_feature_cols]
    if "PM2.5" not in feature_cols:
        raise ValueError("PM2.5 must be present in feature_cols.")
    return feature_cols

def select_static_feature_indices(feature_cols: List[str]) -> List[int]:
    static_cols = []
    for col in feature_cols:
        if col in STATIC_FEATURE_NAMES or col.startswith(STATIC_FEATURE_PREFIXES):
            static_cols.append(col)
    return [feature_cols.index(c) for c in static_cols]

def compute_segments(g: pd.DataFrame, gap_break_hours: int = 2) -> List[pd.DataFrame]:
    g = g.sort_values("dt_utc")
    gaps = g["dt_utc"].diff().dt.total_seconds().div(3600)
    breakpoints = np.where(gaps > gap_break_hours)[0]
    if len(breakpoints) == 0:
        return [g]
    segs = []
    start = 0
    for bp in breakpoints:
        segs.append(g.iloc[start:bp].copy())
        start = bp
    segs.append(g.iloc[start:].copy())
    return [s for s in segs if len(s) > 0]

# ==========================================================
# SCALERS (TRAIN-ONLY)
# ==========================================================
@dataclass
class StationScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

def fit_station_scalers(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    static_feature_idxs: Optional[List[int]] = None,
    global_mean: Optional[np.ndarray] = None,
    global_std: Optional[np.ndarray] = None,
) -> Dict[str, StationScaler]:
    scalers = {}
    for station_key, g in train_df.groupby("station_key", sort=False):
        arr = g[feature_cols].to_numpy(dtype=np.float32)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        if static_feature_idxs and global_mean is not None and global_std is not None:
            mean = mean.astype(np.float32, copy=True)
            std = std.astype(np.float32, copy=True)
            mean[static_feature_idxs] = global_mean[static_feature_idxs]
            std[static_feature_idxs] = global_std[static_feature_idxs]
        scalers[station_key] = StationScaler(mean=mean.astype(np.float32), std=std.astype(np.float32))
    return scalers

def save_scalers_json(scalers: Dict[str, StationScaler], feature_cols: List[str], path: str):
    payload = {
        "feature_cols": feature_cols,
        "scalers": {k: {"mean": v.mean.tolist(), "std": v.std.tolist()} for k, v in scalers.items()}
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

# ==========================================================
# MODEL (RESIDUAL, SINGLE-OUTPUT)
# ==========================================================
class R2Metric(tf.keras.metrics.Metric):
    def __init__(self, name="r2", **kwargs):
        super().__init__(name=name, **kwargs)
        self.ssr = self.add_weight(name="ssr", initializer="zeros")
        self.sum_y = self.add_weight(name="sum_y", initializer="zeros")
        self.sum_y_sq = self.add_weight(name="sum_y_sq", initializer="zeros")
        self.sum_w = self.add_weight(name="sum_w", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        if sample_weight is None:
            w = tf.ones_like(y_true)
        else:
            w = tf.cast(sample_weight, tf.float32) * tf.ones_like(y_true)
        err = y_true - y_pred
        self.ssr.assign_add(tf.reduce_sum(w * tf.square(err)))
        self.sum_y.assign_add(tf.reduce_sum(w * y_true))
        self.sum_y_sq.assign_add(tf.reduce_sum(w * tf.square(y_true)))
        self.sum_w.assign_add(tf.reduce_sum(w))

    def result(self):
        eps = tf.constant(1e-7, dtype=tf.float32)
        ss_tot = self.sum_y_sq - tf.square(self.sum_y) / (self.sum_w + eps)
        return tf.where(ss_tot > 0.0, 1.0 - (self.ssr / (ss_tot + eps)), 0.0)

    def reset_states(self):
        for v in self.variables:
            v.assign(0.0)

def build_single_model(n_features: int, n_stations: int, pm25_index: int) -> Model:
    seq_in = layers.Input(shape=(LOOKBACK_H, n_features), name="x_seq")
    sid_in = layers.Input(shape=(), dtype="int32", name="station_id")

    # baseline pm25_t (normalized) from last timestep
    pm25_t = layers.Lambda(lambda z: z[:, -1, pm25_index:pm25_index + 1], name="pm25_t")(seq_in)  # [B,1]

    # station embedding
    emb_dim = 8
    sid_emb = layers.Embedding(input_dim=n_stations, output_dim=emb_dim, name="station_emb")(sid_in)
    sid_emb = layers.Dense(emb_dim, activation="relu",
                           kernel_regularizer=tf.keras.regularizers.l2(1e-6))(sid_emb)
    sid_rep = layers.RepeatVector(LOOKBACK_H)(sid_emb)

    x = layers.Concatenate(axis=-1)([seq_in, sid_rep])

    x = layers.LSTM(32, return_sequences=True,
                    dropout=0.3,
                    kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.LSTM(16, return_sequences=False,
                    dropout=0.3,
                    kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

    x = layers.Dense(32, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)

    # delta head
    d = layers.Dense(16, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    d = layers.Dropout(0.20)(d)
    delta = layers.Dense(1, name="delta")(d)

    y_hat = layers.Add(name="y_hat")([pm25_t, delta])  # residual add

    model = Model(inputs=[seq_in, sid_in], outputs=y_hat)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"), R2Metric(name="r2")]
    )
    return model

# ==========================================================
# EXAMPLE BUILDERS (single-output per horizon)
# ==========================================================
def build_examples_for_segment_single(
    seg: pd.DataFrame,
    station_idx: int,
    feature_cols: List[str],
    scaler: StationScaler,
    horizon: int,
    lookback_h: int = 168,
    require_no_nan: bool = True,
    anchor_start: Optional[pd.Timestamp] = None,
    anchor_end: Optional[pd.Timestamp] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
      X_seq: [n, lookback, n_features] normalized
      sid:   [n]
      y:     [n, 1] normalized target y(t+h)
      T:     [n] anchor timestamps datetime64[ns]
      pm25_t_norm: [n] normalized baseline at t
    """
    seg = seg.sort_values("dt_utc").reset_index(drop=True)

    x = seg[feature_cols].to_numpy(dtype=np.float32)
    tstamp = seg["dt_utc"].to_numpy(dtype="datetime64[ns]")

    x_norm = scaler.transform(x)

    pm25_idx = feature_cols.index("PM2.5")
    y_pm25 = seg["PM2.5"].to_numpy(dtype=np.float32)
    pm25_norm = (y_pm25 - scaler.mean[pm25_idx]) / scaler.std[pm25_idx]

    n = len(seg)
    start_t = lookback_h - 1
    end_t = n - horizon  # need t+horizon to exist in the mapped target column, but y_h already stored on row t

    X_list, sid_list, y_list, T_list, base_list = [], [], [], [], []

    col = f"y_h{horizon}"
    if col not in seg.columns:
        return (np.empty((0, lookback_h, len(feature_cols)), np.float32),
                np.empty((0,), np.int32),
                np.empty((0, 1), np.float32),
                np.empty((0,), dtype="datetime64[ns]"),
                np.empty((0,), np.float32))

    for t in range(start_t, end_t):
        ts = tstamp[t]
        if anchor_start is not None and ts < np.datetime64(anchor_start.to_datetime64()):
            continue
        if anchor_end is not None and ts > np.datetime64(anchor_end.to_datetime64()):
            continue

        x_win = x_norm[t - lookback_h + 1:t + 1, :]
        if x_win.shape[0] != lookback_h:
            continue
        if require_no_nan and np.isnan(x_win).any():
            continue

        y_val = seg[col].iloc[t]
        if pd.isna(y_val):
            continue

        # normalize target using PM2.5 stats
        y_norm = (float(y_val) - float(scaler.mean[pm25_idx])) / float(scaler.std[pm25_idx])

        base = pm25_norm[t]
        if require_no_nan and (np.isnan(y_norm) or np.isnan(base)):
            continue

        X_list.append(x_win)
        sid_list.append(station_idx)
        y_list.append([y_norm])
        T_list.append(ts)
        base_list.append(base)

    if len(X_list) == 0:
        return (np.empty((0, lookback_h, len(feature_cols)), np.float32),
                np.empty((0,), np.int32),
                np.empty((0, 1), np.float32),
                np.empty((0,), dtype="datetime64[ns]"),
                np.empty((0,), np.float32))

    return (
        np.stack(X_list).astype(np.float32),
        np.array(sid_list, dtype=np.int32),
        np.array(y_list, dtype=np.float32),
        np.array(T_list, dtype="datetime64[ns]"),
        np.array(base_list, dtype=np.float32),
    )

def make_train_val_ds_for_h(
    df_all: pd.DataFrame,
    station_to_idx: Dict[str, int],
    scalers: Dict[str, StationScaler],
    feature_cols: List[str],
    h: int,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
):
    """
    Train: anchors anywhere <= train_end
    Val: anchors within [val_start, val_end] with warmup available
    """
    train_build_df = df_all[df_all["dt_utc"] <= train_end].copy()
    warmup = pd.Timedelta(hours=LOOKBACK_H - 1)
    val_build_df = df_all[(df_all["dt_utc"] >= (val_start - warmup)) & (df_all["dt_utc"] <= val_end)].copy()
    min_seg_len = LOOKBACK_H + h + 1

    def train_gen():
        for station_key, g in train_build_df.groupby("station_key", sort=False):
            if station_key not in station_to_idx or station_key not in scalers:
                continue
            scaler = scalers[station_key]
            sid = station_to_idx[station_key]
            segs = compute_segments(g, gap_break_hours=GAP_BREAK_HOURS)
            for seg in segs:
                if len(seg) < min_seg_len:
                    continue
                Xs, sids, ys, _, _ = build_examples_for_segment_single(
                    seg=seg, station_idx=sid, feature_cols=feature_cols,
                    scaler=scaler, horizon=h, lookback_h=LOOKBACK_H,
                    require_no_nan=True, anchor_start=None, anchor_end=train_end
                )
                if len(Xs) == 0:
                    continue
                for i in range(len(Xs)):
                    yield ({"x_seq": Xs[i], "station_id": sids[i]}, ys[i])

    def val_gen():
        for station_key, g in val_build_df.groupby("station_key", sort=False):
            if station_key not in station_to_idx or station_key not in scalers:
                continue
            scaler = scalers[station_key]
            sid = station_to_idx[station_key]
            segs = compute_segments(g, gap_break_hours=GAP_BREAK_HOURS)
            for seg in segs:
                if len(seg) < min_seg_len:
                    continue
                Xs, sids, ys, _, _ = build_examples_for_segment_single(
                    seg=seg, station_idx=sid, feature_cols=feature_cols,
                    scaler=scaler, horizon=h, lookback_h=LOOKBACK_H,
                    require_no_nan=True, anchor_start=val_start, anchor_end=val_end
                )
                if len(Xs) == 0:
                    continue
                for i in range(len(Xs)):
                    yield ({"x_seq": Xs[i], "station_id": sids[i]}, ys[i])

    n_features = len(feature_cols)
    output_sig = (
        {
            "x_seq": tf.TensorSpec(shape=(LOOKBACK_H, n_features), dtype=tf.float32),
            "station_id": tf.TensorSpec(shape=(), dtype=tf.int32),
        },
        tf.TensorSpec(shape=(1,), dtype=tf.float32),
    )

    train_ds = tf.data.Dataset.from_generator(train_gen, output_signature=output_sig)
    train_ds = train_ds.shuffle(50000, seed=SEED, reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(val_gen, output_signature=output_sig)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

def make_test_generator_for_h(
    df_all: pd.DataFrame,
    station_to_idx: Dict[str, int],
    scalers: Dict[str, StationScaler],
    feature_cols: List[str],
    h: int,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> Iterable[Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray]]]:
    warmup = pd.Timedelta(hours=LOOKBACK_H - 1)
    test_build_df = df_all[(df_all["dt_utc"] >= (test_start - warmup)) & (df_all["dt_utc"] <= test_end)].copy()
    min_seg_len = LOOKBACK_H + h + 1

    for station_key, g in test_build_df.groupby("station_key", sort=False):
        if station_key not in station_to_idx or station_key not in scalers:
            continue
        scaler = scalers[station_key]
        sid = station_to_idx[station_key]
        segs = compute_segments(g, gap_break_hours=GAP_BREAK_HOURS)
        for seg in segs:
            if len(seg) < min_seg_len:
                continue
            Xs, sids, ys, Ts, base_norm = build_examples_for_segment_single(
                seg=seg, station_idx=sid, feature_cols=feature_cols, scaler=scaler,
                horizon=h, lookback_h=LOOKBACK_H, require_no_nan=True,
                anchor_start=test_start, anchor_end=test_end
            )
            if len(Xs) == 0:
                continue
            meta = {
                "station_key": np.array([station_key] * len(Xs), dtype=object),
                "t_anchor": Ts,
                "sid": sids.astype(np.int32),
                "pm25_t_norm": base_norm.astype(np.float32),
            }
            yield ({"x_seq": Xs, "station_id": sids}, ys, meta)

# ==========================================================
# MAIN
# ==========================================================
def main():
    run_t0 = time.time()
    configure_gpu_memory_growth()

    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True, errors="coerce")

    # season encoding
    if "season" in df.columns and df["season"].dtype == object:
        season_map = {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3, "Fall": 3}
        df["season"] = df["season"].map(season_map)
    if "season" in df.columns:
        df["season"] = df["season"].fillna(-1).astype("float32")

    df = make_station_key(df)
    df = recover_station_area_label(df)
    df = df.sort_values(["station_key", "dt_utc"]).drop_duplicates(["station_key", "dt_utc"], keep="first")

    print("Building multi-horizon targets (LR-aligned)...")
    df = build_multi_horizon_targets_like_lr(df, HORIZONS)

    feature_cols = select_feature_columns_like_lstm_script(df)
    pm25_feat_idx = feature_cols.index("PM2.5")
    static_feature_idxs = select_static_feature_indices(feature_cols)

    # keep only stations common to train & test
    train_st = set(df.loc[df["dt_utc"] <= TRAIN_END, "station_key"].unique())
    test_st = set(df.loc[(df["dt_utc"] >= TEST_START) & (df["dt_utc"] <= TEST_END), "station_key"].unique())
    common_st = sorted(list(train_st & test_st))
    df = df[df["station_key"].isin(common_st)].copy()
    print(f"Stations common to train/test: {len(common_st)}")

    train_df = df[df["dt_utc"] <= TRAIN_END].copy()
    print(f"Train rows (raw): {len(train_df):,}")

    # Fit scalers on TRAIN only (drop rows with NaN features)
    print("Fitting per-station scalers on TRAIN-only...")
    train_df_clean = train_df.dropna(subset=feature_cols)
    global_mean = np.nanmean(train_df_clean[feature_cols].to_numpy(dtype=np.float32), axis=0)
    global_std = np.nanstd(train_df_clean[feature_cols].to_numpy(dtype=np.float32), axis=0)
    global_std = np.where(global_std < 1e-6, 1.0, global_std)

    scalers = fit_station_scalers(
        train_df_clean,
        feature_cols,
        static_feature_idxs=static_feature_idxs,
        global_mean=global_mean,
        global_std=global_std,
    )
    save_scalers_json(scalers, feature_cols, SCALERS_JSON)
    print("Saved scalers:", SCALERS_JSON)

    station_to_idx = {k: i for i, k in enumerate(common_st)}
    n_stations = len(common_st)

    # For early stopping, use inner validation from within train window
    VAL_INNER_START = pd.Timestamp("2022-07-01 00:00:00", tz="UTC")
    VAL_INNER_END = TRAIN_END

    # For LR-like reporting, map meta at anchors from test window
    test_meta = df[(df["dt_utc"] >= TEST_START) & (df["dt_utc"] <= TEST_END)][
        ["station_key", "dt_utc", "Country", "SiteNumber", "StationArea_Label"]
    ].drop_duplicates(["station_key", "dt_utc"]).copy()

    meta_key = list(zip(test_meta["station_key"].astype(str).values, test_meta["dt_utc"].values.astype("datetime64[ns]")))
    meta_val = list(zip(test_meta["Country"].astype(str).values,
                        test_meta["SiteNumber"].astype(str).values,
                        test_meta["StationArea_Label"].astype(str).values))
    meta_lookup = dict(zip(meta_key, meta_val))

    # Precompute inverse transform params for PM2.5 per station
    idx_to_station = {v: k for k, v in station_to_idx.items()}
    station_mean = np.zeros(n_stations, dtype=np.float32)
    station_std = np.ones(n_stations, dtype=np.float32)
    for i in range(n_stations):
        sk = idx_to_station[i]
        station_mean[i] = scalers[sk].mean[pm25_feat_idx]
        station_std[i] = scalers[sk].std[pm25_feat_idx]

    # Containers
    per_h_info = {}
    all_pred_rows = []
    pred_sample = []
    sample_limit = 20000

    # Train/eval a model per horizon
    for h in HORIZONS:
        print("\n" + "=" * 70)
        print(f"Training single-output residual LSTM for horizon h={h}...")
        print("=" * 70)

        tf.keras.backend.clear_session()

        train_ds, val_ds = make_train_val_ds_for_h(
            df_all=df,
            station_to_idx=station_to_idx,
            scalers=scalers,
            feature_cols=feature_cols,
            h=h,
            train_end=TRAIN_END,
            val_start=VAL_INNER_START,
            val_end=VAL_INNER_END
        )

        model = build_single_model(n_features=len(feature_cols), n_stations=n_stations, pm25_index=pm25_feat_idx)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_mae",
                patience=PATIENCE,
                restore_best_weights=True,
                min_delta=1e-4
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_mae",
                factor=0.5,
                patience=2,
                min_lr=1e-5,
                verbose=1
            ),
        ]

        t_fit0 = time.time()
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        fit_seconds = time.time() - t_fit0

        # Save model & history
        model_path = os.path.join(MODELS_DIR, f"lstm_residual_h{h}.keras")
        model.save(model_path)
        print("Saved model:", model_path)
        print(f"Training completed in {fit_seconds:.1f}s")

        hist_df = pd.DataFrame(history.history)
        hist_df.insert(0, "epoch", np.arange(1, len(hist_df) + 1))
        hist_path = os.path.join(HIST_DIR, f"history_h{h}.csv")
        hist_df.to_csv(hist_path, index=False)
        print("Saved history:", hist_path)

        # Predict on test (streaming)
        print("Predicting on TEST (streaming)...")
        gen = make_test_generator_for_h(
            df_all=df,
            station_to_idx=station_to_idx,
            scalers=scalers,
            feature_cols=feature_cols,
            h=h,
            test_start=TEST_START,
            test_end=TEST_END
        )

        test_count_h = 0
        for inputs, y_true_norm, meta in gen:
            # predict normalized y_hat (shape [n,1])
            y_pred_norm = model.predict(inputs, batch_size=BATCH_SIZE, verbose=0)

            sid = meta["sid"]
            std_s = station_std[sid][:, None]
            mean_s = station_mean[sid][:, None]

            y_true = y_true_norm * std_s + mean_s
            y_pred = y_pred_norm * std_s + mean_s

            station_keys = meta["station_key"]
            t_anchor = meta["t_anchor"]

            for i in range(len(station_keys)):
                sk = str(station_keys[i])
                ta = np.datetime64(t_anchor[i]).astype("datetime64[ns]")
                m = meta_lookup.get((sk, ta), None)
                if m is None:
                    continue
                country, sitenumber, area_label = m

                all_pred_rows.append({
                    "Country": country,
                    "SiteNumber": sitenumber,
                    "StationArea_Label": area_label,
                    "dt_utc": ta,
                    "horizon": int(h),
                    "y_true": float(y_true[i, 0]),
                    "y_pred": float(y_pred[i, 0]),
                })

                if len(pred_sample) < sample_limit:
                    pred_sample.append({
                        "station_key": sk,
                        "dt_utc": ta,
                        "horizon": int(h),
                        "y_true": float(y_true[i, 0]),
                        "y_pred": float(y_pred[i, 0]),
                    })

                test_count_h += 1

        # Store run info for this horizon
        per_h_info[f"h{h}"] = {
            "fit_seconds": round(fit_seconds, 2),
            "epochs_run": int(len(hist_df)),
            "test_n": int(test_count_h),
            "model_path": model_path,
        }

    if len(all_pred_rows) == 0:
        raise RuntimeError("No test predictions produced. Check segmentation/NaNs/test coverage.")

    full_results_df = pd.DataFrame(all_pred_rows)

    print("\n=== Calculating Granular Metrics (aligned with LR) ===")

    global_metrics = full_results_df.groupby("horizon").apply(calculate_metrics_df).reset_index()
    global_metrics["Scope"] = "Global"
    global_metrics["Group"] = "All"

    country_metrics = full_results_df.groupby(["horizon", "Country"]).apply(calculate_metrics_df).reset_index()
    country_metrics["Scope"] = "Country"
    country_metrics = country_metrics.rename(columns={"Country": "Group"})

    area_metrics = full_results_df.groupby(["horizon", "StationArea_Label"]).apply(calculate_metrics_df).reset_index()
    area_metrics["Scope"] = "StationArea"
    area_metrics = area_metrics.rename(columns={"StationArea_Label": "Group"})

    final_report = pd.concat([global_metrics, country_metrics, area_metrics], ignore_index=True)
    final_report = final_report[["Scope", "Group", "horizon", "MAE", "RMSE", "R2", "Bias", "Count"]]

    print("\n=== Global Results ===")
    print(final_report[final_report["Scope"] == "Global"].sort_values("horizon"))

    final_report.to_csv(OUTPUT_METRICS_PATH, index=False)
    print(f"\nSaved metrics to: {OUTPUT_METRICS_PATH}")

    pd.DataFrame(pred_sample).to_csv(OUTPUT_PREDS_PATH, index=False)
    print(f"Saved prediction sample to: {OUTPUT_PREDS_PATH}")

    # Save run info
    run_info = {
        "run_seconds": round(time.time() - run_t0, 2),
        "lookback_h": int(LOOKBACK_H),
        "batch_size": int(BATCH_SIZE),
        "max_epochs": int(EPOCHS),
        "learning_rate": float(LR),
        "patience": int(PATIENCE),
        "per_horizon": per_h_info,
    }
    with open(OUTPUT_RUN_INFO_JSON, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    print(f"Run info saved to: {OUTPUT_RUN_INFO_JSON}")

    print(f"\nDone in {time.time() - run_t0:.1f}s")

if __name__ == "__main__":
    main()
