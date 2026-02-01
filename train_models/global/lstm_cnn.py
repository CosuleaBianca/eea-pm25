import os
import math
import json
import time
import gc
import random
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import List, Dict, Optional, Iterator, Tuple

import tensorflow as tf
from tensorflow.keras import layers, Model

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'train_models'))

# =========================================================
# Config (aligned with LR + LSTM reporting)
# =========================================================
CSV_PATH = "ml_ready_dataset_full_realistic.csv"

OUTPUT_METRICS_PATH = "results/lstm_cnn_metrics.csv"
OUTPUT_PREDS_PATH   = "results/lstm_cnn_predictions_sample.csv"
TRAINING_METRICS_DIR = "train_models/history/lstm_cnn_histories"  # per horizon
MODELS_DIR = "models/lstm_cnn_models"                     # per horizon
SCALERS_JSON = "train_models/scalers/lstm_cnn_scalers.json"
RUN_INFO_JSON = "train_models/global/runs_info/lstm_cnn_run_info.json"

TRAIN_END  = pd.Timestamp("2022-12-31 23:00:00", tz="UTC")
TEST_START = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
TEST_END   = pd.Timestamp("2024-12-31 23:00:00", tz="UTC")

# Inner validation window for early stopping (aligned with other models)
VAL_START  = pd.Timestamp("2022-07-01 00:00:00", tz="UTC")
VAL_END    = TRAIN_END

HORIZONS = [1, 3, 6, 12, 24]

LOOKBACK_H = 168
GAP_BREAK_HOURS = 2
MIN_SEG_LEN = LOOKBACK_H + max([1, 3, 6, 12, 24]) + 1  # Aligned with lstm_global.py

BATCH_SIZE = 256     # keep conservative to avoid OOM/Killed
EPOCHS = 50
LR = 5e-4
PATIENCE = 6

PRED_SAMPLE_MAX = 10000

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
os.makedirs(TRAINING_METRICS_DIR, exist_ok=True)
print("All directories created successfully.")

# =========================================================
# GPU memory safety
# =========================================================
def configure_gpu_memory_growth():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[TF] Enabled memory growth on {len(gpus)} GPU(s).")
        else:
            print("[TF] No GPU found. Running on CPU.")
    except Exception as e:
        print("[TF] Could not set memory growth:", e)

configure_gpu_memory_growth()

# =========================================================
# Metrics helpers
# =========================================================
def skill(baseline_mae, model_mae):
    return (baseline_mae - model_mae) / baseline_mae if baseline_mae > 0 else np.nan

class RunningMetrics:
    """
    Streaming metrics accumulator for: MAE, RMSE, R2, Bias
    Also for persistence baseline + skill.
    """
    __slots__ = ("n", "sum_abs", "sum_sq", "sum_err", "sum_y", "sum_y2",
                 "n_b", "sum_abs_b", "sum_sq_b")

    def __init__(self):
        self.n = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_err = 0.0
        self.sum_y = 0.0
        self.sum_y2 = 0.0

        self.n_b = 0
        self.sum_abs_b = 0.0
        self.sum_sq_b = 0.0

    def update(self, y_true: np.ndarray, y_pred: np.ndarray, y_pers: np.ndarray):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_pers = np.asarray(y_pers, dtype=np.float64)

        err = y_pred - y_true
        self.n += y_true.size
        self.sum_abs += float(np.abs(err).sum())
        self.sum_sq += float((err * err).sum())
        self.sum_err += float(err.sum())
        self.sum_y += float(y_true.sum())
        self.sum_y2 += float((y_true * y_true).sum())

        err_b = y_pers - y_true
        self.n_b += y_true.size
        self.sum_abs_b += float(np.abs(err_b).sum())
        self.sum_sq_b += float((err_b * err_b).sum())

    def finalize(self) -> Dict[str, float]:
        if self.n == 0:
            return {
                "Count": 0,
                "MAE": np.nan,
                "RMSE": np.nan,
                "R2": np.nan,
                "Bias": np.nan,
                "MAE_persistence": np.nan,
                "RMSE_persistence": np.nan,
                "skill_vs_persistence_mae": np.nan,
            }

        mae = self.sum_abs / self.n
        rmse = math.sqrt(self.sum_sq / self.n)
        bias = self.sum_err / self.n

        ss_tot = self.sum_y2 - (self.sum_y * self.sum_y) / self.n
        r2 = float("nan") if ss_tot <= 0 else float(1.0 - (self.sum_sq / ss_tot))

        mae_b = self.sum_abs_b / self.n_b
        rmse_b = math.sqrt(self.sum_sq_b / self.n_b)
        skl = skill(mae_b, mae)

        return {
            "Count": int(self.n),
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
            "Bias": float(bias),
            "MAE_persistence": float(mae_b),
            "RMSE_persistence": float(rmse_b),
            "skill_vs_persistence_mae": float(skl),
        }

# =========================================================
# Data helpers
# =========================================================
def make_station_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SiteNumber"] = df["SiteNumber"].astype(str)
    df["station_key"] = df["Country"].astype(str) + "_" + df["SiteNumber"].astype(str)
    return df

def recover_categorical_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    area_cols = [c for c in df.columns if c.startswith("StationArea_")]
    if area_cols:
        df["StationArea_Label"] = df[area_cols].idxmax(axis=1).apply(lambda x: x.replace("StationArea_", ""))
    else:
        df["StationArea_Label"] = "unknown"
    return df

def compute_segments(g: pd.DataFrame, gap_break_hours=2) -> List[pd.DataFrame]:
    g = g.sort_values("dt_utc")
    gaps = g["dt_utc"].diff().dt.total_seconds().div(3600)
    breakpoints = np.where(gaps > gap_break_hours)[0]
    if len(breakpoints) == 0:
        return [g]
    segs, start = [], 0
    for bp in breakpoints:
        segs.append(g.iloc[start:bp].copy())
        start = bp
    segs.append(g.iloc[start:].copy())
    return [s for s in segs if len(s) > 0]

# =========================================================
# Per-station scalers
# =========================================================
@dataclass
class StationScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

def fit_station_scalers(train_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, StationScaler]:
    scalers = {}
    for station_key, g in train_df.groupby("station_key", sort=False):
        arr = g[feature_cols].to_numpy(dtype=np.float32)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        scalers[station_key] = StationScaler(mean=mean, std=std)
    return scalers

def save_scalers_json(scalers: Dict[str, StationScaler], feature_cols: List[str], path: str):
    payload = {
        "feature_cols": feature_cols,
        "scalers": {k: {"mean": v.mean.tolist(), "std": v.std.tolist()} for k, v in scalers.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

def build_multi_horizon_targets_like_lr(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    """
    Build multi-horizon targets using TIME-BASED shifting on complete hourly grids.
    This ensures targets are exactly N hours in the future, even with data gaps.
    Copied from lstm_global.py for alignment.
    """
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

# =========================================================
# Streaming sample generator (NO huge RAM windows)
# =========================================================
def iter_samples_single_h(
    df_build: pd.DataFrame,
    feature_cols: List[str],
    pm25_col: str,
    horizon: int,
    lookback_h: int,
    scalers: Dict[str, StationScaler],
    station_to_idx: Dict[str, int],
    gap_break_hours: int,
    anchor_start: Optional[pd.Timestamp],
    anchor_end: Optional[pd.Timestamp],
    require_no_nan: bool = True,
) -> Iterator[Tuple[Dict[str, np.ndarray], np.float32, np.float32, np.datetime64, str]]:
    pm25_feat_index = feature_cols.index(pm25_col)
    n_features = len(feature_cols)

    for station_key, g in df_build.groupby("station_key", sort=False):
        if station_key not in scalers:
            continue
        scaler = scalers[station_key]
        sid = np.int32(station_to_idx[station_key])

        segs = compute_segments(g, gap_break_hours=gap_break_hours)
        for seg in segs:
            if len(seg) < MIN_SEG_LEN:
                continue

            seg = seg.sort_values("dt_utc").reset_index(drop=True)

            # Check if pre-computed target column exists
            target_col = f"y_h{horizon}"
            if target_col not in seg.columns:
                continue  # Skip if target column not present

            x = seg[feature_cols].to_numpy(dtype=np.float32)
            y_baseline = seg[pm25_col].to_numpy(dtype=np.float32)
            y_target = seg[target_col].to_numpy(dtype=np.float32)  # Pre-computed time-based target
            tstamp = seg["dt_utc"].to_numpy(dtype="datetime64[ns]")

            x_norm = scaler.transform(x)
            # Normalize baseline (current PM2.5)
            y_norm_baseline = (y_baseline - scaler.mean[pm25_feat_index]) / scaler.std[pm25_feat_index]
            # Normalize time-based target
            y_norm_target = (y_target - scaler.mean[pm25_feat_index]) / scaler.std[pm25_feat_index]

            n = len(seg)
            start_t = lookback_h - 1
            end_t = n - horizon

            for t in range(start_t, end_t):
                ts = tstamp[t]
                if anchor_start is not None and ts < np.datetime64(anchor_start.to_datetime64()):
                    continue
                if anchor_end is not None and ts > np.datetime64(anchor_end.to_datetime64()):
                    continue

                x_win = x_norm[t - lookback_h + 1 : t + 1, :]
                if x_win.shape != (lookback_h, n_features):
                    continue
                if require_no_nan and np.isnan(x_win).any():
                    continue

                # Use pre-computed time-based target (not row-based!)
                target = y_norm_target[t]
                base = y_norm_baseline[t]
                if require_no_nan and (np.isnan(target) or np.isnan(base)):
                    continue

                x_dict = {
                    "x_seq": x_win.astype(np.float32),
                    "station_id": sid,
                }
                yield x_dict, np.float32(target), np.float32(base), ts, station_key

def make_stream_ds_single(
    df_build: pd.DataFrame,
    feature_cols: List[str],
    horizon: int,
    scalers: Dict[str, StationScaler],
    station_to_idx: Dict[str, int],
    anchor_start: Optional[pd.Timestamp],
    anchor_end: Optional[pd.Timestamp],
    shuffle: bool,
    batch_size: int,
    seed: int,
):
    n_features = len(feature_cols)

    def gen_xy():
        for x_dict, y_norm, _b, _ts, _sk in iter_samples_single_h(
            df_build=df_build,
            feature_cols=feature_cols,
            pm25_col="PM2.5",
            horizon=horizon,
            lookback_h=LOOKBACK_H,
            scalers=scalers,
            station_to_idx=station_to_idx,
            gap_break_hours=GAP_BREAK_HOURS,
            anchor_start=anchor_start,
            anchor_end=anchor_end,
            require_no_nan=True,
        ):
            yield x_dict, y_norm

    output_signature = (
        {
            "x_seq": tf.TensorSpec(shape=(LOOKBACK_H, n_features), dtype=tf.float32),
            "station_id": tf.TensorSpec(shape=(), dtype=tf.int32),
        },
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(gen_xy, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=50000, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(2)
    return ds

def count_samples_stream(
    df_build: pd.DataFrame,
    feature_cols: List[str],
    horizon: int,
    scalers: Dict[str, StationScaler],
    station_to_idx: Dict[str, int],
    anchor_start: Optional[pd.Timestamp],
    anchor_end: Optional[pd.Timestamp],
) -> int:
    c = 0
    for _x, _y, _b, _ts, _sk in iter_samples_single_h(
        df_build=df_build,
        feature_cols=feature_cols,
        pm25_col="PM2.5",
        horizon=horizon,
        lookback_h=LOOKBACK_H,
        scalers=scalers,
        station_to_idx=station_to_idx,
        gap_break_hours=GAP_BREAK_HOURS,
        anchor_start=anchor_start,
        anchor_end=anchor_end,
        require_no_nan=True,
    ):
        c += 1
    return c

# =========================================================
# Model: CNN+LSTM SINGLE-OUTPUT with residual on PM2.5(t)
# =========================================================
def build_cnn_lstm_model_single(
    n_features: int,
    n_stations: int,
    pm25_index: int
) -> Model:
    seq_in = layers.Input(shape=(LOOKBACK_H, n_features), name="x_seq")
    sid_in = layers.Input(shape=(), dtype="int32", name="station_id")

    pm25_t = layers.Lambda(lambda z: z[:, -1, pm25_index:pm25_index+1], name="pm25_t")(seq_in)  # [B,1]

    emb_dim = 16
    sid_emb = layers.Embedding(input_dim=n_stations, output_dim=emb_dim, name="station_emb")(sid_in)
    sid_emb = layers.Dense(emb_dim, activation="relu",
                           kernel_regularizer=tf.keras.regularizers.l2(1e-6))(sid_emb)
    sid_rep = layers.RepeatVector(LOOKBACK_H)(sid_emb)

    x = layers.Concatenate(axis=-1)([seq_in, sid_rep])

    x = layers.Conv1D(96, kernel_size=5, padding="causal")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Conv1D(96, kernel_size=5, padding="causal")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.15)(x)

    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.LSTM(64, return_sequences=True, dropout=0.15)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.LSTM(32, return_sequences=False, dropout=0.15)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Dense(96, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-6))(x)
    x = layers.Dropout(0.20)(x)

    h = layers.Dense(32, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-6))(x)
    h = layers.Dropout(0.10)(h)
    delta = layers.Dense(1, name="delta")(h)

    out = layers.Add(name="y_hat")([pm25_t, delta])

    model = Model(inputs=[seq_in, sid_in], outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=1.0),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model

# =========================================================
# Streaming evaluation: Global + per Country + per StationArea
# =========================================================
def eval_test_stream_grouped(
    model: tf.keras.Model,
    test_build_df: pd.DataFrame,
    feature_cols: List[str],
    horizon: int,
    scalers: Dict[str, StationScaler],
    station_to_idx: Dict[str, int],
    pm25_mean: np.ndarray,
    pm25_std: np.ndarray,
    country_map: Dict[str, str],
    area_map: Dict[str, str],
    site_map: Dict[str, str],
    sample_max: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      metrics_df: rows with Scope/Group/horizon + MAE/RMSE/R2/Bias/Count + persistence + skill
      preds_sample_df: up to sample_max rows with y_true/y_pred/y_persistence + meta
    """
    global_acc = RunningMetrics()
    country_acc: Dict[str, RunningMetrics] = {}
    area_acc: Dict[str, RunningMetrics] = {}

    sample_rows = []

    def batch_iter(it, batch_size):
        batch = []
        for item in it:
            batch.append(item)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    gen_full = iter_samples_single_h(
        df_build=test_build_df,
        feature_cols=feature_cols,
        pm25_col="PM2.5",
        horizon=horizon,
        lookback_h=LOOKBACK_H,
        scalers=scalers,
        station_to_idx=station_to_idx,
        gap_break_hours=GAP_BREAK_HOURS,
        anchor_start=TEST_START,
        anchor_end=TEST_END,
        require_no_nan=True,
    )

    for batch in batch_iter(gen_full, BATCH_SIZE):
        x_seq = np.stack([b[0]["x_seq"] for b in batch], axis=0).astype(np.float32)
        sid = np.array([b[0]["station_id"] for b in batch], dtype=np.int32)

        y_true_norm = np.array([b[1] for b in batch], dtype=np.float32)
        y_base_norm = np.array([b[2] for b in batch], dtype=np.float32)
        ts_arr = np.array([b[3] for b in batch], dtype="datetime64[ns]")
        sk_arr = np.array([b[4] for b in batch], dtype=object)

        y_pred_norm = model.predict({"x_seq": x_seq, "station_id": sid}, verbose=0).reshape(-1).astype(np.float32)

        std_s = pm25_std[sid]
        mean_s = pm25_mean[sid]

        y_true = y_true_norm * std_s + mean_s
        y_pred = y_pred_norm * std_s + mean_s
        y_pers = y_base_norm * std_s + mean_s

        # update global
        global_acc.update(y_true, y_pred, y_pers)

        # update grouped
        for i in range(len(y_true)):
            sk = str(sk_arr[i])
            ctry = country_map.get(sk, "unknown")
            area = area_map.get(sk, "unknown")

            if ctry not in country_acc:
                country_acc[ctry] = RunningMetrics()
            if area not in area_acc:
                area_acc[area] = RunningMetrics()

            # one-by-one update (cheap enough; batches are small)
            country_acc[ctry].update(y_true[i:i+1], y_pred[i:i+1], y_pers[i:i+1])
            area_acc[area].update(y_true[i:i+1], y_pred[i:i+1], y_pers[i:i+1])

        # collect sample rows
        if len(sample_rows) < sample_max:
            take = min(sample_max - len(sample_rows), len(y_true))
            for i in range(take):
                sk = str(sk_arr[i])
                sample_rows.append({
                    "Country": country_map.get(sk, "unknown"),
                    "SiteNumber": site_map.get(sk, "unknown"),
                    "StationArea_Label": area_map.get(sk, "unknown"),
                    "dt_utc": np.datetime_as_string(ts_arr[i], unit="s"),
                    "horizon": int(horizon),
                    "y_true": float(y_true[i]),
                    "y_pred": float(y_pred[i]),
                    "y_persistence": float(y_pers[i]),
                })

    # Build metrics rows
    rows = []

    g = global_acc.finalize()
    rows.append({
        "Scope": "Global",
        "Group": "All",
        "horizon": int(horizon),
        **g
    })

    for ctry, acc in sorted(country_acc.items(), key=lambda x: x[0]):
        m = acc.finalize()
        rows.append({
            "Scope": "Country",
            "Group": ctry,
            "horizon": int(horizon),
            **m
        })

    for area, acc in sorted(area_acc.items(), key=lambda x: x[0]):
        m = acc.finalize()
        rows.append({
            "Scope": "StationArea",
            "Group": area,
            "horizon": int(horizon),
            **m
        })

    metrics_df = pd.DataFrame(rows)
    preds_sample_df = pd.DataFrame(sample_rows)
    return metrics_df, preds_sample_df

# =========================================================
# Main
# =========================================================
def main():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True, errors="coerce")

    if "season" in df.columns and df["season"].dtype == object:
        season_map = {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3, "Fall": 3}
        df["season"] = df["season"].map(season_map).astype("float32")

    df = make_station_id(df)
    df = recover_categorical_metadata(df)

    df = df.sort_values(["Country", "SiteNumber", "dt_utc"]).drop_duplicates(
        subset=["Country", "SiteNumber", "dt_utc"], keep="first"
    )

    # Build time-based multi-horizon targets (aligned with lstm_global.py)
    print("Building time-based multi-horizon targets...")
    df = build_multi_horizon_targets_like_lr(df, HORIZONS)
    print(f"Created target columns: {[f'y_h{h}' for h in HORIZONS]}")

    features_from_file = [
        "Latitude", "Longitude", "Altitude",
        "StationType_background", "StationType_industrial", "StationType_traffic",
        "StationArea_rural", "StationArea_rural-nearcity", "StationArea_suburban", "StationArea_urban",
        "temperature_2m", "relative_humidity_2m", "dew_point_2m", "wind_u", "wind_v",
        "precipitation", "surface_pressure",
        "PM2.5", "NO2", "PM10",
        "hour", "day_of_week", "day_of_month", "month", "year", "is_weekend",
        "season", "hour_sin", "hour_cos", "month_sin", "month_cos",
        "NO2_lag_12h", "NO2_lag_168h", "NO2_lag_1h", "NO2_lag_24h",
        "NO2_lag_2h", "NO2_lag_3h", "NO2_lag_6h",
        "PM10_lag_12h", "PM10_lag_168h", "PM10_lag_1h", "PM10_lag_24h",
        "PM10_lag_2h", "PM10_lag_3h", "PM10_lag_6h",
        "PM2.5_lag_12h", "PM2.5_lag_168h", "PM2.5_lag_1h", "PM2.5_lag_24h",
        "PM2.5_lag_2h", "PM2.5_lag_3h", "PM2.5_lag_6h",
        "NO2_rolling_mean_12h", "NO2_rolling_mean_24h", "NO2_rolling_mean_3h", "NO2_rolling_mean_6h",
        "PM10_rolling_mean_12h", "PM10_rolling_mean_24h", "PM10_rolling_mean_3h", "PM10_rolling_mean_6h",
        "PM2.5_rolling_mean_12h", "PM2.5_rolling_mean_24h", "PM2.5_rolling_mean_3h", "PM2.5_rolling_mean_6h",
        "NO2_rolling_std_12h", "NO2_rolling_std_24h", "NO2_rolling_std_3h", "NO2_rolling_std_6h",
        "PM10_rolling_std_12h", "PM10_rolling_std_24h", "PM10_rolling_std_3h", "PM10_rolling_std_6h",
        "PM2.5_rolling_std_12h", "PM2.5_rolling_std_24h", "PM2.5_rolling_std_3h", "PM2.5_rolling_std_6h",
    ]

    non_feature_cols = {"Country", "SiteNumber", "dt_utc", "dt_local", "Start", "PM2.5_next"}
    feature_cols = [c for c in features_from_file if c in df.columns and c not in non_feature_cols]
    if "PM2.5" not in feature_cols:
        raise ValueError("PM2.5 must be present in feature_cols.")
    pm25_idx = feature_cols.index("PM2.5")

    df_model = df.dropna(subset=feature_cols).copy()
    df_model["dt_utc"] = pd.to_datetime(df_model["dt_utc"], utc=True)

    train_base = df_model[df_model["dt_utc"] <= TRAIN_END].copy()
    test_base  = df_model[(df_model["dt_utc"] >= TEST_START) & (df_model["dt_utc"] <= TEST_END)].copy()
    print(f"Base Train samples (features valid): {len(train_base):,} | Base Test samples: {len(test_base):,}")

    train_st = set(train_base["station_key"].unique())
    test_st  = set(test_base["station_key"].unique())
    common_st = sorted(list(train_st & test_st))
    if not common_st:
        raise RuntimeError("No common stations between train and test splits.")

    WARMUP = pd.Timedelta(hours=LOOKBACK_H - 1)

    train_build_df = df_model[df_model["dt_utc"] <= TRAIN_END].copy()
    train_build_df = train_build_df[train_build_df["station_key"].isin(common_st)]

    val_build_df = df_model[(df_model["dt_utc"] >= (VAL_START - WARMUP)) & (df_model["dt_utc"] <= VAL_END)].copy()
    val_build_df = val_build_df[val_build_df["station_key"].isin(common_st)]

    test_build_df = df_model[(df_model["dt_utc"] >= (TEST_START - WARMUP)) & (df_model["dt_utc"] <= TEST_END)].copy()
    test_build_df = test_build_df[test_build_df["station_key"].isin(common_st)]

    station_to_idx = {k: i for i, k in enumerate(common_st)}
    n_stations = len(common_st)

    # =====================================================
    # Fit per-station scalers (prefer before val window, fallback <=TRAIN_END)
    # =====================================================
    print("Fitting per-station scalers (prefer before val window, fallback <=TRAIN_END)...")
    # Use data before validation window for fitting scalers (more conservative, aligned with lstm_global_attention)
    train_scaler_df = df_model[df_model["dt_utc"] < VAL_START].copy()
    train_scaler_df = train_scaler_df[train_scaler_df["station_key"].isin(common_st)]

    scalers = fit_station_scalers(train_scaler_df, feature_cols)

    missing = [sk for sk in common_st if sk not in scalers]
    if missing:
        print(f"[WARN] {len(missing)} stations have no scaler before val window. Using fallback <=TRAIN_END for them.")
        fallback_df = train_build_df[train_build_df["station_key"].isin(missing)].copy()
        fallback_scalers = fit_station_scalers(fallback_df, feature_cols)
        scalers.update(fallback_scalers)

    still_missing = [sk for sk in common_st if sk not in scalers]
    if still_missing:
        raise RuntimeError(f"Still missing scalers for {len(still_missing)} stations: e.g. {still_missing[:5]}")

    save_scalers_json(scalers, feature_cols, SCALERS_JSON)
    print(f"Saved scalers: {SCALERS_JSON} | stations: {len(scalers)} (expected {len(common_st)})")

    # Precompute pm25 mean/std per station idx
    idx_to_station = {i: k for k, i in station_to_idx.items()}
    pm25_mean = np.zeros(n_stations, dtype=np.float32)
    pm25_std  = np.ones(n_stations, dtype=np.float32)
    for i in range(n_stations):
        sk = idx_to_station[i]
        pm25_mean[i] = scalers[sk].mean[pm25_idx]
        pm25_std[i]  = scalers[sk].std[pm25_idx]

    # Metadata maps
    station_meta = (
        df_model.loc[df_model["station_key"].isin(common_st), ["station_key", "Country", "SiteNumber", "StationArea_Label"]]
        .drop_duplicates("station_key")
        .set_index("station_key")
    )
    country_map = station_meta["Country"].to_dict()
    area_map    = station_meta["StationArea_Label"].to_dict()
    site_map    = station_meta["SiteNumber"].to_dict()

    # =====================================================
    # Train separate model per horizon + evaluate grouped metrics
    # =====================================================
    script_t0 = time.time()
    all_metrics = []
    all_pred_samples = []
    # Updated run_info format (aligned with other models)
    run_info = {
        "run_seconds": 0.0,  # Will be filled at the end
        "lookback_h": LOOKBACK_H,
        "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS,
        "learning_rate": LR,
        "patience": PATIENCE,
        "per_horizon": {}
    }

    for h in HORIZONS:
        print("\n" + "="*70)
        print(f"Training CNN+LSTM SINGLE-OUTPUT for horizon h={h}h")
        t0 = time.time()

        print("Counting samples (streaming)...")
        n_train = count_samples_stream(
            df_build=train_build_df,
            feature_cols=feature_cols,
            horizon=h,
            scalers=scalers,
            station_to_idx=station_to_idx,
            anchor_start=None,
            anchor_end=TRAIN_END,
        )
        n_val = count_samples_stream(
            df_build=val_build_df,
            feature_cols=feature_cols,
            horizon=h,
            scalers=scalers,
            station_to_idx=station_to_idx,
            anchor_start=VAL_START,
            anchor_end=VAL_END,
        )
        print(f"  Train samples: {n_train:,} | Val samples: {n_val:,}")

        if n_train == 0 or n_val == 0:
            print(f"Skipping h={h}: insufficient train/val samples.")
            continue

        steps_per_epoch = math.ceil(n_train / BATCH_SIZE)
        val_steps = max(1, math.ceil(n_val / BATCH_SIZE))

        train_ds = make_stream_ds_single(
            df_build=train_build_df,
            feature_cols=feature_cols,
            horizon=h,
            scalers=scalers,
            station_to_idx=station_to_idx,
            anchor_start=None,
            anchor_end=TRAIN_END,
            shuffle=True,
            batch_size=BATCH_SIZE,
            seed=SEED + h,
        )
        val_ds = make_stream_ds_single(
            df_build=val_build_df,
            feature_cols=feature_cols,
            horizon=h,
            scalers=scalers,
            station_to_idx=station_to_idx,
            anchor_start=VAL_START,
            anchor_end=VAL_END,
            shuffle=False,
            batch_size=BATCH_SIZE,
            seed=SEED + h,
        )

        model = build_cnn_lstm_model_single(
            n_features=len(feature_cols),
            n_stations=n_stations,
            pm25_index=pm25_idx
        )

        class LrLogger(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                print(f"Epoch {epoch + 1}: lr={lr:.6g}")

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
            LrLogger(),
        ]

        history = model.fit(
            train_ds.repeat(),
            validation_data=val_ds,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1
        )

        # Save training curve
        hist_df = pd.DataFrame(history.history)
        hist_df.insert(0, "epoch", np.arange(1, len(hist_df) + 1))
        hist_path = os.path.join(TRAINING_METRICS_DIR, f"train_curve_h{h}.csv")
        hist_df.to_csv(hist_path, index=False)

        # Save model
        model_path = os.path.join(MODELS_DIR, f"lstm_cnn_h{h}.keras")
        model.save(model_path)

        # Evaluate on test (streaming, grouped)
        print("Evaluating on TEST (streaming, grouped metrics)...")
        metrics_h_df, preds_h_sample = eval_test_stream_grouped(
            model=model,
            test_build_df=test_build_df,
            feature_cols=feature_cols,
            horizon=h,
            scalers=scalers,
            station_to_idx=station_to_idx,
            pm25_mean=pm25_mean,
            pm25_std=pm25_std,
            country_map=country_map,
            area_map=area_map,
            site_map=site_map,
            sample_max=max(1, PRED_SAMPLE_MAX // max(1, len(HORIZONS))),
        )

        all_metrics.append(metrics_h_df)
        if not preds_h_sample.empty:
            all_pred_samples.append(preds_h_sample)

        # Print quick global line
        g = metrics_h_df[(metrics_h_df["Scope"] == "Global")].iloc[0].to_dict()
        elapsed = time.time() - t0
        print(f"  > Done h={h} in {elapsed:.1f}s | Test Count={int(g['Count']):,}")
        print(f"  > Global MAE={g['MAE']:.4f} RMSE={g['RMSE']:.4f} R2={g['R2']:.4f} Bias={g['Bias']:.4f}")
        print(f"  > Persistence MAE={g['MAE_persistence']:.4f} skill(MAE)={g['skill_vs_persistence_mae']:.4f}")

        # Track run info for this horizon (aligned format)
        run_info["per_horizon"][f"h{h}"] = {
            "train_seconds": round(elapsed, 2),
            "train_n": int(n_train),
            "val_n": int(n_val),
            "test_n": int(g['Count']),
            "model_path": model_path,
            "history_path": hist_path,
        }

        # Cleanup between horizons
        del model, history, train_ds, val_ds
        gc.collect()
        tf.keras.backend.clear_session()

    # =====================================================
    # Save final outputs
    # =====================================================
    if not all_metrics:
        print("No results generated!")
        return

    final_metrics = pd.concat(all_metrics, axis=0, ignore_index=True)

    # Order columns exactly LR-like + baseline extras
    cols_order = [
        "Scope", "Group", "horizon",
        "MAE", "RMSE", "R2", "Bias", "Count",
        "MAE_persistence", "RMSE_persistence", "skill_vs_persistence_mae"
    ]
    final_metrics = final_metrics[cols_order].sort_values(["Scope", "Group", "horizon"]).reset_index(drop=True)

    final_metrics.to_csv(OUTPUT_METRICS_PATH, index=False)
    print(f"\nDetailed metrics saved to: {OUTPUT_METRICS_PATH}")

    print("\n=== Sample of Results (Global) ===")
    print(final_metrics[final_metrics["Scope"] == "Global"].sort_values("horizon"))

    if all_pred_samples:
        preds_df = pd.concat(all_pred_samples, axis=0, ignore_index=True).head(PRED_SAMPLE_MAX)
        preds_df.to_csv(OUTPUT_PREDS_PATH, index=False)
        print(f"\nPrediction sample saved to: {OUTPUT_PREDS_PATH} ({len(preds_df)} rows)")
    else:
        print("\nNo prediction sample collected (unexpected).")

    # Save run info (update total run_seconds)
    run_info["run_seconds"] = round(time.time() - script_t0, 2)
    with open(RUN_INFO_JSON, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    print(f"\nRun info saved to: {RUN_INFO_JSON}")

if __name__ == "__main__":
    main()
