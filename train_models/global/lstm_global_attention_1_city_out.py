import os
import json
import math
import time
import gc
import csv
import random
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Iterator

import tensorflow as tf
from tensorflow.keras import layers, Model

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'train_models'))

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "ml_ready_dataset_full_realistic.csv"

# LEAVE-ONE-CITY-OUT: Country to exclude from training
EXCLUDE_COUNTRY = "FR"  # Change this to test different cities

TRAIN_END = pd.Timestamp("2022-12-31 23:00:00", tz="UTC")
TEST_START = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
TEST_END   = pd.Timestamp("2024-12-31 23:00:00", tz="UTC")
VAL_START = pd.Timestamp("2022-07-01 00:00:00", tz="UTC")
VAL_END   = TRAIN_END

LOOKBACK_H = 168
HORIZONS = [1, 3, 6, 12, 24]
GAP_BREAK_HOURS = 2
MIN_SEG_LEN = LOOKBACK_H + max(HORIZONS) + 1

BATCH_SIZE = 256
EPOCHS = 50
LR = 5e-4
PATIENCE = 6
SHUFFLE_BUFFER = 50000

# Output paths include excluded country
METRICS_CSV = f"results/lstm_attention_metrics_exclude_{EXCLUDE_COUNTRY}.csv"
PRED_SAMPLE_CSV = f"results/lstm_attention_predictions_sample_exclude_{EXCLUDE_COUNTRY}.csv"
TRAINING_HISTORY_DIR = f"train_models/history/lstm_attention_histories_exclude_{EXCLUDE_COUNTRY}"
MODELS_DIR = f"models/lstm_attention_models_exclude_{EXCLUDE_COUNTRY}"
SCALERS_JSON = f"train_models/scalers/lstm_attention_scalers_exclude_{EXCLUDE_COUNTRY}.json"
RUN_INFO_JSON = f"train_models/global/runs_info/lstm_attention_run_info_exclude_{EXCLUDE_COUNTRY}.json"

# Create all output directories upfront
print("Creating output directories...")
os.makedirs("train_models/global/runs_info", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SCALERS_JSON), exist_ok=True)
os.makedirs(TRAINING_HISTORY_DIR, exist_ok=True)
print("All directories created successfully.")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# GPU config
try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass


# ============================================================
# METRICS
# ============================================================
def mae_np(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse_np(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def r2_score_np(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

def bias_np(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.mean(y_pred - y_true))


class RunningRegressionMetrics:
    def __init__(self):
        self.n = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_err = 0.0
        self.sum_y = 0.0
        self.sum_y2 = 0.0
        self.ssr = 0.0

    def update(self, y_true: np.ndarray, y_pred: np.ndarray):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        err = y_pred - y_true
        self.n += y_true.size
        self.sum_abs += float(np.sum(np.abs(err)))
        self.sum_sq += float(np.sum(err**2))
        self.sum_err += float(np.sum(err))
        self.sum_y += float(np.sum(y_true))
        self.sum_y2 += float(np.sum(y_true**2))
        self.ssr += float(np.sum((y_true - y_pred)**2))

    def finalize(self):
        if self.n == 0:
            return dict(MAE=np.nan, RMSE=np.nan, R2=np.nan, Bias=np.nan, Count=0)
        mae = self.sum_abs / self.n
        rmse = math.sqrt(self.sum_sq / self.n)
        bias = self.sum_err / self.n
        y_mean = self.sum_y / self.n
        ss_tot = self.sum_y2 - self.n * (y_mean**2)
        r2 = 1.0 - (self.ssr / ss_tot) if ss_tot > 0 else float("nan")
        return dict(MAE=mae, RMSE=rmse, R2=r2, Bias=bias, Count=int(self.n))


class RunningGroupedMetrics:
    def __init__(self):
        self.m = {}

    def update(self, keys: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
        keys = np.asarray(keys)
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        err = y_pred - y_true
        for k, yt, yp, e in zip(keys, y_true, y_pred, err):
            kk = str(k)
            s = self.m.get(kk)
            if s is None:
                s = {
                    "n": 0,
                    "sum_abs": 0.0,
                    "sum_sq": 0.0,
                    "sum_err": 0.0,
                    "sum_y": 0.0,
                    "sum_y2": 0.0,
                    "ssr": 0.0,
                }
                self.m[kk] = s
            s["n"] += 1
            s["sum_abs"] += abs(e)
            s["sum_sq"] += e * e
            s["sum_err"] += e
            s["sum_y"] += yt
            s["sum_y2"] += yt * yt
            s["ssr"] += (yt - yp) * (yt - yp)

    def finalize_rows(self, scope: str, horizon: int) -> List[Dict]:
        rows = []
        for group, s in self.m.items():
            n = s["n"]
            if n <= 0:
                continue
            mae = s["sum_abs"] / n
            rmse = math.sqrt(s["sum_sq"] / n)
            bias = s["sum_err"] / n
            y_mean = s["sum_y"] / n
            ss_tot = s["sum_y2"] - n * (y_mean**2)
            r2 = 1.0 - (s["ssr"] / ss_tot) if ss_tot > 0 else float("nan")
            rows.append({
                "Scope": scope,
                "Group": group,
                "horizon": horizon,
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R2": float(r2),
                "Bias": float(bias),
                "Count": float(n),
            })
        return rows


# ============================================================
# HELPERS (dataset)
# ============================================================
def make_station_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SiteNumber"] = df["SiteNumber"].astype(str)
    df["station_key"] = df["Country"].astype(str) + "_" + df["SiteNumber"].astype(str)
    return df

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

@dataclass
class StationScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

def fit_station_scalers(train_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, StationScaler]:
    scalers: Dict[str, StationScaler] = {}
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


# ============================================================
# STREAMING WINDOW ITERATOR
# ============================================================
def iter_examples_for_segment_single_h(
    seg: pd.DataFrame,
    station_idx: int,
    station_key: str,
    feature_cols: List[str],
    pm25_col: str,
    lookback_h: int,
    horizon: int,
    scaler: StationScaler,
    require_no_nan: bool = True,
    anchor_start: Optional[pd.Timestamp] = None,
    anchor_end: Optional[pd.Timestamp] = None,
):
    seg = seg.sort_values("dt_utc").reset_index(drop=True)

    target_col = f"y_h{horizon}"
    if target_col not in seg.columns:
        return

    x = seg[feature_cols].to_numpy(dtype=np.float32)
    y_raw = seg[pm25_col].to_numpy(dtype=np.float32)
    y_target_raw = seg[target_col].to_numpy(dtype=np.float32)
    tstamp = seg["dt_utc"].to_numpy(dtype="datetime64[ns]")

    x_norm = scaler.transform(x)

    pm25_feat_index = feature_cols.index(pm25_col)
    y_norm_baseline = (y_raw - scaler.mean[pm25_feat_index]) / scaler.std[pm25_feat_index]
    y_norm_target = (y_target_raw - scaler.mean[pm25_feat_index]) / scaler.std[pm25_feat_index]

    n = len(seg)
    start_t = lookback_h - 1
    end_t = n - horizon

    astart64 = np.datetime64(anchor_start.to_datetime64()) if anchor_start is not None else None
    aend64   = np.datetime64(anchor_end.to_datetime64()) if anchor_end is not None else None

    for t in range(start_t, end_t):
        ts = tstamp[t]
        if astart64 is not None and ts < astart64:
            continue
        if aend64 is not None and ts > aend64:
            continue

        x_win = x_norm[t - lookback_h + 1 : t + 1, :]
        if x_win.shape[0] != lookback_h:
            continue
        if require_no_nan and np.isnan(x_win).any():
            continue

        target = y_norm_target[t]
        base = y_norm_baseline[t]
        if require_no_nan and (np.isnan(target) or np.isnan(base)):
            continue

        ts_int = np.int64(ts.astype("datetime64[ns]").astype(np.int64))
        yield x_win.astype(np.float32), np.int32(station_idx), np.float32(target), np.float32(base), ts_int


def make_tf_dataset_streaming(
    df_split: pd.DataFrame,
    scalers: Dict[str, StationScaler],
    station_to_idx: Dict[str, int],
    feature_cols: List[str],
    horizon: int,
    lookback_h: int,
    shuffle: bool,
    batch_size: int,
    anchor_start: Optional[pd.Timestamp] = None,
    anchor_end: Optional[pd.Timestamp] = None,
) -> tf.data.Dataset:
    n_features = len(feature_cols)

    def gen():
        for station_key, g in df_split.groupby("station_key", sort=False):
            if station_key not in scalers:
                continue
            station_idx = station_to_idx[station_key]
            for seg in compute_segments(g, gap_break_hours=GAP_BREAK_HOURS):
                if len(seg) < MIN_SEG_LEN:
                    continue
                for x_win, sid, y, b, ts_int in iter_examples_for_segment_single_h(
                    seg=seg,
                    station_idx=station_idx,
                    station_key=station_key,
                    feature_cols=feature_cols,
                    pm25_col="PM2.5",
                    lookback_h=lookback_h,
                    horizon=horizon,
                    scaler=scalers[station_key],
                    require_no_nan=True,
                    anchor_start=anchor_start,
                    anchor_end=anchor_end,
                ):
                    yield {"x_seq": x_win, "station_id": sid}, np.array([y], dtype=np.float32)

    output_signature = (
        {
            "x_seq": tf.TensorSpec(shape=(lookback_h, n_features), dtype=tf.float32),
            "station_id": tf.TensorSpec(shape=(), dtype=tf.int32),
        },
        tf.TensorSpec(shape=(1,), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ============================================================
# TEST STREAMING DATASET WITH METADATA
# ============================================================
def make_test_dataset_with_meta(
    test_build_df: pd.DataFrame,
    scalers: Dict[str, StationScaler],
    station_to_idx: Dict[str, int],
    feature_cols: List[str],
    horizon: int,
    lookback_h: int,
    anchor_start: pd.Timestamp,
    anchor_end: pd.Timestamp,
    batch_size: int,
) -> tf.data.Dataset:
    n_features = len(feature_cols)

    def gen():
        for station_key, g in test_build_df.groupby("station_key", sort=False):
            if station_key not in scalers:
                continue
            station_idx = station_to_idx[station_key]

            gmap = g[["dt_utc", "Country", "SiteNumber", "StationArea_Label"]].drop_duplicates("dt_utc").set_index("dt_utc")

            for seg in compute_segments(g, gap_break_hours=GAP_BREAK_HOURS):
                if len(seg) < MIN_SEG_LEN:
                    continue

                for x_win, sid, y, b, ts_int in iter_examples_for_segment_single_h(
                    seg=seg,
                    station_idx=station_idx,
                    station_key=station_key,
                    feature_cols=feature_cols,
                    pm25_col="PM2.5",
                    lookback_h=lookback_h,
                    horizon=horizon,
                    scaler=scalers[station_key],
                    require_no_nan=True,
                    anchor_start=anchor_start,
                    anchor_end=anchor_end,
                ):
                    ts = pd.to_datetime(int(ts_int), utc=True)
                    if ts in gmap.index:
                        row = gmap.loc[ts]
                        country = str(row["Country"])
                        site = str(row["SiteNumber"])
                        area = str(row["StationArea_Label"])
                    else:
                        country = station_key.split("_")[0]
                        site = station_key.split("_")[1] if "_" in station_key else "NA"
                        area = "Unknown"

                    yield (
                        x_win,
                        np.int32(sid),
                        np.float32(y),
                        np.float32(b),
                        np.int64(ts_int),
                        country.encode("utf-8"),
                        site.encode("utf-8"),
                        area.encode("utf-8"),
                    )

    output_signature = (
        tf.TensorSpec(shape=(lookback_h, n_features), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64),
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.string),
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ============================================================
# MODEL
# ============================================================
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
        self.ssr.assign(0.0)
        self.sum_y.assign(0.0)
        self.sum_y_sq.assign(0.0)
        self.sum_w.assign(0.0)

def build_model_residual_attention_single(
    n_features: int,
    n_stations: int,
    pm25_index: int,
) -> Model:
    seq_in = layers.Input(shape=(LOOKBACK_H, n_features), name="x_seq")
    sid_in = layers.Input(shape=(), dtype="int32", name="station_id")

    pm25_t = layers.Lambda(lambda z: z[:, -1, pm25_index:pm25_index + 1], name="pm25_t")(seq_in)

    emb_dim = 8
    sid_emb = layers.Embedding(input_dim=n_stations, output_dim=emb_dim, name="station_emb")(sid_in)
    sid_emb = layers.Dense(emb_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-6))(sid_emb)
    sid_rep = layers.RepeatVector(LOOKBACK_H)(sid_emb)

    x_in = layers.Concatenate(axis=-1)([seq_in, sid_rep])

    x = layers.LSTM(
        32, return_sequences=True,
        dropout=0.3, recurrent_dropout=0.0,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x_in)

    lstm_out = layers.LSTM(
        16, return_sequences=True,
        dropout=0.3, recurrent_dropout=0.0,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)

    att_score = layers.Dense(1, activation="tanh", name="att_score_dense")(lstm_out)
    att_score = layers.Flatten()(att_score)
    att_weights = layers.Activation("softmax", name="attention_weights")(att_score)
    context_vector = layers.Dot(axes=1)([att_weights, lstm_out])

    last_lstm_state = layers.Lambda(lambda z: z[:, -1, :], name="last_lstm_step")(lstm_out)
    raw_last_features = layers.Lambda(lambda z: z[:, -1, :], name="raw_features_t")(seq_in)

    x = layers.Concatenate(name="concat_all_info")([context_vector, last_lstm_state, raw_last_features])

    x = layers.Dense(
        64, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(
        32, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="head_dense"
    )(x)
    x = layers.Dropout(0.3)(x)
    delta = layers.Dense(1, name="delta")(x)

    y_hat = layers.Add(name="y_hat")([pm25_t, delta])

    model = Model(inputs=[seq_in, sid_in], outputs=y_hat)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"), R2Metric(name="r2")]
    )
    return model


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print(">>> [LSTM Attention - Leave-One-City-Out] Loading CSV...")
    print(f">>> EXCLUDED COUNTRY FROM TRAINING: {EXCLUDE_COUNTRY}")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True, errors="coerce")

    if "season" in df.columns:
        season_map = {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3, "Fall": 3}
        if df["season"].dtype == object:
            df["season"] = df["season"].map(season_map)
        df["season"] = df["season"].astype("float32")

    df = make_station_id(df)
    df = df.sort_values(["station_key", "dt_utc"]).drop_duplicates(["station_key", "dt_utc"], keep="first")
    df = recover_station_area_label(df)

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
    feature_cols = [c for c in features_from_file if (c in df.columns and c not in non_feature_cols)]

    if "PM2.5" not in feature_cols:
        raise ValueError("PM2.5 must be present in feature_cols.")
    pm25_idx = feature_cols.index("PM2.5")

    print(f">>> Features used: {len(feature_cols)}")
    print(f">>> Horizons: {HORIZONS}")

    print(">>> Building time-based multi-horizon targets...")
    df = build_multi_horizon_targets_like_lr(df, HORIZONS)

    # Split data
    train_df_full = df[df["dt_utc"] <= TRAIN_END].copy()
    test_df  = df[(df["dt_utc"] >= TEST_START) & (df["dt_utc"] <= TEST_END)].copy()

    # =========================
    # LEAVE-ONE-CITY-OUT: Exclude country from training
    # =========================
    train_before = len(train_df_full)
    train_df = train_df_full[train_df_full["Country"] != EXCLUDE_COUNTRY].copy()
    excluded_samples = train_before - len(train_df)

    print("\n" + "="*70)
    print(f">>> LEAVE-ONE-CITY-OUT: Excluded {EXCLUDE_COUNTRY} from training")
    print(f">>> Training samples removed: {excluded_samples:,} ({100*excluded_samples/train_before:.1f}%)")
    print(f">>> Training samples remaining: {len(train_df):,}")
    print(">>> Test data unchanged (includes all cities)")
    print("="*70 + "\n")

    test_excluded = len(test_df[test_df["Country"] == EXCLUDE_COUNTRY])
    test_total = len(test_df)
    print(f">>> Test samples from {EXCLUDE_COUNTRY}: {test_excluded:,} ({100*test_excluded/test_total:.1f}%)")
    print(f">>> Test samples from other cities: {test_total - test_excluded:,}\n")

    # For leave-one-city-out: we want to test on ALL stations (including excluded country)
    # Strategy:
    # 1. Fit scalers on ALL training data (before exclusion) - this is realistic as we have historical data
    # 2. Train model only on non-excluded country data
    # 3. Test on ALL stations (excluded country will use its scalers but untrained embeddings)
    train_st = set(train_df["station_key"].unique())
    test_st  = set(test_df["station_key"].unique())
    train_st_full = set(train_df_full["station_key"].unique())  # Before exclusion

    # Use all test stations for evaluation (including excluded country)
    all_test_st = sorted(list(test_st))

    print(f">>> Stations before exclusion: {len(train_st_full)}")
    print(f">>> Stations after excluding {EXCLUDE_COUNTRY}: {len(train_st)}")
    print(f">>> Stations in test set (all): {len(test_st)}")

    # Keep all test data for evaluation
    train_df = train_df[train_df["station_key"].isin(train_st)]
    test_df  = test_df[test_df["station_key"].isin(all_test_st)]

    # Warmup for building val/test sequences
    WARMUP = pd.Timedelta(hours=LOOKBACK_H - 1)
    val_build_df  = df[(df["dt_utc"] >= (VAL_START - WARMUP)) & (df["dt_utc"] <= VAL_END)].copy()
    # IMPORTANT: Also exclude from val_build_df for consistency
    val_build_df = val_build_df[val_build_df["Country"] != EXCLUDE_COUNTRY].copy()
    val_build_df  = val_build_df[val_build_df["station_key"].isin(train_st)]

    test_build_df = df[(df["dt_utc"] >= (TEST_START - WARMUP)) & (df["dt_utc"] <= TEST_END)].copy()
    test_build_df = test_build_df[test_build_df["station_key"].isin(all_test_st)]

    # Station id mapping: use ALL test stations (including excluded)
    # The model will have embeddings for all stations, but only non-excluded ones will be trained
    station_to_idx = {k: i for i, k in enumerate(all_test_st)}
    idx_to_station = {i: k for k, i in station_to_idx.items()}
    n_stations = len(all_test_st)

    # Fit per-station scalers on ALL training data (before exclusion, excluding val window)
    # This allows us to normalize excluded country stations during test (realistic scenario)
    print(">>> Fitting per-station scalers (ALL training data before exclusion)...")
    train_for_scaler = train_df_full[train_df_full["dt_utc"] < VAL_START].dropna(subset=feature_cols)
    if len(train_for_scaler) < 1000:
        print(">>> WARNING: Less than 1000 samples before val window, using all train data for scalers")
        train_for_scaler = train_df_full.dropna(subset=feature_cols)
    scalers = fit_station_scalers(train_for_scaler, feature_cols)
    save_scalers_json(scalers, feature_cols, SCALERS_JSON)
    print(f">>> Saved scalers: {SCALERS_JSON}")
    print(f">>> Scalers fitted for {len(scalers)} stations (including excluded country)")

    # Precompute station mean/std for inverse transform
    station_mean = np.zeros(n_stations, dtype=np.float32)
    station_std  = np.ones(n_stations, dtype=np.float32)
    for i in range(n_stations):
        sk = idx_to_station[i]
        station_mean[i] = scalers[sk].mean[pm25_idx]
        station_std[i]  = scalers[sk].std[pm25_idx]

    metrics_rows = []
    run_info = {
        "run_seconds": 0.0,
        "excluded_country": EXCLUDE_COUNTRY,
        "lookback_h": LOOKBACK_H,
        "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS,
        "learning_rate": LR,
        "patience": PATIENCE,
        "per_horizon": {}
    }

    for h in HORIZONS:
        print("\n" + "=" * 70)
        print(f">>> Horizon h={h} (single-output residual-attention) [STREAMING DATA]")
        print("=" * 70)
        t_horizon_start = time.time()

        train_ds = make_tf_dataset_streaming(
            df_split=train_df,
            scalers=scalers,
            station_to_idx=station_to_idx,
            feature_cols=feature_cols,
            horizon=h,
            lookback_h=LOOKBACK_H,
            shuffle=True,
            batch_size=BATCH_SIZE,
            anchor_start=None,
            anchor_end=None,
        )

        val_ds = make_tf_dataset_streaming(
            df_split=val_build_df,
            scalers=scalers,
            station_to_idx=station_to_idx,
            feature_cols=feature_cols,
            horizon=h,
            lookback_h=LOOKBACK_H,
            shuffle=False,
            batch_size=BATCH_SIZE,
            anchor_start=VAL_START,
            anchor_end=VAL_END,
        )

        # Count samples
        print(f">>> Counting training samples for h={h}...")
        train_n = 0
        for station_key, g in train_df.groupby("station_key", sort=False):
            if station_key not in scalers:
                continue
            station_idx = station_to_idx[station_key]
            for seg in compute_segments(g, gap_break_hours=GAP_BREAK_HOURS):
                if len(seg) < MIN_SEG_LEN:
                    continue
                for _ in iter_examples_for_segment_single_h(
                    seg=seg, station_idx=station_idx, station_key=station_key,
                    feature_cols=feature_cols, pm25_col="PM2.5", lookback_h=LOOKBACK_H,
                    horizon=h, scaler=scalers[station_key], require_no_nan=True,
                    anchor_start=None, anchor_end=None
                ):
                    train_n += 1
        print(f">>> Training samples: {train_n:,}")

        print(f">>> Counting validation samples for h={h}...")
        val_n = 0
        for station_key, g in val_build_df.groupby("station_key", sort=False):
            if station_key not in scalers:
                continue
            station_idx = station_to_idx[station_key]
            for seg in compute_segments(g, gap_break_hours=GAP_BREAK_HOURS):
                if len(seg) < MIN_SEG_LEN:
                    continue
                for _ in iter_examples_for_segment_single_h(
                    seg=seg, station_idx=station_idx, station_key=station_key,
                    feature_cols=feature_cols, pm25_col="PM2.5", lookback_h=LOOKBACK_H,
                    horizon=h, scaler=scalers[station_key], require_no_nan=True,
                    anchor_start=VAL_START, anchor_end=VAL_END
                ):
                    val_n += 1
        print(f">>> Validation samples: {val_n:,}")

        n_features = len(feature_cols)
        model = build_model_residual_attention_single(
            n_features=n_features,
            n_stations=n_stations,
            pm25_index=pm25_idx
        )
        print(model.summary())

        class LrLogger(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                print(f"Epoch {epoch + 1}: lr={lr:.6g}")

        cb = [
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

        steps_per_epoch = math.ceil(train_n / BATCH_SIZE)
        val_steps = max(1, math.ceil(val_n / BATCH_SIZE))

        print(f">>> Training h={h} ...")
        history = model.fit(
            train_ds.repeat(),
            validation_data=val_ds,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            callbacks=cb,
            verbose=1
        )

        # Save history
        hist_df = pd.DataFrame(history.history)
        hist_df.insert(0, "epoch", np.arange(1, len(hist_df) + 1))
        hist_path = os.path.join(TRAINING_HISTORY_DIR, f"history_h{h}.csv")
        hist_df.to_csv(hist_path, index=False)
        print(f">>> Saved training history: {hist_path}")

        # Save model
        model_path = os.path.join(MODELS_DIR, f"lstm_attention_h{h}.keras")
        model.save(model_path)
        print(f">>> Saved model: {model_path}")

        # STREAMING test dataset with metadata
        test_ds = make_test_dataset_with_meta(
            test_build_df=test_build_df,
            scalers=scalers,
            station_to_idx=station_to_idx,
            feature_cols=feature_cols,
            horizon=h,
            lookback_h=LOOKBACK_H,
            anchor_start=TEST_START,
            anchor_end=TEST_END,
            batch_size=BATCH_SIZE,
        )

        # Predict + metrics streaming
        print(f">>> Predicting TEST h={h} (streaming) ...")
        run_global = RunningRegressionMetrics()
        run_country = RunningGroupedMetrics()
        run_area = RunningGroupedMetrics()

        sample_path = os.path.join(os.path.dirname(PRED_SAMPLE_CSV), f"pred_sample_h{h}.csv")
        with open(sample_path, "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=[
                "horizon","Country","SiteNumber","StationArea_Label",
                "t_anchor_utc","t_target_utc",
                "y_true","y_pred","y_persistence"
            ])
            writer.writeheader()

            written = 0
            n_sample = 5000

            for batch in test_ds:
                x_win_b, sid_b, y_norm_b, b_norm_b, ts_int_b, country_b, site_b, area_b = batch

                y_pred_norm = model.predict(
                    {"x_seq": x_win_b, "station_id": sid_b},
                    verbose=0
                ).reshape(-1).astype(np.float32)

                y_true_norm = y_norm_b.numpy().reshape(-1).astype(np.float32)
                b_norm = b_norm_b.numpy().reshape(-1).astype(np.float32)
                sid_np = sid_b.numpy().reshape(-1).astype(np.int32)

                std_s = station_std[sid_np]
                mean_s = station_mean[sid_np]

                y_true = y_true_norm * std_s + mean_s
                y_pred = y_pred_norm * std_s + mean_s
                y_base = b_norm * std_s + mean_s

                run_global.update(y_true, y_pred)

                country_np = np.array([c.decode("utf-8") for c in country_b.numpy().tolist()], dtype=object)
                area_np    = np.array([a.decode("utf-8") for a in area_b.numpy().tolist()], dtype=object)

                run_country.update(country_np, y_true, y_pred)
                run_area.update(area_np, y_true, y_pred)

                if written < n_sample:
                    ts_int_np = ts_int_b.numpy().reshape(-1).astype(np.int64)
                    site_np   = np.array([s.decode("utf-8") for s in site_b.numpy().tolist()], dtype=object)

                    take = min(n_sample - written, len(y_true))
                    for i in range(take):
                        t_anchor = pd.to_datetime(int(ts_int_np[i]), utc=True)
                        t_target = t_anchor + pd.Timedelta(hours=h)
                        writer.writerow({
                            "horizon": h,
                            "Country": country_np[i],
                            "SiteNumber": site_np[i],
                            "StationArea_Label": area_np[i],
                            "t_anchor_utc": t_anchor.isoformat(),
                            "t_target_utc": t_target.isoformat(),
                            "y_true": float(y_true[i]),
                            "y_pred": float(y_pred[i]),
                            "y_persistence": float(y_base[i]),
                        })
                    written += take

        # finalize metrics
        g = run_global.finalize()
        print(f">>> TEST Global h={h}: MAE={g['MAE']:.4f} RMSE={g['RMSE']:.4f} R2={g['R2']:.4f} Bias={g['Bias']:.4f} N={g['Count']:,}")

        metrics_rows.append({
            "Scope": "Global",
            "Group": "All",
            "horizon": h,
            "MAE": g["MAE"],
            "RMSE": g["RMSE"],
            "R2": g["R2"],
            "Bias": g["Bias"],
            "Count": float(g["Count"]),
        })

        metrics_rows.extend(run_country.finalize_rows(scope="Country", horizon=h))
        metrics_rows.extend(run_area.finalize_rows(scope="StationArea", horizon=h))

        elapsed_h = time.time() - t_horizon_start
        run_info["per_horizon"][f"h{h}"] = {
            "train_seconds": round(elapsed_h, 2),
            "train_n": int(train_n),
            "val_n": int(val_n),
            "test_n": int(g['Count']),
            "model_path": model_path,
            "history_path": hist_path,
        }

        tf.keras.backend.clear_session()
        gc.collect()

    # Save metrics
    metrics_df = pd.DataFrame(metrics_rows)
    cols_order = ["Scope", "Group", "horizon", "MAE", "RMSE", "R2", "Bias", "Count"]
    metrics_df = metrics_df[cols_order].sort_values(["Scope", "Group", "horizon"], kind="stable")
    metrics_df.to_csv(METRICS_CSV, index=False)
    print("\n>>> Saved metrics:", METRICS_CSV)
    print(metrics_df[metrics_df["Scope"] == "Global"])

    # Show performance on excluded city
    excluded_metrics = metrics_df[(metrics_df["Scope"] == "Country") & (metrics_df["Group"] == EXCLUDE_COUNTRY)]
    if len(excluded_metrics) > 0:
        print(f"\n=== Performance on EXCLUDED City ({EXCLUDE_COUNTRY}) - ZERO-SHOT ===")
        print(excluded_metrics.sort_values("horizon"))

    # Merge horizon sample CSVs
    print(">>> Merging per-horizon prediction samples ...")
    sample_rows = []
    for h in HORIZONS:
        p = os.path.join(os.path.dirname(PRED_SAMPLE_CSV), f"pred_sample_h{h}.csv")
        if not os.path.exists(p):
            continue
        sdf = pd.read_csv(p)
        sample_rows.append(sdf)
    if sample_rows:
        pred_df = pd.concat(sample_rows, axis=0, ignore_index=True)
        pred_df.to_csv(PRED_SAMPLE_CSV, index=False)
        print(">>> Saved prediction sample:", PRED_SAMPLE_CSV)

    # Save run info
    run_info["run_seconds"] = round(time.time() - t0, 2)
    with open(RUN_INFO_JSON, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    print(f">>> Run info saved to: {RUN_INFO_JSON}")

    print(f"\n>>> Done in {time.time() - t0:.1f}s.")

if __name__ == "__main__":
    main()
