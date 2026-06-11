import os
import json
import math
import time
import gc
import csv
import glob
import random
import argparse
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

TRAIN_END = pd.Timestamp("2022-12-31 23:00:00", tz="UTC")
TEST_START = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
TEST_END   = pd.Timestamp("2024-12-31 23:00:00", tz="UTC")
# Inner validation window for early stopping (within training period)
VAL_START = pd.Timestamp("2022-07-01 00:00:00", tz="UTC")
VAL_END   = TRAIN_END

LOOKBACK_H = 168
HORIZONS = [1, 3, 6, 12, 24]
GAP_BREAK_HOURS = 2
MIN_SEG_LEN = LOOKBACK_H + max(HORIZONS) + 1  # = 194, aligned with the LSTM scripts

# Training schedule (matched to the LSTM family for a fair comparison)
BATCH_SIZE = 256
EPOCHS = 50
LR = 5e-4
PATIENCE = 6
SHUFFLE_BUFFER = 50000

# iTransformer architecture (single sane config; no HPO for a baseline)
D_MODEL = 128
N_HEADS = 8
FF_DIM = 256
N_BLOCKS = 2
DROPOUT = 0.2
STATION_TOKEN = True  # prepend a learned station token so attention can use site identity

SEED = 42

# Output paths (standardized, mirrors lstm_attention_* naming)
MODELS_DIR = "models/itransformer_models"
SCALERS_JSON_BASE = "train_models/scalers/itransformer_scalers"  # + _h{h}.json
TRAINING_HISTORY_DIR = "train_models/history/itransformer_histories"
RUNS_INFO_DIR = "train_models/global/runs_info"
RESULTS_DIR = "results"
METRICS_CSV = "results/itransformer_metrics.csv"
PRED_SAMPLE_CSV = "results/itransformer_predictions_sample.csv"

COLS_ORDER = ["Scope", "Group", "horizon", "MAE", "RMSE", "R2", "Bias", "Count"]


def make_dirs():
    os.makedirs(RUNS_INFO_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SCALERS_JSON_BASE), exist_ok=True)
    os.makedirs(TRAINING_HISTORY_DIR, exist_ok=True)


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)


def configure_gpu():
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


# ============================================================
# METRICS (identical to lstm_global_attention.py)
# ============================================================
class RunningRegressionMetrics:
    """Global running metrics without storing arrays."""
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
    """Running metrics per group key (Country / StationArea) without storing arrays."""
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
                s = {"n": 0, "sum_abs": 0.0, "sum_sq": 0.0, "sum_err": 0.0,
                     "sum_y": 0.0, "sum_y2": 0.0, "ssr": 0.0}
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
            rows.append({"Scope": scope, "Group": group, "horizon": horizon,
                         "MAE": float(mae), "RMSE": float(rmse), "R2": float(r2),
                         "Bias": float(bias), "Count": float(n)})
        return rows


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


# ============================================================
# DATA HELPERS (identical to lstm_global_attention.py)
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
    """Time-based shifting on complete hourly grids (gap-correct). Copied from the LSTM scripts."""
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
# STREAMING WINDOW ITERATORS (identical to lstm_global_attention.py)
# ============================================================
def iter_examples_for_segment_single_h(
    seg, station_idx, station_key, feature_cols, pm25_col,
    lookback_h, horizon, scaler,
    require_no_nan=True, anchor_start=None, anchor_end=None,
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
    aend64 = np.datetime64(anchor_end.to_datetime64()) if anchor_end is not None else None

    for t in range(start_t, end_t):
        ts = tstamp[t]
        if astart64 is not None and ts < astart64:
            continue
        if aend64 is not None and ts > aend64:
            continue
        x_win = x_norm[t - lookback_h + 1: t + 1, :]
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
    df_split, scalers, station_to_idx, feature_cols, horizon, lookback_h,
    shuffle, batch_size, anchor_start=None, anchor_end=None,
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
                    seg=seg, station_idx=station_idx, station_key=station_key,
                    feature_cols=feature_cols, pm25_col="PM2.5", lookback_h=lookback_h,
                    horizon=horizon, scaler=scalers[station_key], require_no_nan=True,
                    anchor_start=anchor_start, anchor_end=anchor_end,
                ):
                    yield {"x_seq": x_win, "station_id": sid}, np.array([y], dtype=np.float32)

    output_signature = (
        {"x_seq": tf.TensorSpec(shape=(lookback_h, n_features), dtype=tf.float32),
         "station_id": tf.TensorSpec(shape=(), dtype=tf.int32)},
        tf.TensorSpec(shape=(1,), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_test_dataset_with_meta(
    test_build_df, scalers, station_to_idx, feature_cols, horizon, lookback_h,
    anchor_start, anchor_end, batch_size,
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
                    seg=seg, station_idx=station_idx, station_key=station_key,
                    feature_cols=feature_cols, pm25_col="PM2.5", lookback_h=lookback_h,
                    horizon=horizon, scaler=scalers[station_key], require_no_nan=True,
                    anchor_start=anchor_start, anchor_end=anchor_end,
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
                    yield (x_win, np.int32(sid), np.float32(y), np.float32(b), np.int64(ts_int),
                           country.encode("utf-8"), site.encode("utf-8"), area.encode("utf-8"))

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


def count_examples(df_split, scalers, station_to_idx, feature_cols, horizon,
                   anchor_start=None, anchor_end=None) -> int:
    n = 0
    for station_key, g in df_split.groupby("station_key", sort=False):
        if station_key not in scalers:
            continue
        station_idx = station_to_idx[station_key]
        for seg in compute_segments(g, gap_break_hours=GAP_BREAK_HOURS):
            if len(seg) < MIN_SEG_LEN:
                continue
            for _ in iter_examples_for_segment_single_h(
                seg=seg, station_idx=station_idx, station_key=station_key,
                feature_cols=feature_cols, pm25_col="PM2.5", lookback_h=LOOKBACK_H,
                horizon=horizon, scaler=scalers[station_key], require_no_nan=True,
                anchor_start=anchor_start, anchor_end=anchor_end,
            ):
                n += 1
    return n


# ============================================================
# MODEL: iTransformer (inverted Transformer over variate tokens)
# ============================================================
def build_itransformer_single(n_features: int, n_stations: int, pm25_index: int) -> Model:
    """
    iTransformer (Liu et al., ICLR 2024), adapted for single-output direct forecasting.

    Key idea ("inversion"): each input VARIATE (feature) — its full length-L window —
    is embedded into one token of dimension D. Self-attention then operates ACROSS the
    variate tokens (multivariate correlations), not across time steps; there is
    deliberately no temporal positional encoding. We additionally prepend a learned
    STATION token so the model can attend over site identity alongside the variates
    (this is the part that gives the model an inter-series / quasi-spatial mechanism).

    The head predicts a residual delta added to the last observed (normalized) PM2.5,
    matching the residual anchoring used by the LSTM family.
    """
    seq_in = layers.Input(shape=(LOOKBACK_H, n_features), name="x_seq")
    sid_in = layers.Input(shape=(), dtype="int32", name="station_id")

    # Residual anchor: last observed (normalized) PM2.5 at time t
    pm25_t = layers.Lambda(lambda z: z[:, -1, pm25_index:pm25_index + 1], name="pm25_t")(seq_in)

    # --- Inversion: [B, L, F] -> [B, F, L] -> embed each variate series to D ---
    x = layers.Permute((2, 1), name="invert_to_variate_tokens")(seq_in)   # [B, F, L]
    x = layers.Dense(D_MODEL, name="variate_embed")(x)                    # [B, F, D]
    x = layers.Dropout(DROPOUT)(x)

    pm25_token_pos = pm25_index  # position of the PM2.5 token among variate tokens

    # --- Optional station token, prepended at position 0 ---
    if STATION_TOKEN:
        sid_emb = layers.Embedding(input_dim=n_stations, output_dim=D_MODEL, name="station_emb")(sid_in)  # [B, D]
        sid_tok = layers.Reshape((1, D_MODEL), name="station_token")(sid_emb)                              # [B, 1, D]
        x = layers.Concatenate(axis=1, name="tokens")([sid_tok, x])       # [B, F+1, D]
        pm25_token_pos = pm25_index + 1

    # --- Transformer encoder over tokens (Pre-LN blocks) ---
    for i in range(N_BLOCKS):
        h = layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{i}")(x)
        attn = layers.MultiHeadAttention(
            num_heads=N_HEADS, key_dim=max(1, D_MODEL // N_HEADS),
            dropout=DROPOUT, name=f"mha_{i}",
        )(h, h)
        x = layers.Add(name=f"res1_{i}")([x, attn])

        h = layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{i}")(x)
        h = layers.Dense(FF_DIM, activation="gelu", name=f"ff1_{i}")(h)
        h = layers.Dropout(DROPOUT)(h)
        h = layers.Dense(D_MODEL, name=f"ff2_{i}")(h)
        x = layers.Add(name=f"res2_{i}")([x, h])

    x = layers.LayerNormalization(epsilon=1e-6, name="ln_final")(x)       # [B, T, D]

    # --- Head: PM2.5 variate token + global pooled context -> residual delta ---
    pm25_token = layers.Lambda(lambda z: z[:, pm25_token_pos, :], name="pm25_token")(x)  # [B, D]
    pooled = layers.GlobalAveragePooling1D(name="token_pool")(x)                          # [B, D]
    head = layers.Concatenate(name="head_concat")([pm25_token, pooled])

    head = layers.Dense(128, activation="relu",
                        kernel_regularizer=tf.keras.regularizers.l2(1e-5))(head)
    head = layers.Dropout(DROPOUT)(head)
    head = layers.Dense(64, activation="relu",
                        kernel_regularizer=tf.keras.regularizers.l2(1e-5))(head)
    head = layers.Dropout(DROPOUT)(head)
    delta = layers.Dense(1, name="delta")(head)

    y_hat = layers.Add(name="y_hat")([pm25_t, delta])

    model = Model(inputs=[seq_in, sid_in], outputs=y_hat)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"), R2Metric(name="r2")],
    )
    return model


# ============================================================
# DATA PREP (shared)
# ============================================================
FEATURES_FROM_FILE = [
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


def prepare_data():
    print(">>> Loading CSV...")
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

    non_feature_cols = {"Country", "SiteNumber", "dt_utc", "dt_local", "Start", "PM2.5_next"}
    feature_cols = [c for c in FEATURES_FROM_FILE if (c in df.columns and c not in non_feature_cols)]
    if "PM2.5" not in feature_cols:
        raise ValueError("PM2.5 must be present in feature_cols.")
    pm25_idx = feature_cols.index("PM2.5")
    print(f">>> Features used: {len(feature_cols)}")

    print(">>> Building time-based multi-horizon targets...")
    df = build_multi_horizon_targets_like_lr(df, HORIZONS)

    train_df = df[df["dt_utc"] <= TRAIN_END].copy()
    test_df = df[(df["dt_utc"] >= TEST_START) & (df["dt_utc"] <= TEST_END)].copy()

    train_st = set(train_df["station_key"].unique())
    test_st = set(test_df["station_key"].unique())
    common_st = sorted(list(train_st & test_st))
    print(f">>> Stations in common (train & test): {len(common_st)}")

    train_df = train_df[train_df["station_key"].isin(common_st)]

    WARMUP = pd.Timedelta(hours=LOOKBACK_H - 1)
    val_build_df = df[(df["dt_utc"] >= (VAL_START - WARMUP)) & (df["dt_utc"] <= VAL_END)].copy()
    val_build_df = val_build_df[val_build_df["station_key"].isin(common_st)]
    test_build_df = df[(df["dt_utc"] >= (TEST_START - WARMUP)) & (df["dt_utc"] <= TEST_END)].copy()
    test_build_df = test_build_df[test_build_df["station_key"].isin(common_st)]

    station_to_idx = {k: i for i, k in enumerate(common_st)}
    idx_to_station = {i: k for k, i in station_to_idx.items()}
    n_stations = len(common_st)

    print(">>> Fitting per-station scalers (TRAIN only, excluding inner val window)...")
    train_for_scaler = train_df[train_df["dt_utc"] < VAL_START].dropna(subset=feature_cols)
    if len(train_for_scaler) < 1000:
        print(">>> WARNING: <1000 samples before val window, using all train data for scalers")
        train_for_scaler = train_df.dropna(subset=feature_cols)
    scalers = fit_station_scalers(train_for_scaler, feature_cols)

    station_mean = np.zeros(n_stations, dtype=np.float32)
    station_std = np.ones(n_stations, dtype=np.float32)
    for i in range(n_stations):
        sk = idx_to_station[i]
        station_mean[i] = scalers[sk].mean[pm25_idx]
        station_std[i] = scalers[sk].std[pm25_idx]

    return dict(
        feature_cols=feature_cols, pm25_idx=pm25_idx,
        train_df=train_df, val_build_df=val_build_df, test_build_df=test_build_df,
        scalers=scalers, station_to_idx=station_to_idx, n_stations=n_stations,
        station_mean=station_mean, station_std=station_std,
    )


# ============================================================
# TRAIN + EVAL ONE HORIZON
# ============================================================
def run_horizon(h: int, data: dict, epochs: int) -> Tuple[List[Dict], Dict]:
    feature_cols = data["feature_cols"]
    pm25_idx = data["pm25_idx"]
    scalers = data["scalers"]
    station_to_idx = data["station_to_idx"]
    n_stations = data["n_stations"]
    station_mean = data["station_mean"]
    station_std = data["station_std"]
    train_df = data["train_df"]
    val_build_df = data["val_build_df"]
    test_build_df = data["test_build_df"]

    print("\n" + "=" * 70)
    print(f">>> iTransformer horizon h={h} [STREAMING DATA]")
    print("=" * 70)
    t_horizon_start = time.time()

    train_ds = make_tf_dataset_streaming(
        train_df, scalers, station_to_idx, feature_cols, h, LOOKBACK_H,
        shuffle=True, batch_size=BATCH_SIZE, anchor_start=None, anchor_end=None,
    )
    val_ds = make_tf_dataset_streaming(
        val_build_df, scalers, station_to_idx, feature_cols, h, LOOKBACK_H,
        shuffle=False, batch_size=BATCH_SIZE, anchor_start=VAL_START, anchor_end=VAL_END,
    )

    print(f">>> Counting training samples for h={h}...")
    train_n = count_examples(train_df, scalers, station_to_idx, feature_cols, h, None, None)
    print(f">>> Training samples: {train_n:,}")
    print(f">>> Counting validation samples for h={h}...")
    val_n = count_examples(val_build_df, scalers, station_to_idx, feature_cols, h, VAL_START, VAL_END)
    print(f">>> Validation samples: {val_n:,}")

    model = build_itransformer_single(len(feature_cols), n_stations, pm25_idx)
    print(model.summary())

    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=PATIENCE,
                                         restore_best_weights=True, min_delta=1e-4),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=2,
                                             min_lr=1e-5, verbose=1),
    ]

    steps_per_epoch = math.ceil(train_n / BATCH_SIZE)
    val_steps = max(1, math.ceil(val_n / BATCH_SIZE))

    print(f">>> Training h={h} ...")
    history = model.fit(
        train_ds.repeat(), validation_data=val_ds, epochs=epochs,
        steps_per_epoch=steps_per_epoch, validation_steps=val_steps,
        callbacks=cb, verbose=1,
    )

    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0, "epoch", np.arange(1, len(hist_df) + 1))
    hist_path = os.path.join(TRAINING_HISTORY_DIR, f"history_h{h}.csv")
    hist_df.to_csv(hist_path, index=False)
    print(f">>> Saved training history: {hist_path}")

    model_path = os.path.join(MODELS_DIR, f"itransformer_h{h}.keras")
    model.save(model_path)
    print(f">>> Saved model: {model_path}")

    test_ds = make_test_dataset_with_meta(
        test_build_df, scalers, station_to_idx, feature_cols, h, LOOKBACK_H,
        anchor_start=TEST_START, anchor_end=TEST_END, batch_size=BATCH_SIZE,
    )

    print(f">>> Predicting TEST h={h} (streaming) ...")
    run_global = RunningRegressionMetrics()
    run_country = RunningGroupedMetrics()
    run_area = RunningGroupedMetrics()

    sample_path = os.path.join(RESULTS_DIR, f"itransformer_pred_sample_h{h}.csv")
    with open(sample_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=[
            "horizon", "Country", "SiteNumber", "StationArea_Label",
            "t_anchor_utc", "t_target_utc", "y_true", "y_pred", "y_persistence",
        ])
        writer.writeheader()
        written = 0
        n_sample = 5000
        for batch in test_ds:
            x_win_b, sid_b, y_norm_b, b_norm_b, ts_int_b, country_b, site_b, area_b = batch
            y_pred_norm = model.predict({"x_seq": x_win_b, "station_id": sid_b}, verbose=0).reshape(-1).astype(np.float32)
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
            area_np = np.array([a.decode("utf-8") for a in area_b.numpy().tolist()], dtype=object)
            run_country.update(country_np, y_true, y_pred)
            run_area.update(area_np, y_true, y_pred)

            if written < n_sample:
                ts_int_np = ts_int_b.numpy().reshape(-1).astype(np.int64)
                site_np = np.array([s.decode("utf-8") for s in site_b.numpy().tolist()], dtype=object)
                take = min(n_sample - written, len(y_true))
                for i in range(take):
                    t_anchor = pd.to_datetime(int(ts_int_np[i]), utc=True)
                    t_target = t_anchor + pd.Timedelta(hours=h)
                    writer.writerow({
                        "horizon": h, "Country": country_np[i], "SiteNumber": site_np[i],
                        "StationArea_Label": area_np[i],
                        "t_anchor_utc": t_anchor.isoformat(), "t_target_utc": t_target.isoformat(),
                        "y_true": float(y_true[i]), "y_pred": float(y_pred[i]),
                        "y_persistence": float(y_base[i]),
                    })
                written += take

    g = run_global.finalize()
    print(f">>> TEST Global h={h}: MAE={g['MAE']:.4f} RMSE={g['RMSE']:.4f} "
          f"R2={g['R2']:.4f} Bias={g['Bias']:.4f} N={g['Count']:,}")

    rows_h = [{
        "Scope": "Global", "Group": "All", "horizon": h,
        "MAE": g["MAE"], "RMSE": g["RMSE"], "R2": g["R2"], "Bias": g["Bias"],
        "Count": float(g["Count"]),
    }]
    rows_h.extend(run_country.finalize_rows(scope="Country", horizon=h))
    rows_h.extend(run_area.finalize_rows(scope="StationArea", horizon=h))

    # Per-horizon metrics file (so parallel per-machine runs never clobber each other)
    per_h_metrics = os.path.join(RESULTS_DIR, f"itransformer_metrics_h{h}.csv")
    pd.DataFrame(rows_h)[COLS_ORDER].to_csv(per_h_metrics, index=False)
    print(f">>> Saved per-horizon metrics: {per_h_metrics}")

    elapsed_h = time.time() - t_horizon_start
    run_info = {
        "model": "itransformer", "lookback_h": LOOKBACK_H, "batch_size": BATCH_SIZE,
        "max_epochs": epochs, "learning_rate": LR, "patience": PATIENCE,
        "d_model": D_MODEL, "n_heads": N_HEADS, "ff_dim": FF_DIM, "n_blocks": N_BLOCKS,
        "dropout": DROPOUT, "station_token": STATION_TOKEN,
        "run_seconds": round(elapsed_h, 2),
        "per_horizon": {f"h{h}": {
            "train_seconds": round(elapsed_h, 2),
            "train_n": int(train_n), "val_n": int(val_n), "test_n": int(g["Count"]),
            "epochs_run": int(len(hist_df)),
            "model_path": model_path, "history_path": hist_path,
        }},
    }
    per_h_info = os.path.join(RUNS_INFO_DIR, f"itransformer_h{h}_run_info.json")
    with open(per_h_info, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    print(f">>> Saved per-horizon run info: {per_h_info}")

    tf.keras.backend.clear_session()
    gc.collect()
    return rows_h, run_info["per_horizon"][f"h{h}"]


# ============================================================
# MERGE per-horizon outputs into combined files
# ============================================================
def merge_outputs():
    print(">>> Merging per-horizon iTransformer outputs...")
    metric_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "itransformer_metrics_h*.csv")))
    dfs = [pd.read_csv(p) for p in metric_files]
    if dfs:
        allm = pd.concat(dfs, ignore_index=True)[COLS_ORDER]
        allm = allm.sort_values(["Scope", "Group", "horizon"], kind="stable")
        allm.to_csv(METRICS_CSV, index=False)
        print(f">>> Wrote {METRICS_CSV} ({len(metric_files)} horizon files)")
        print(allm[allm["Scope"] == "Global"])
    else:
        print(">>> No per-horizon metrics files found.")

    sfiles = sorted(glob.glob(os.path.join(RESULTS_DIR, "itransformer_pred_sample_h*.csv")))
    sdfs = [pd.read_csv(p) for p in sfiles]
    if sdfs:
        pd.concat(sdfs, ignore_index=True).to_csv(PRED_SAMPLE_CSV, index=False)
        print(f">>> Wrote {PRED_SAMPLE_CSV}")

    rfiles = sorted(glob.glob(os.path.join(RUNS_INFO_DIR, "itransformer_h*_run_info.json")))
    merged = {"model": "itransformer", "per_horizon": {}}
    total = 0.0
    for p in rfiles:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        merged["per_horizon"].update(d.get("per_horizon", {}))
        for kk in ["lookback_h", "batch_size", "max_epochs", "learning_rate", "patience",
                   "d_model", "n_heads", "ff_dim", "n_blocks", "dropout", "station_token"]:
            if kk in d:
                merged[kk] = d[kk]
        total += float(d.get("run_seconds", 0.0))
    merged["run_seconds"] = round(total, 2)
    if rfiles:
        with open(os.path.join(RUNS_INFO_DIR, "itransformer_run_info.json"), "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
        print(f">>> Wrote {os.path.join(RUNS_INFO_DIR, 'itransformer_run_info.json')}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="iTransformer baseline (Protocol B).")
    parser.add_argument("--horizon", type=int, default=None, choices=HORIZONS,
                        help="Train/eval a single horizon (for one-per-machine parallelism). "
                             "If omitted, runs all horizons sequentially.")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Max epochs (early stopping still applies). Lower to bound long horizons.")
    parser.add_argument("--merge", action="store_true",
                        help="Only merge existing per-horizon outputs into combined files, then exit.")
    args = parser.parse_args()

    make_dirs()

    if args.merge:
        merge_outputs()
        return

    set_seeds()
    configure_gpu()

    t0 = time.time()
    data = prepare_data()

    horizons_to_run = [args.horizon] if args.horizon is not None else HORIZONS
    # Save scalers (per-horizon filename when single, to avoid any same-machine collision)
    scaler_path = (f"{SCALERS_JSON_BASE}_h{args.horizon}.json"
                   if args.horizon is not None else f"{SCALERS_JSON_BASE}.json")
    save_scalers_json(data["scalers"], data["feature_cols"], scaler_path)
    print(f">>> Saved scalers: {scaler_path}")

    all_rows = []
    for h in horizons_to_run:
        rows_h, _ = run_horizon(h, data, epochs=args.epochs)
        all_rows.extend(rows_h)

    # If we ran the full set on one machine, also write the combined files directly.
    if args.horizon is None:
        merge_outputs()

    print(f"\n>>> Done in {time.time() - t0:.1f}s. "
          f"Ran horizon(s): {horizons_to_run}")
    if args.horizon is not None:
        print(">>> When all horizons are collected on one machine, run:")
        print(">>>   python train_models/global/itransformer_global.py --merge")


if __name__ == "__main__":
    main()
