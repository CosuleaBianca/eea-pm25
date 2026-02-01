import os
import json
import time
import pickle
import gc
import random
import numpy as np
import pandas as pd
from typing import Optional

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'train_models'))

from dl_sampling_utils import (
    GAP_BREAK_HOURS,
    LOOKBACK_H,
    build_lstm_aligned_anchors,
    build_multi_horizon_targets_like_lstm,
    filter_df_by_anchors,
    make_station_key,
    normalize_season_column,
    select_feature_columns_like_lstm_script,
)

# =========================
# CONFIG
# =========================
CSV_PATH = "ml_ready_dataset_full_realistic.csv"

OUTPUT_METRICS_PATH = "results/rf_dl_metrics_tuned.csv"
OUTPUT_PREDS_PATH = "results/rf_dl_predictions_sample.csv"
OUTPUT_BEST_PARAMS_JSON = "train_models/global/rf_dl_best_params_by_horizon.json"
OUTPUT_RUN_INFO_JSON = "train_models/global/runs_info/rf_dl_run_info.json"

MODEL_DIR = "models/models_rf_dl"

# Create all output directories upfront to prevent data loss
print("Creating output directories...")
os.makedirs("train_models/global/runs_info", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
print("All directories created successfully.")

HORIZONS = [1, 3, 6, 12, 24]

# Train/Test split
TRAIN_END = pd.Timestamp("2022-12-31 23:00:00", tz="UTC")
TEST_START = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
TEST_END = pd.Timestamp("2024-12-31 23:00:00", tz="UTC")

# Internal split for tuning (within TRAIN)
TUNE_TRAIN_END = pd.Timestamp("2021-12-31 23:00:00", tz="UTC")
TUNE_VAL_START = pd.Timestamp("2022-01-01 00:00:00", tz="UTC")
TUNE_VAL_END = pd.Timestamp("2022-12-31 23:00:00", tz="UTC")

# Speed optimization parameters
TUNE_SAMPLE_N = 60_000
N_ITER_SEARCH = 10
CV_SPLITS = 2
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# RF param space
PARAM_DISTRIBUTIONS_FAST = {
    "n_estimators": [200, 300, 400],
    "max_depth": [20, 30, 40, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 1.0],
    "bootstrap": [True],
}

# =========================
# Helpers
# =========================
def build_multi_horizon_targets(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    df = df.copy()
    df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True, errors="coerce")
    df = df.sort_values(["Country", "SiteNumber", "dt_utc"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["Country", "SiteNumber", "dt_utc"], keep="first")

    df["_row_id"] = np.arange(len(df))
    out_rows = []

    for (country, site), g in df.groupby(["Country", "SiteNumber"], sort=False):
        g = g.sort_values("dt_utc")
        idx_orig = g["dt_utc"]
        row_ids = g["_row_id"].values

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


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "Start", "dt_utc", "dt_local",
        "Country", "SiteNumber",
        "PM2.5", "NO2", "PM10",
        "season", "PM2.5_next"
    }
    candidates = []
    for c in df.columns:
        if c in exclude:
            continue
        if c.startswith("y_h"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            candidates.append(c)

    # Include raw NO2/PM10 as in LR
    for c in ["NO2", "PM10"]:
        if c in df.columns:
            candidates.append(c)

    return list(dict.fromkeys(candidates))


def recover_categorical_metadata(df: pd.DataFrame) -> pd.DataFrame:
    area_cols = [c for c in df.columns if c.startswith("StationArea_") and "Label" not in c]
    if not area_cols:
        df["StationArea_Label"] = "Unknown"
        return df
    df["StationArea_Label"] = df[area_cols].idxmax(axis=1).apply(lambda x: x.replace("StationArea_", ""))
    return df


def calculate_metrics(sub_df: pd.DataFrame) -> pd.Series:
    y_true = sub_df["y_true"].to_numpy()
    y_pred = sub_df["y_pred"].to_numpy()

    if len(y_true) == 0:
        return pd.Series({"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "Bias": np.nan, "Count": 0})

    return pd.Series({
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "Bias": float(np.mean(y_pred - y_true)),
        "Count": int(len(y_true)),
    })


def take_last_n_sorted(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n is None or len(df) <= n:
        return df.sort_values("dt_utc")
    return df.sort_values("dt_utc").iloc[-n:]


def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_hyperparameter_optimization(
    train_df_base: pd.DataFrame,
    feature_cols: list[str],
    horizons: list[int],
    train_anchors_by_h: Optional[dict[int, pd.DataFrame]] = None,
) -> dict:
    """
    Run hyperparameter optimization for each horizon.

    Returns:
        dict: best_params_by_h, a dictionary mapping f"h{h}" to best params
    """
    best_params_by_h = {}

    for h in horizons:
        print(f"\n================= HPO for HORIZON h={h} =================")

        train_h = train_df_base.dropna(subset=[f"y_h{h}"]).copy()
        if train_anchors_by_h is not None:
            train_h = filter_df_by_anchors(train_h, train_anchors_by_h.get(h))

        if len(train_h) == 0:
            print(f"Skipping h={h} (no data)")
            continue

        # tuning pool (prefer <=2021)
        tune_train = train_h[train_h["dt_utc"] <= TUNE_TRAIN_END].copy()
        tune_val = train_h[(train_h["dt_utc"] >= TUNE_VAL_START) & (train_h["dt_utc"] <= TUNE_VAL_END)].copy()

        use_internal_val = (len(tune_train) > 10_000 and len(tune_val) > 10_000)
        tune_pool = tune_train if use_internal_val else train_h

        tune_pool_s = take_last_n_sorted(tune_pool, TUNE_SAMPLE_N)

        X_tune = tune_pool_s[feature_cols].to_numpy(dtype=np.float32)
        y_tune = tune_pool_s[f"y_h{h}"].to_numpy(dtype=np.float32)

        print(f"[Tuning] pool_n={len(tune_pool_s):,} | cv_splits={CV_SPLITS} | n_iter={N_ITER_SEARCH}")

        rf_base = RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

        search = RandomizedSearchCV(
            estimator=rf_base,
            param_distributions=PARAM_DISTRIBUTIONS_FAST,
            n_iter=N_ITER_SEARCH,
            scoring="neg_mean_absolute_error",
            cv=tscv,
            verbose=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        t_search0 = time.time()
        search.fit(X_tune, y_tune)
        tune_sec = time.time() - t_search0

        best_params = search.best_params_
        best_params_by_h[f"h{h}"] = best_params
        print(f"[Tuning] Done in {tune_sec:.1f}s | best MAE (CV) = {-search.best_score_:.4f}")
        print(f"[Tuning] Best params: {best_params}")

    return best_params_by_h


def get_or_tune_best_params(
    train_df_base: pd.DataFrame,
    feature_cols: list[str],
    horizons: list[int],
    json_path: str,
    train_anchors_by_h: Optional[dict[int, pd.DataFrame]] = None,
) -> dict:
    """
    Check if best params JSON exists. If yes, load it. If not, run HPO and save.

    Returns:
        dict: best_params_by_h
    """
    if os.path.exists(json_path):
        print(f"\n>>> Loading existing best params from: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            best_params_by_h = json.load(f)
        print(f">>> Loaded params for horizons: {list(best_params_by_h.keys())}")
        return best_params_by_h
    else:
        print(f"\n>>> No existing params found at: {json_path}")
        print(">>> Running hyperparameter optimization...")
        best_params_by_h = run_hyperparameter_optimization(
            train_df_base,
            feature_cols,
            horizons,
            train_anchors_by_h=train_anchors_by_h,
        )

        # Save the params
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(best_params_by_h, f, indent=2)
        print(f">>> Best params saved to: {json_path}")

        return best_params_by_h


# =========================
# MAIN
# =========================
def main():
    run_t0 = time.time()
    print(">>> [RF Tuned FAST] Loading CSV...")
    df = pd.read_csv(CSV_PATH, low_memory=False)

    print(">>> Building multi-horizon targets...")
    df = build_multi_horizon_targets_like_lstm(df, HORIZONS)
    df = make_station_key(df)
    df = recover_categorical_metadata(df)
    df = normalize_season_column(df)

    lstm_feature_cols = select_feature_columns_like_lstm_script(df)
    train_anchors_by_h, test_anchors_by_h, common_stations, scaler_stations = build_lstm_aligned_anchors(
        df=df,
        feature_cols=lstm_feature_cols,
        horizons=HORIZONS,
        train_end=TRAIN_END,
        test_start=TEST_START,
        test_end=TEST_END,
        lookback_h=LOOKBACK_H,
        gap_break_hours=GAP_BREAK_HOURS,
    )
    df = df[df["station_key"].isin(common_stations)].copy()
    print(f"LSTM-aligned stations: {len(common_stations)} | stations with train features: {len(scaler_stations)}")

    feature_cols = select_feature_columns(df)
    df_model = df.dropna(subset=feature_cols).copy()

    df_model["dt_utc"] = pd.to_datetime(df_model["dt_utc"], utc=True)
    train_mask = df_model["dt_utc"] <= TRAIN_END
    test_mask = (df_model["dt_utc"] >= TEST_START) & (df_model["dt_utc"] <= TEST_END)

    train_df_base = df_model.loc[train_mask].copy()
    test_df_base = df_model.loc[test_mask].copy()

    print(f"Base Train (features valid): {len(train_df_base):,} | Base Test: {len(test_df_base):,}")
    print(f"Features used: {len(feature_cols)}")
    print("-" * 70)

    # Get or tune best params
    best_params_by_h = get_or_tune_best_params(
        train_df_base,
        feature_cols,
        HORIZONS,
        OUTPUT_BEST_PARAMS_JSON,
        train_anchors_by_h=train_anchors_by_h,
    )

    per_h_info = {}
    all_results = []

    for h in HORIZONS:
        h_t0 = time.time()
        print(f"\n================= HORIZON h={h} =================")

        train_h = train_df_base.dropna(subset=[f"y_h{h}"]).copy()
        test_h = test_df_base.dropna(subset=[f"y_h{h}"]).copy()
        train_h = filter_df_by_anchors(train_h, train_anchors_by_h.get(h))
        test_h = filter_df_by_anchors(test_h, test_anchors_by_h.get(h))

        if len(train_h) == 0 or len(test_h) == 0:
            print(f"Skipping h={h} (no data)")
            continue

        # Get best params for this horizon
        best_params = best_params_by_h.get(f"h{h}")
        if best_params is None:
            print(f"No best params found for h={h}, skipping")
            continue

        print(f"[Using params] {best_params}")

        # Train final on full TRAIN<=2022 (per-h subset)
        X_train_full = train_h[feature_cols].to_numpy(dtype=np.float32)
        y_train_full = train_h[f"y_h{h}"].to_numpy(dtype=np.float32)

        final_rf = RandomForestRegressor(
            **best_params,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        t_fit0 = time.time()
        final_rf.fit(X_train_full, y_train_full)
        fit_sec = time.time() - t_fit0
        print(f"[Train] Fit done in {fit_sec:.1f}s | train_n={len(train_h):,}")

        # Save model
        model_path = os.path.join(MODEL_DIR, f"rf_h{h}.pkl")
        save_pickle(final_rf, model_path)
        print(f"[Save] Model saved: {model_path}")

        # Test eval
        X_test = test_h[feature_cols].to_numpy(dtype=np.float32)
        y_test = test_h[f"y_h{h}"].to_numpy(dtype=np.float32)
        y_pred = final_rf.predict(X_test)

        res_h = test_h[["Country", "SiteNumber", "StationArea_Label"]].copy()
        res_h["dt_utc"] = test_h["dt_utc"]
        res_h["horizon"] = h
        res_h["y_true"] = y_test
        res_h["y_pred"] = y_pred
        all_results.append(res_h)

        per_h_info[f"h{h}"] = {
            "train_seconds": round(fit_sec, 2),  # Renamed from fit_seconds
            "train_n": int(len(train_h)),
            "val_n": 0,  # RF uses CV during HPO, not explicit validation in final training
            "test_n": int(len(test_h)),
            "model_path": model_path,
            "best_params": best_params,
        }

        print(f"[Test] Done | test_n={len(test_h):,} | horizon_total_sec={time.time()-h_t0:.1f}s")

        # Memory cleanup
        del train_h, test_h, X_train_full, y_train_full, X_test, y_test, y_pred, final_rf
        gc.collect()

    if not all_results:
        print("No results generated.")
        return

    # Metrics (identic LR)
    full_results_df = pd.concat(all_results, axis=0, ignore_index=True)

    print("\n=== Calculating Granular Metrics (aligned with LR) ===")
    global_metrics = full_results_df.groupby("horizon").apply(calculate_metrics).reset_index()
    global_metrics["Scope"] = "Global"
    global_metrics["Group"] = "All"

    country_metrics = full_results_df.groupby(["horizon", "Country"]).apply(calculate_metrics).reset_index()
    country_metrics["Scope"] = "Country"
    country_metrics = country_metrics.rename(columns={"Country": "Group"})

    area_metrics = full_results_df.groupby(["horizon", "StationArea_Label"]).apply(calculate_metrics).reset_index()
    area_metrics["Scope"] = "StationArea"
    area_metrics = area_metrics.rename(columns={"StationArea_Label": "Group"})

    final_report = pd.concat([global_metrics, country_metrics, area_metrics], ignore_index=True)
    final_report = final_report[["Scope", "Group", "horizon", "MAE", "RMSE", "R2", "Bias", "Count"]]

    print("\n=== Sample of Results (Global RF Tuned FAST) ===")
    print(final_report[final_report["Scope"] == "Global"].sort_values("horizon"))

    final_report.to_csv(OUTPUT_METRICS_PATH, index=False)
    print(f"\nDetailed metrics saved to: {OUTPUT_METRICS_PATH}")

    full_results_df.head(10000).to_csv(OUTPUT_PREDS_PATH, index=False)
    print(f"Predictions sample saved to: {OUTPUT_PREDS_PATH}")

    run_info = {
        "run_seconds": round(time.time() - run_t0, 2),
        "tune_sample_n": int(TUNE_SAMPLE_N),
        "n_iter_search": int(N_ITER_SEARCH),
        "cv_splits": int(CV_SPLITS),
        "param_space": "PARAM_DISTRIBUTIONS_FAST",
        "per_horizon": per_h_info,
    }
    with open(OUTPUT_RUN_INFO_JSON, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    print(f"Run info saved to: {OUTPUT_RUN_INFO_JSON}")

    print(f"\nDone in {time.time() - run_t0:.1f}s")


if __name__ == "__main__":
    main()
