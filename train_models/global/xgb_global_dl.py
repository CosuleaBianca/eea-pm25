import json
import time
import os
import joblib
import gc
import random
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
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
OUTPUT_METRICS_PATH = "results/xgb_dl_metrics_tuned.csv"
OUTPUT_PREDS_PATH = "results/xgb_dl_predictions_sample.csv"
OUTPUT_BEST_PARAMS_JSON = "train_models/global/xgb_dl_best_params_by_horizon.json"
OUTPUT_RUN_INFO_JSON = "train_models/global/runs_info/xgb_dl_run_info.json"
OUTPUT_MODEL_DIR = "models/xgb_dl_models"

HORIZONS = [1, 3, 6, 12, 24]

# Train/Test split
TRAIN_END = pd.Timestamp("2022-12-31 23:00:00", tz="UTC")
TEST_START = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
TEST_END = pd.Timestamp("2024-12-31 23:00:00", tz="UTC")

# Internal split for tuning only (within TRAIN)
TUNE_TRAIN_END = pd.Timestamp("2021-12-31 23:00:00", tz="UTC")
TUNE_VAL_START = pd.Timestamp("2022-01-01 00:00:00", tz="UTC")
TUNE_VAL_END = pd.Timestamp("2022-12-31 23:00:00", tz="UTC")

# Tuning controls
N_ITER_SEARCH = 20
CV_SPLITS = 3
TUNE_SAMPLE_N = 120_000   # Adjust based on RAM/time constraints
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Param distributions (ca în exemplul tău, ușor restrâns pt viteză)
PARAM_DISTRIBUTIONS = {
    "n_estimators": [300, 600, 1000, 1500],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7, 9],
    "min_child_weight": [1, 3, 5, 7],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [1.0, 1.5, 3.0],
}

EARLY_STOPPING_ROUNDS = 50

# Create all output directories upfront to prevent data loss
print("Creating output directories...")
os.makedirs("train_models/global/runs_info", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
print("All directories created successfully.")

# =========================
# Helpers (identice ca logică cu LR)
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

    # Include raw NO2/PM10 (ca în LR)
    for c in ["NO2", "PM10"]:
        if c in df.columns:
            candidates.append(c)

    # dedup
    candidates = list(dict.fromkeys(candidates))
    return candidates


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
    """Take last n rows by dt_utc (preserves temporal order)."""
    if n is None or len(df) <= n:
        return df.sort_values("dt_utc")
    return df.sort_values("dt_utc").iloc[-n:]


def tune_hyperparameters(
    train_h: pd.DataFrame,
    feature_cols: list[str],
    h: int,
    params_file_path: str = None
) -> tuple[dict, bool, pd.DataFrame, float]:
    """
    Perform hyperparameter tuning using RandomizedSearchCV with TimeSeriesSplit.

    Uses internal temporal split (2021 vs 2022) if enough data is available,
    otherwise falls back to cross-validation on entire training set.

    If params_file_path is provided and contains parameters for this horizon,
    loads them instead of running the expensive hyperparameter search.

    Args:
        train_h: Training data for this horizon
        feature_cols: List of feature column names
        h: Forecast horizon
        params_file_path: Path to JSON file for caching best parameters (optional)

    Returns:
        Tuple of (best_params, use_internal_val, tune_val, tuning_seconds):
            - best_params: Dictionary of best hyperparameters found
            - use_internal_val: Boolean indicating if validation split was used
            - tune_val: Validation dataframe (for early stopping), or empty df if not used
            - tuning_seconds: Time spent on hyperparameter tuning
    """
    # Check if we can load cached parameters
    if params_file_path and Path(params_file_path).exists():
        try:
            with open(params_file_path, "r", encoding="utf-8") as f:
                cached_params = json.load(f)

            horizon_key = f"h{h}"
            if horizon_key in cached_params:
                best_params = cached_params[horizon_key]
                print(f"[Tuning] Loaded cached params for h={h} from {params_file_path}")
                print(f"[Tuning] Cached params: {best_params}")

                # Still need to determine use_internal_val and tune_val for early stopping
                tune_train = train_h[train_h["dt_utc"] <= TUNE_TRAIN_END].copy()
                tune_val = train_h[(train_h["dt_utc"] >= TUNE_VAL_START) & (train_h["dt_utc"] <= TUNE_VAL_END)].copy()
                use_internal_val = (len(tune_train) > 10_000 and len(tune_val) > 10_000)

                if not use_internal_val:
                    tune_val = pd.DataFrame()

                return best_params, use_internal_val, tune_val, 0.0
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[Tuning] Warning: Could not load cached params: {e}. Running HPO...")
    print(f"[Tuning] Starting hyperparameter tuning for h={h}...")
    # Split tuning data: 2021 for training, 2022 for validation
    tune_train = train_h[train_h["dt_utc"] <= TUNE_TRAIN_END].copy()
    tune_val = train_h[(train_h["dt_utc"] >= TUNE_VAL_START) & (train_h["dt_utc"] <= TUNE_VAL_END)].copy()

    # Check if we have enough data for internal validation split
    use_internal_val = (len(tune_train) > 10_000 and len(tune_val) > 10_000)

    if use_internal_val:
        tune_pool = tune_train
        print(f"[Tuning] Using internal split: tune_train<=2021 ({len(tune_train):,}) | tune_val=2022 ({len(tune_val):,})")
    else:
        tune_pool = train_h
        tune_val = pd.DataFrame()  # Empty df if not using internal validation
        print(f"[Tuning] Fallback: using all TRAIN for CV (size={len(train_h):,})")

    # Sample last N rows to maintain temporal order
    tune_pool_s = take_last_n_sorted(tune_pool, TUNE_SAMPLE_N)

    X_tune = tune_pool_s[feature_cols].to_numpy(dtype=np.float32)
    y_tune = tune_pool_s[f"y_h{h}"].to_numpy(dtype=np.float32)

    # Base estimator for search (no early stopping in CV)
    base_est = XGBRegressor(
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

    search = RandomizedSearchCV(
        estimator=base_est,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER_SEARCH,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    t_search0 = time.time()
    search.fit(X_tune, y_tune)
    tuning_seconds = time.time() - t_search0
    print(f"[Tuning] Done in {tuning_seconds:.1f}s | best MAE (CV) = {-search.best_score_:.4f}")

    best_params = search.best_params_
    print(f"[Tuning] Best params for h={h}: {best_params}")

    # Save parameters to file for future use
    if params_file_path:
        try:
            # Load existing params if file exists
            if Path(params_file_path).exists():
                with open(params_file_path, "r", encoding="utf-8") as f:
                    all_params = json.load(f)
            else:
                all_params = {}

            # Update with new horizon params
            all_params[f"h{h}"] = best_params

            # Save back to file
            with open(params_file_path, "w", encoding="utf-8") as f:
                json.dump(all_params, f, indent=2)

            print(f"[Tuning] Saved params for h={h} to {params_file_path}")
        except Exception as e:
            print(f"[Tuning] Warning: Could not save params to file: {e}")

    return best_params, use_internal_val, tune_val, tuning_seconds


# =========================
# MAIN
# =========================
def main():
    run_t0 = time.time()
    print(">>> [XGB Tuned] Loading CSV...")
    df = pd.read_csv(CSV_PATH, low_memory=False)

    # Build targets
    print(">>> Building multi-horizon targets...")
    df = build_multi_horizon_targets_like_lstm(df, HORIZONS)
    df = make_station_key(df)

    # metadata pt raportare
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

    # Feature selection and data cleaning
    feature_cols = select_feature_columns(df)
    df_model = df.dropna(subset=feature_cols).copy()

    # Temporal split
    df_model["dt_utc"] = pd.to_datetime(df_model["dt_utc"], utc=True)
    train_mask = df_model["dt_utc"] <= TRAIN_END
    test_mask = (df_model["dt_utc"] >= TEST_START) & (df_model["dt_utc"] <= TEST_END)

    train_df_base = df_model.loc[train_mask].copy()
    test_df_base = df_model.loc[test_mask].copy()

    print(f"Base Train (features valid): {len(train_df_base):,} | Base Test: {len(test_df_base):,}")
    print(f"Features used: {len(feature_cols)}")
    print("-" * 70)

    best_params_by_h = {}
    per_h_info = {}
    all_results = []

    for h in HORIZONS:
        print(f"\n================= HORIZON h={h} =================")

        # Valid subsets per horizon
        train_h = train_df_base.dropna(subset=[f"y_h{h}"]).copy()
        test_h = test_df_base.dropna(subset=[f"y_h{h}"]).copy()
        train_h = filter_df_by_anchors(train_h, train_anchors_by_h.get(h))
        test_h = filter_df_by_anchors(test_h, test_anchors_by_h.get(h))

        if len(train_h) == 0 or len(test_h) == 0:
            print(f"Skipping h={h} (no data)")
            continue

        # -------------------------
        # 1) TUNING (doar în TRAIN, cu split intern 2021 vs 2022)
        # -------------------------
        best_params, use_internal_val, tune_val, tuning_seconds = tune_hyperparameters(
            train_h, feature_cols, h, params_file_path=OUTPUT_BEST_PARAMS_JSON
        )
        best_params_by_h[f"h{h}"] = best_params

        # -------------------------
        # 2) TRAIN FINAL (pe TRAIN complet <=2022) + EARLY STOPPING (pe 2022 dacă există)
        # -------------------------
        X_train_full = train_h[feature_cols].to_numpy(dtype=np.float32)
        y_train_full = train_h[f"y_h{h}"].to_numpy(dtype=np.float32)

        # early stopping eval set
        eval_set = None
        if use_internal_val:
            X_val = tune_val[feature_cols].to_numpy(dtype=np.float32)
            y_val = tune_val[f"y_h{h}"].to_numpy(dtype=np.float32)
            eval_set = [(X_val, y_val)]

        final_model = XGBRegressor(
            **best_params,
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS if eval_set is not None else None,
        )

        t_fit0 = time.time()
        if eval_set is not None:
            final_model.fit(X_train_full, y_train_full, eval_set=eval_set, verbose=False)
            best_iter = getattr(final_model, "best_iteration", None)
            if best_iter is not None:
                print(f"[Train] Early-stopped at best_iteration={best_iter}")
        else:
            final_model.fit(X_train_full, y_train_full, verbose=False)
        fit_seconds = time.time() - t_fit0
        print(f"[Train] Fit done in {fit_seconds:.1f}s | train_n={len(train_h):,}")

        # -------------------------
        # 3) TEST EVAL (identic LR ca structură)
        # -------------------------
        X_test = test_h[feature_cols].to_numpy(dtype=np.float32)
        y_test = test_h[f"y_h{h}"].to_numpy(dtype=np.float32)
        y_pred = final_model.predict(X_test)

        res_h = test_h[["Country", "SiteNumber", "StationArea_Label"]].copy()
        res_h["dt_utc"] = test_h["dt_utc"]
        res_h["horizon"] = h
        res_h["y_true"] = y_test
        res_h["y_pred"] = y_pred
        all_results.append(res_h)

        # Save trained model
        model_path = os.path.join(OUTPUT_MODEL_DIR, f"xgb_h{h}.pkl")
        joblib.dump(final_model, model_path)
        print(f"  Model saved to: {model_path}")

        # Store run info
        per_h_info[f"h{h}"] = {
            "tuning_seconds": round(tuning_seconds, 2),
            "train_seconds": round(fit_seconds, 2),  # Renamed from fit_seconds
            "train_n": int(len(train_h)),
            "val_n": 0,  # XGB uses CV during HPO, not explicit validation in final training
            "test_n": int(len(test_h)),
            "model_path": model_path,
        }

        print(f"[Test] Done | test_n={len(test_h):,}")

        # Memory cleanup
        del train_h, test_h, X_train_full, y_train_full, X_test, y_test, y_pred, final_model
        gc.collect()

    # =========================
    # METRICS (identic LR)
    # =========================
    if not all_results:
        print("No results generated.")
        return

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

    print("\n=== Sample of Results (Global XGB Tuned) ===")
    print(final_report[final_report["Scope"] == "Global"].sort_values("horizon"))

    final_report.to_csv(OUTPUT_METRICS_PATH, index=False)
    print(f"\nDetailed metrics saved to: {OUTPUT_METRICS_PATH}")

    full_results_df.head(10000).to_csv(OUTPUT_PREDS_PATH, index=False)
    print(f"Predictions sample saved to: {OUTPUT_PREDS_PATH}")

    with open(OUTPUT_BEST_PARAMS_JSON, "w", encoding="utf-8") as f:
        json.dump(best_params_by_h, f, indent=2)
    print(f"Best params saved to: {OUTPUT_BEST_PARAMS_JSON}")

    # Save run info
    run_info = {
        "run_seconds": round(time.time() - run_t0, 2),
        "tune_sample_n": int(TUNE_SAMPLE_N),
        "n_iter_search": int(N_ITER_SEARCH),
        "cv_splits": int(CV_SPLITS),
        "early_stopping_rounds": int(EARLY_STOPPING_ROUNDS),
        "per_horizon": per_h_info,
    }
    with open(OUTPUT_RUN_INFO_JSON, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    print(f"Run info saved to: {OUTPUT_RUN_INFO_JSON}")

    print(f"\nDone in {time.time() - run_t0:.1f}s")


if __name__ == "__main__":
    main()
