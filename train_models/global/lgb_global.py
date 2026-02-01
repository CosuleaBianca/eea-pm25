import os
import json
import time
import pickle
import gc
import random
import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterSampler

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'train_models'))


# =========================
# CONFIG
# =========================
CSV_PATH = "ml_ready_dataset_full_realistic.csv"

OUTPUT_METRICS_PATH = "results/lgb_metrics_tuned.csv"
OUTPUT_PREDS_PATH = "results/lgb_predictions_sample.csv"
OUTPUT_BEST_PARAMS_JSON = "train_models/global/lgb_best_params_by_horizon.json"
OUTPUT_RUN_INFO_JSON = "train_models/global/runs_info/lgb_run_info.json"

MODEL_DIR = "models/models_lgb"

# Create all output directories upfront to prevent data loss
print("Creating output directories...")
os.makedirs("train_models/global/runs_info", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
print("All directories created successfully.")

HORIZONS = [1, 3, 6, 12, 24]

# Train/Test
TRAIN_END = pd.Timestamp("2022-12-31 23:00:00", tz="UTC")
TEST_START = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
TEST_END = pd.Timestamp("2024-12-31 23:00:00", tz="UTC")

# Internal split for tuning (within TRAIN)
TUNE_TRAIN_END = pd.Timestamp("2021-12-31 23:00:00", tz="UTC")
TUNE_VAL_START = pd.Timestamp("2022-01-01 00:00:00", tz="UTC")
TUNE_VAL_END = pd.Timestamp("2022-12-31 23:00:00", tz="UTC")

# Speed/memory optimization parameters
TUNE_TRAIN_SAMPLE_N = 120_000
TUNE_VAL_SAMPLE_N = 120_000
N_CANDIDATES = 10
RANDOM_STATE = 42

# Early stopping
EARLY_STOPPING_ROUNDS = 200

# Set large n_estimators and let early stopping decide best_iteration
MAX_ESTIMATORS = 20_000

# Param distributions (fără n_estimators)
PARAM_DISTRIBUTIONS = {
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "num_leaves": [31, 63, 127, 255],
    "max_depth": [-1, 8, 12, 16],
    "min_child_samples": [10, 20, 50, 100],     # am adăugat 10 pt mai multă flexibilitate
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [0.0, 0.5, 1.0, 2.0, 5.0],   # am adăugat 0.0
}

# LGB speed / stability
LGB_COMMON_KWARGS = dict(
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1,
    force_col_wise=True,  # reduce overhead-ul de auto-choosing
)

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

    for (_, _), g in df.groupby(["Country", "SiteNumber"], sort=False):
        g = g.sort_values("dt_utc")
        idx_orig = g["dt_utc"]
        row_ids = g["_row_id"].values

        g_idx = g.set_index("dt_utc")

        # FIX: "H" -> "h"
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

    # include raw NO2/PM10 ca în LR
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

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    return pd.Series({
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
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


def fit_and_score_candidate(params: dict, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray) -> tuple[float, int]:
    """
    Train cu early stopping și returnează (MAE pe val, best_iteration).
    """
    model = lgb.LGBMRegressor(
        n_estimators=MAX_ESTIMATORS,
        **params,
        **LGB_COMMON_KWARGS
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)]
    )

    best_iter = getattr(model, "best_iteration_", None)
    preds = model.predict(X_val)
    mae = float(mean_absolute_error(y_val, preds))
    return mae, int(best_iter) if best_iter is not None else MAX_ESTIMATORS


def tune_hyperparameters(train_df_base: pd.DataFrame, feature_cols: list[str],
                         horizons: list[int], sampler: list) -> dict:
    """
    Perform hyperparameter optimization for each horizon.
    Returns a dictionary mapping horizon names (e.g., 'h1') to best parameters.
    """
    print("\n" + "="*70)
    print(">>> Starting Hyperparameter Optimization")
    print("="*70)

    best_params_by_h = {}

    for h in horizons:
        print(f"\n================= TUNING HORIZON h={h} =================")

        train_h = train_df_base.dropna(subset=[f"y_h{h}"]).copy()

        if len(train_h) == 0:
            print(f"Skipping h={h} (no data)")
            continue

        # split tuning (strict temporal)
        tune_train = train_h[train_h["dt_utc"] <= TUNE_TRAIN_END].copy()
        tune_val = train_h[(train_h["dt_utc"] >= TUNE_VAL_START) & (train_h["dt_utc"] <= TUNE_VAL_END)].copy()

        use_internal_val = (len(tune_train) > 10_000 and len(tune_val) > 10_000)
        if not use_internal_val:
            # fallback: dacă nu ai val, nu are sens early stopping pe tuning => sari direct la train pe full
            print(f"[Tuning] Not enough internal val data for h={h}. Using default params.")
            best_params = dict(
                learning_rate=0.05,
                num_leaves=127,
                max_depth=-1,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.0,
                reg_lambda=1.0,
            )
            best_params_by_h[f"h{h}"] = best_params
        else:
            print(f"[Tuning] Holdout: train<=2021 ({len(tune_train):,}) | val=2022 ({len(tune_val):,})")

            # Sample for tuning (last N)
            tune_train_s = take_last_n_sorted(tune_train, TUNE_TRAIN_SAMPLE_N)
            tune_val_s = take_last_n_sorted(tune_val, TUNE_VAL_SAMPLE_N)

            X_tune_tr = tune_train_s[feature_cols].to_numpy(dtype=np.float32)
            y_tune_tr = tune_train_s[f"y_h{h}"].to_numpy(dtype=np.float32)
            X_tune_va = tune_val_s[feature_cols].to_numpy(dtype=np.float32)
            y_tune_va = tune_val_s[f"y_h{h}"].to_numpy(dtype=np.float32)

            # Evaluate candidates (N_CANDIDATES fits, each with early stopping)
            t_search0 = time.time()
            best_mae = float("inf")
            best_params = None
            best_iter = None

            for i, params in enumerate(sampler, start=1):
                mae, bi = fit_and_score_candidate(params, X_tune_tr, y_tune_tr, X_tune_va, y_tune_va)
                print(f"[Tuning] cand {i}/{N_CANDIDATES} | MAE={mae:.4f} | best_iter={bi} | {params}")

                if mae < best_mae:
                    best_mae = mae
                    best_params = params
                    best_iter = bi

            tune_sec = time.time() - t_search0
            print(f"[Tuning] Done in {tune_sec:.1f}s | best MAE (holdout 2022) = {best_mae:.4f}")
            print(f"[Tuning] Best params for h={h}: {best_params} | best_iter={best_iter}")

            best_params_by_h[f"h{h}"] = best_params

    print("\n" + "="*70)
    print(">>> Hyperparameter Optimization Complete")
    print("="*70 + "\n")

    return best_params_by_h


# =========================
# MAIN
# =========================
def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    run_t0 = time.time()
    print(">>> [LGB Fast Tuned] Loading CSV...")
    df = pd.read_csv(CSV_PATH, low_memory=False)

    print(">>> Building multi-horizon targets...")
    df = build_multi_horizon_targets(df, HORIZONS)

    df = recover_categorical_metadata(df)

    feature_cols = select_feature_columns(df)

    # Clean inf/-inf before dropna
    df = df.replace([np.inf, -np.inf], np.nan)

    # dropna only on features (targets filtered per-horizon)
    df_model = df.dropna(subset=feature_cols).copy()
    df_model["dt_utc"] = pd.to_datetime(df_model["dt_utc"], utc=True)

    train_mask = df_model["dt_utc"] <= TRAIN_END
    test_mask = (df_model["dt_utc"] >= TEST_START) & (df_model["dt_utc"] <= TEST_END)

    train_df_base = df_model.loc[train_mask].copy()
    test_df_base = df_model.loc[test_mask].copy()

    print(f"Base Train (features valid): {len(train_df_base):,} | Base Test: {len(test_df_base):,}")
    print(f"Features used: {len(feature_cols)}")
    print("-" * 70)

    # =========================
    # Check if hyperparameters already exist, otherwise run HPO
    # =========================
    if os.path.exists(OUTPUT_BEST_PARAMS_JSON):
        print(f"\n>>> Loading existing hyperparameters from: {OUTPUT_BEST_PARAMS_JSON}")
        with open(OUTPUT_BEST_PARAMS_JSON, "r", encoding="utf-8") as f:
            best_params_by_h = json.load(f)
        print(f">>> Loaded parameters for horizons: {list(best_params_by_h.keys())}")
    else:
        print(f"\n>>> No existing hyperparameters found at: {OUTPUT_BEST_PARAMS_JSON}")
        print(">>> Running hyperparameter optimization...")
        # pregătim candidații o singură dată (reproducibil)
        sampler = list(ParameterSampler(PARAM_DISTRIBUTIONS, n_iter=N_CANDIDATES, random_state=RANDOM_STATE))
        best_params_by_h = tune_hyperparameters(train_df_base, feature_cols, HORIZONS, sampler)

    per_h_info = {}
    all_results = []

    # =========================
    # Train models for each horizon
    # =========================
    for h in HORIZONS:
        h_t0 = time.time()
        print(f"\n================= HORIZON h={h} =================")

        train_h = train_df_base.dropna(subset=[f"y_h{h}"]).copy()
        test_h = test_df_base.dropna(subset=[f"y_h{h}"]).copy()

        if len(train_h) == 0 or len(test_h) == 0:
            print(f"Skipping h={h} (no data)")
            continue

        # Get best params for this horizon
        best_params = best_params_by_h.get(f"h{h}")
        if best_params is None:
            print(f"WARNING: No hyperparameters found for h={h}, using defaults")
            best_params = dict(
                learning_rate=0.05,
                num_leaves=127,
                max_depth=-1,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.0,
                reg_lambda=1.0,
            )

        print(f"[Train] Using params: {best_params}")

        # split tuning (strict temporal) - needed for early stopping validation
        tune_train = train_h[train_h["dt_utc"] <= TUNE_TRAIN_END].copy()
        tune_val = train_h[(train_h["dt_utc"] >= TUNE_VAL_START) & (train_h["dt_utc"] <= TUNE_VAL_END)].copy()
        use_internal_val = (len(tune_train) > 10_000 and len(tune_val) > 10_000)

        # =========================
        # Train final pe tot TRAIN (<=2022) cu early stopping pe 2022 (dacă există)
        # =========================
        X_train_full = train_h[feature_cols].to_numpy(dtype=np.float32)
        y_train_full = train_h[f"y_h{h}"].to_numpy(dtype=np.float32)

        eval_set = None
        if use_internal_val:
            X_val = tune_val[feature_cols].to_numpy(dtype=np.float32)
            y_val = tune_val[f"y_h{h}"].to_numpy(dtype=np.float32)
            eval_set = [(X_val, y_val)]

        final_model = lgb.LGBMRegressor(
            n_estimators=MAX_ESTIMATORS,
            **best_params,
            **LGB_COMMON_KWARGS
        )

        t_fit0 = time.time()
        if eval_set is not None:
            final_model.fit(
                X_train_full, y_train_full,
                eval_set=eval_set,
                eval_metric="l1",
                callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)]
            )
            best_iter_final = getattr(final_model, "best_iteration_", None)
            if best_iter_final is not None:
                print(f"[Train] Early-stopped best_iteration_={best_iter_final}")
        else:
            final_model.fit(X_train_full, y_train_full)
        fit_sec = time.time() - t_fit0

        print(f"[Train] Fit done in {fit_sec:.1f}s | train_n={len(train_h):,}")

        # save model
        model_path = os.path.join(MODEL_DIR, f"lgb_h{h}.pkl")
        save_pickle(final_model, model_path)
        print(f"[Save] Model saved: {model_path}")

        # test predict
        X_test = test_h[feature_cols].to_numpy(dtype=np.float32)
        y_test = test_h[f"y_h{h}"].to_numpy(dtype=np.float32)
        y_pred = final_model.predict(X_test).astype(np.float32)

        res_h = test_h[["Country", "SiteNumber", "StationArea_Label"]].copy()
        res_h["dt_utc"] = test_h["dt_utc"]
        res_h["horizon"] = h
        res_h["y_true"] = y_test
        res_h["y_pred"] = y_pred
        all_results.append(res_h)

        per_h_info[f"h{h}"] = {
            "train_seconds": round(fit_sec, 2),  # Renamed from fit_seconds
            "train_n": int(len(train_h)),
            "val_n": 0,  # LGB uses early stopping during HPO, not explicit validation in final training
            "test_n": int(len(test_h)),
            "model_path": model_path,
        }

        print(f"[Test] Done | test_n={len(test_h):,} | horizon_total_sec={time.time()-h_t0:.1f}s")

        # Memory cleanup
        del train_h, test_h, X_train_full, y_train_full, X_test, y_test, y_pred, final_model
        gc.collect()

    if not all_results:
        print("No results generated.")
        return

    # =========================
    # METRICS
    # =========================
    full_results_df = pd.concat(all_results, axis=0, ignore_index=True)

    print("\n=== Calculating Granular Metrics ===")
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

    print("\n=== Sample of Results (Global LGB Fast Tuned) ===")
    print(final_report[final_report["Scope"] == "Global"].sort_values("horizon"))

    final_report.to_csv(OUTPUT_METRICS_PATH, index=False)
    print(f"\nDetailed metrics saved to: {OUTPUT_METRICS_PATH}")

    full_results_df.head(10000).to_csv(OUTPUT_PREDS_PATH, index=False)
    print(f"Predictions sample saved to: {OUTPUT_PREDS_PATH}")

    with open(OUTPUT_BEST_PARAMS_JSON, "w", encoding="utf-8") as f:
        json.dump(best_params_by_h, f, indent=2)
    print(f"Best params saved to: {OUTPUT_BEST_PARAMS_JSON}")

    run_info = {
        "run_seconds": round(time.time() - run_t0, 2),
        "tune_train_sample_n": int(TUNE_TRAIN_SAMPLE_N),
        "tune_val_sample_n": int(TUNE_VAL_SAMPLE_N),
        "n_candidates": int(N_CANDIDATES),
        "early_stopping_rounds": int(EARLY_STOPPING_ROUNDS),
        "max_estimators": int(MAX_ESTIMATORS),
        "per_horizon": per_h_info,
    }
    with open(OUTPUT_RUN_INFO_JSON, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    print(f"Run info saved to: {OUTPUT_RUN_INFO_JSON}")

    print(f"\nDone in {time.time() - run_t0:.1f}s")


if __name__ == "__main__":
    main()
