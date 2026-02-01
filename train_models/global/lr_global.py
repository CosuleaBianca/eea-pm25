import gc
import json
import random
import pandas as pd
import numpy as np
import time
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'train_models'))

# Deterministic seed for any random operations (e.g., data splitting)
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

CSV_PATH = "ml_ready_dataset_full_realistic.csv"
OUTPUT_METRICS_PATH = "results/lr_metrics.csv"
OUTPUT_PREDS_PATH = "results/lr_predictions_sample.csv"
OUTPUT_RUN_INFO_JSON = "train_models/global/runs_info/lr_run_info.json"
OUTPUT_MODEL_DIR = "models/lr_models"

HORIZONS = [1, 3, 6, 12, 24]

TRAIN_END = pd.Timestamp("2022-12-31 23:00:00", tz="UTC")
TEST_START = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
TEST_END = pd.Timestamp("2024-12-31 23:00:00", tz="UTC")

# Create all output directories upfront to prevent data loss
print("Creating output directories...")
os.makedirs("train_models/global/runs_info", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
print("All directories created successfully.")

def build_multi_horizon_targets(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """
    Builds PM2.5 targets for multiple horizons per station.
    Same logic as before.
    """
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
        if c in exclude: continue
        if c.startswith("y_h"): continue
        if pd.api.types.is_numeric_dtype(df[c]):
            candidates.append(c)
    
    # Optional: include raw values if needed
    for c in ["NO2", "PM10"]:
        if c in df.columns: candidates.append(c)
    return candidates

def recover_categorical_metadata(df):
    """
    Helper to reconstruct readable 'StationArea' from One-Hot columns 
    for the final report.
    """
    # Define prefixes to look for
    area_cols = [c for c in df.columns if c.startswith("StationArea_")]
    
    if not area_cols:
        return df
    
    # Create a single column taking the name of the column with max value (1)
    # We strip 'StationArea_' prefix for cleaner charts
    df['StationArea_Label'] = df[area_cols].idxmax(axis=1).apply(lambda x: x.replace('StationArea_', ''))
    
    return df

def calculate_metrics(sub_df):
    """
    Computes metrics for a given subset of data.
    """
    y_true = sub_df['y_true']
    y_pred = sub_df['y_pred']
    
    if len(y_true) == 0:
        return pd.Series({'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'Bias': np.nan, 'Count': 0})

    return pd.Series({
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'Bias': np.mean(y_pred - y_true), # Positive = Model Overestimates
        'Count': len(y_true)
    })


def main():
    run_t0 = time.time()
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    # 1. Build Targets
    print("Building multi-horizon targets...")
    df = build_multi_horizon_targets(df, HORIZONS)

    # 2. Recover Metadata
    df = recover_categorical_metadata(df)

    # 3. Clean Data - LAZY DROPPING PREPARATION
    feature_cols = select_feature_columns(df)

    # Delete only if input (X) is missing. Handle output (y) dynamically.
    df_model = df.dropna(subset=feature_cols).copy()

    # 4. Temporal Split
    df_model["dt_utc"] = pd.to_datetime(df_model["dt_utc"], utc=True)
    train_mask = df_model["dt_utc"] <= TRAIN_END
    test_mask = (df_model["dt_utc"] >= TEST_START) & (df_model["dt_utc"] <= TEST_END)

    train_df = df_model.loc[train_mask]
    test_df = df_model.loc[test_mask]


    print(f"Base Train samples (features valid): {len(train_df):,} | Base Test samples: {len(test_df):,}")
    print("-" * 60)

    # =========================================================================
    # DIRECT STRATEGY LOOP
    # =========================================================================

    per_h_info = {}
    all_results = []

    for h in HORIZONS:
        print(f"\nTraining Direct Model for Horizon h={h}...")
        start_time = time.time()

        # Select valid subset ONLY for horizon h
        train_h_subset = train_df.dropna(subset=[f"y_h{h}"])
        test_h_subset = test_df.dropna(subset=[f"y_h{h}"])

        if len(train_h_subset) == 0:
            print(f"Skipping h={h} (no data)")
            continue

        # Extract numpy matrices from filtered subsets
        X_train_h = train_h_subset[feature_cols].to_numpy()
        y_train_h = train_h_subset[f"y_h{h}"].to_numpy()
        
        X_test_h = test_h_subset[feature_cols].to_numpy()
        y_test_h = test_h_subset[f"y_h{h}"].to_numpy()
        
        # Initialize and Train Separate Model
        model = LinearRegression(n_jobs=-1)

        model.fit(X_train_h, y_train_h)

        # Predict
        y_pred_h = model.predict(X_test_h)

        elapsed = time.time() - start_time
        print(f"  > Done in {elapsed:.2f}s | Train Size: {len(y_train_h)}")

        # Store predictions
        res_h = test_h_subset[['Country', 'SiteNumber', 'StationArea_Label']].copy()
        res_h['dt_utc'] = test_h_subset['dt_utc']
        res_h['horizon'] = h
        res_h['y_true'] = y_test_h
        res_h['y_pred'] = y_pred_h

        all_results.append(res_h)

        # Save model for this horizon
        model_path = os.path.join(OUTPUT_MODEL_DIR, f"lr_h{h}.pkl")
        joblib.dump(model, model_path)
        print(f"  Model saved to: {model_path}")

        # Store run info
        per_h_info[f"h{h}"] = {
            "train_seconds": round(elapsed, 2),  # Renamed from fit_seconds
            "train_n": int(len(y_train_h)),
            "val_n": 0,  # LR doesn't use validation
            "test_n": int(len(y_test_h)),
            "model_path": model_path,
        }
        
        del train_h_subset, test_h_subset, X_train_h, y_train_h, model
        gc.collect() # Force-release memory back to the OS

    # Combine all horizons
    if not all_results:
        print("No results generated!")
        return

    full_results_df = pd.concat(all_results, axis=0, ignore_index=True)

    print("\n=== Calculating Granular Metrics ===")

    # 1. Global Metrics
    global_metrics = full_results_df.groupby('horizon').apply(calculate_metrics, include_groups=False).reset_index()
    global_metrics['Scope'] = 'Global'
    global_metrics['Group'] = 'All'
    country_metrics = full_results_df.groupby(['horizon', 'Country']).apply(calculate_metrics, include_groups=False).reset_index()
    country_metrics['Scope'] = 'Country'
    country_metrics = country_metrics.rename(columns={'Country': 'Group'})
    
    area_metrics = full_results_df.groupby(['horizon', 'StationArea_Label']).apply(calculate_metrics, include_groups=False).reset_index()
    area_metrics['Scope'] = 'StationArea'
    area_metrics = area_metrics.rename(columns={'StationArea_Label': 'Group'})

    final_report = pd.concat([global_metrics, country_metrics, area_metrics], ignore_index=True)
    cols_order = ['Scope', 'Group', 'horizon', 'MAE', 'RMSE', 'R2', 'Bias', 'Count']
    final_report = final_report[cols_order]

    print("\n=== Sample of Results (Global) ===")
    print(final_report[final_report['Scope'] == 'Global'])

    final_report.to_csv(OUTPUT_METRICS_PATH, index=False)
    print(f"\nDetailed metrics saved to: {OUTPUT_METRICS_PATH}")

    full_results_df.head(10000).to_csv(OUTPUT_PREDS_PATH, index=False)
    print(f"Predictions sample saved to: {OUTPUT_PREDS_PATH}")

    # Save run info
    run_info = {
        "run_seconds": round(time.time() - run_t0, 2),
        "per_horizon": per_h_info,
    }
    with open(OUTPUT_RUN_INFO_JSON, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    print(f"Run info saved to: {OUTPUT_RUN_INFO_JSON}")

    print(f"\nDone in {time.time() - run_t0:.1f}s")

if __name__ == "__main__":
    main()