import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import time

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

# Standardized output paths
OUTPUT_METRICS_PATH = "results/persistence_dl_metrics.csv"
OUTPUT_PREDS_PATH = "results/persistence_dl_predictions_sample.csv"
RUN_INFO_PATH = "train_models/global/runs_info/persistence_dl_run_info.json"

HORIZONS = [1, 3, 6, 12, 24]

# Train/Test split definition
TRAIN_END = pd.Timestamp("2022-12-31 23:00:00", tz="UTC")
TEST_START = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
TEST_END = pd.Timestamp("2024-12-31 23:00:00", tz="UTC")

# Select baselines to evaluate
BASELINES_TO_RUN = [
    "persistence",     # yhat(t+h) = PM2.5(t)
]

# =========================
# Helpers
# =========================
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
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse,
        "R2": float(r2_score(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
        "Count": int(len(y_true)),
    })


def build_targets_and_predictions_for_group(g: pd.DataFrame, horizon: int, baseline_name: str) -> pd.DataFrame:
    """
    g: data for a (Country, SiteNumber), sorted by dt_utc.
    Returns a DF with: dt_utc, y_true, y_pred, horizon
    for the chosen baseline and single horizon.
    """
    g = g.copy()
    g = g.sort_values("dt_utc")

    # Use asfreq('h') to align to hourly grid; then map back to original timestamps
    idx_orig = g["dt_utc"]
    g_idx = g.set_index("dt_utc")
    g_hourly = g_idx.asfreq("h")

    y_true = g_hourly["PM2.5"].shift(-horizon)  # PM2.5(t+h)

    if baseline_name != "persistence":
        raise ValueError(f"Unknown baseline_name: {baseline_name}")

    y_pred = g_hourly["PM2.5"]         # PM2.5(t)

    tmp = pd.DataFrame({
        "dt_utc": g_hourly.index,
        "horizon": horizon,
        "y_true": y_true,
        "y_pred": y_pred,
    })

    # Map to original timestamps (to maintain same support as original dataset)
    tmp = tmp.loc[idx_orig].copy()

    return tmp


def main():
    # Create output directories upfront to prevent data loss
    os.makedirs("results", exist_ok=True)
    os.makedirs("train_models/global/runs_info", exist_ok=True)

    run_start = time.time()
    run_info = {
        "model_name": "persistence",
        "horizons": HORIZONS,
        "per_horizon": {}
    }

    print(">>> [Baseline] Loading CSV...")
    df = pd.read_csv(CSV_PATH, low_memory=False)

    # basic cleanup / parsing
    df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["dt_utc", "Country", "SiteNumber", "PM2.5"]).copy()
    df = df.sort_values(["Country", "SiteNumber", "dt_utc"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["Country", "SiteNumber", "dt_utc"], keep="first")

    df = make_station_key(df)
    df = recover_categorical_metadata(df)
    df = normalize_season_column(df)

    # LSTM-aligned anchors (sampling uses LSTM feature set + lookback)
    df_sampling = build_multi_horizon_targets_like_lstm(df, HORIZONS)
    df_sampling = normalize_season_column(df_sampling)
    lstm_feature_cols = select_feature_columns_like_lstm_script(df_sampling)
    train_anchors_by_h, test_anchors_by_h, common_stations, scaler_stations = build_lstm_aligned_anchors(
        df=df_sampling,
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

    # Test window
    test_df = df[(df["dt_utc"] >= TEST_START) & (df["dt_utc"] <= TEST_END)].copy()
    print(f"Test rows (raw): {len(test_df):,}")

    all_preds = []

    for baseline_name in BASELINES_TO_RUN:
        print(f"\n=== Running baseline: {baseline_name} ===")

        # Process each horizon separately to track timing
        for h in HORIZONS:
            h_start = time.time()
            print(f"  Processing horizon {h}h...")

            # Generate predictions per station for this horizon
            for (country, site), g in test_df.groupby(["Country", "SiteNumber"], sort=False):
                pred_g = build_targets_and_predictions_for_group(g, h, baseline_name)

                pred_g["Country"] = country
                pred_g["SiteNumber"] = site
                pred_g["StationArea_Label"] = g["StationArea_Label"].iloc[0]
                pred_g["station_key"] = g["station_key"].iloc[0]

                pred_g["baseline"] = baseline_name
                all_preds.append(pred_g)

            h_time = time.time() - h_start
            run_info["per_horizon"][str(h)] = {
                "horizon_seconds": h_time
            }
            print(f"  Horizon {h}h completed in {h_time:.2f}s")

    if not all_preds:
        print("No predictions generated.")
        return

    preds_df = pd.concat(all_preds, axis=0, ignore_index=True)

    aligned_preds = []
    for h in HORIZONS:
        pred_h = preds_df[preds_df["horizon"] == h].copy()
        pred_h = filter_df_by_anchors(pred_h, test_anchors_by_h.get(h))
        aligned_preds.append(pred_h)

    preds_df = pd.concat(aligned_preds, axis=0, ignore_index=True)

    # drop rows where either y_true or y_pred missing
    preds_df = preds_df.dropna(subset=["y_true", "y_pred"]).copy()

    print(f"\nPred rows after dropna(y_true,y_pred): {len(preds_df):,}")

    # =========================
    # METRICS
    # =========================
    print("\n=== Calculating Metrics ===")

    # Global per baseline & horizon
    global_metrics = (
        preds_df.groupby(["baseline", "horizon"])
        .apply(calculate_metrics)
        .reset_index()
    )
    global_metrics["Scope"] = "Global"
    global_metrics["Group"] = "All"

    # Country
    country_metrics = (
        preds_df.groupby(["baseline", "horizon", "Country"])
        .apply(calculate_metrics)
        .reset_index()
        .rename(columns={"Country": "Group"})
    )
    country_metrics["Scope"] = "Country"

    # StationArea
    area_metrics = (
        preds_df.groupby(["baseline", "horizon", "StationArea_Label"])
        .apply(calculate_metrics)
        .reset_index()
        .rename(columns={"StationArea_Label": "Group"})
    )
    area_metrics["Scope"] = "StationArea"

    final_report = pd.concat([global_metrics, country_metrics, area_metrics], ignore_index=True)
    final_report = final_report[["baseline", "Scope", "Group", "horizon", "MAE", "RMSE", "R2", "Bias", "Count"]]

    print("\n=== Global metrics (sample) ===")
    print(final_report[final_report["Scope"] == "Global"].sort_values(["baseline", "horizon"]))

    final_report.to_csv(OUTPUT_METRICS_PATH, index=False)
    print(f"\nMetrics saved to: {OUTPUT_METRICS_PATH}")

    preds_df.head(10000).to_csv(OUTPUT_PREDS_PATH, index=False)
    print(f"Predictions sample saved to: {OUTPUT_PREDS_PATH}")

    # Add test sample counts to per-horizon information
    for h in HORIZONS:
        h_data = preds_df[preds_df["horizon"] == h]
        run_info["per_horizon"][str(h)]["test_n"] = int(len(h_data))

    # Save run info
    run_info["run_seconds"] = time.time() - run_start
    with open(RUN_INFO_PATH, "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"\nRun info saved to: {RUN_INFO_PATH}")


if __name__ == "__main__":
    main()
