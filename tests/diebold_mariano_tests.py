"""
Diebold-Mariano Test for Model Comparison

Performs pairwise statistical tests for equal predictive accuracy between models.
Uses squared error loss and Newey-West HAC variance estimator.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'train_models'))

from dl_sampling_utils import (
    build_lstm_aligned_anchors,
    select_feature_columns_like_lstm_script,
    build_multi_horizon_targets_like_lstm,
    make_station_key,
    filter_df_by_anchors,
    normalize_season_column,
    recover_station_area_label
)

# =========================
# CONFIG
# =========================
CSV_PATH = "../ml_ready_dataset_full_realistic.csv"
OUTPUT_DIR = "../results"
FIGURES_DIR = "../figures"

HORIZONS = [1, 3, 6, 12, 24]
TRAIN_END = pd.Timestamp("2022-12-31 23:00:00", tz="UTC")
TEST_START = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
TEST_END = pd.Timestamp("2024-12-31 23:00:00", tz="UTC")

# Protocol definitions
PROTOCOL_A_MODELS = ["xgb", "lgb"]  # GAM models not saved for Protocol A
PROTOCOL_A_HORIZONS = [1, 12, 24]

PROTOCOL_B_MODELS = ["xgb_dl", "lgb_dl", "gam_dl", "lstm_attention", "lstm_residual"]
PROTOCOL_B_HORIZONS = [1, 3, 6, 12, 24]

# Limit predictions to avoid extremely long runtime
# Statistical tests are still valid with large samples (1K is sufficient for DM test)
MAX_PREDICTIONS_PER_HORIZON = 1000  # Limit to 1K predictions for speed

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 80)
print("Diebold-Mariano Model Comparison Tests")
print("=" * 80)


# =========================
# Helper Functions
# =========================

def hac_variance(series: np.ndarray, max_lags: Optional[int] = None) -> float:
    """
    Compute HAC variance using Newey-West estimator.

    Args:
        series: Time series of loss differentials
        max_lags: Maximum number of lags for autocorrelation (default: floor(n^0.25))

    Returns:
        HAC variance estimate
    """
    n = len(series)
    if max_lags is None:
        max_lags = min(10, int(np.floor(n ** 0.25)))

    mean_diff = np.mean(series)
    centered = series - mean_diff

    # Variance (lag 0)
    gamma_0 = np.mean(centered ** 2)

    # Autocovariances
    gamma_sum = 0.0
    for lag in range(1, max_lags + 1):
        if lag >= n:
            break
        gamma_lag = np.mean(centered[lag:] * centered[:-lag])
        weight = 1 - lag / (max_lags + 1)  # Bartlett kernel
        gamma_sum += 2 * weight * gamma_lag

    hac_var = (gamma_0 + gamma_sum) / n
    return max(hac_var, 1e-10)  # Prevent division by zero


def compute_dm_statistic(
    errors1: np.ndarray,
    errors2: np.ndarray,
    loss: str = 'squared'
) -> Tuple[float, float]:
    """
    Compute Diebold-Mariano test statistic.

    Args:
        errors1: Prediction errors from model 1 (y_true - y_pred1)
        errors2: Prediction errors from model 2 (y_true - y_pred2)
        loss: Loss function ('squared' for RMSE comparison, 'absolute' for MAE)

    Returns:
        (DM statistic, p-value)
    """
    if len(errors1) != len(errors2):
        raise ValueError("Error arrays must have same length")

    n = len(errors1)
    if n < 2:
        return np.nan, np.nan

    # Compute loss differential
    if loss == 'squared':
        d = errors1 ** 2 - errors2 ** 2
    elif loss == 'absolute':
        d = np.abs(errors1) - np.abs(errors2)
    else:
        raise ValueError(f"Unknown loss function: {loss}")

    mean_d = np.mean(d)

    # Compute HAC variance
    var_d = hac_variance(d)

    # DM statistic
    dm_stat = mean_d / np.sqrt(var_d)

    # Two-tailed t-test (H0: equal predictive accuracy)
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n-1))

    return dm_stat, p_value


def load_model(model_path: str, model_type: str):
    """Load trained model from file."""
    if model_type in ['xgb', 'lgb', 'gam', 'xgb_dl', 'lgb_dl', 'gam_dl']:
        return joblib.load(model_path)
    elif model_type in ['lstm_attention', 'lstm_residual', 'lstm_global']:
        import tensorflow as tf
        # Enable unsafe deserialization for Lambda layers (models trained locally, trusted)
        tf.keras.config.enable_unsafe_deserialization()
        return tf.keras.models.load_model(model_path, compile=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def identify_best_models(
    models: List[str],
    horizon: int,
    protocol: str
) -> Tuple[str, str, float, float]:
    """
    Identify best two models for comparison at given horizon.

    Returns:
        (best_model1, best_model2, rmse1, rmse2)
    """
    results = []

    for model_name in models:
        # Construct metrics file path
        if protocol == 'A':
            metrics_file = f"{OUTPUT_DIR}/{model_name}_metrics_tuned.csv"
        else:  # Protocol B
            if model_name.startswith('lstm'):
                # Handle special naming for lstm_residual
                if model_name == 'lstm_residual':
                    metrics_file = f"{OUTPUT_DIR}/lstm_residual_singleoutput_aligned_metrics.csv"
                else:
                    metrics_file = f"{OUTPUT_DIR}/{model_name}_metrics.csv"
            else:
                # For DL models (xgb_dl, lgb_dl, gam_dl)
                metrics_file = f"{OUTPUT_DIR}/{model_name}_metrics_tuned.csv"

        if not os.path.exists(metrics_file):
            print(f"  WARNING: Metrics file not found: {metrics_file}")
            continue

        # Load metrics
        df_metrics = pd.read_csv(metrics_file)

        # Filter to Global/All scope and target horizon
        mask = (df_metrics['Scope'] == 'Global') & \
               (df_metrics['Group'] == 'All') & \
               (df_metrics['horizon'] == horizon)

        if mask.sum() == 0:
            print(f"  WARNING: No metrics found for {model_name} at horizon {horizon}")
            continue

        rmse = df_metrics.loc[mask, 'RMSE'].values[0]
        results.append((model_name, rmse))

    if len(results) < 2:
        raise ValueError(f"Need at least 2 models for comparison, found {len(results)}")

    # Sort by RMSE (ascending)
    results.sort(key=lambda x: x[1])

    # Return best two models
    best1, rmse1 = results[0]
    best2, rmse2 = results[1]

    return best1, best2, rmse1, rmse2


def generate_predictions_classical(
    model_path: str,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    horizon: int,
    model_type: str
) -> pd.DataFrame:
    """
    Generate predictions for classical/ML models.

    Returns:
        DataFrame with columns: Country, SiteNumber, dt_utc, horizon, y_true, y_pred
    """
    # Load model
    model = load_model(model_path, model_type)

    # Prepare features
    X_test = test_df[feature_cols].values
    y_test = test_df[f'y_h{horizon}'].values

    # Generate predictions
    y_pred = model.predict(X_test)

    # Create result DataFrame
    result = pd.DataFrame({
        'Country': test_df['Country'].values,
        'SiteNumber': test_df['SiteNumber'].values,
        'dt_utc': test_df['dt_utc'].values,
        'horizon': horizon,
        'y_true': y_test,
        'y_pred': y_pred
    })

    # Remove NaN targets
    result = result.dropna(subset=['y_true'])

    return result


def generate_predictions_lstm(
    model_path: str,
    scaler_path: str,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    horizon: int,
    model_type: str,
    lookback_h: int = 168,
    max_predictions: Optional[int] = 20000
) -> pd.DataFrame:
    """
    Generate predictions for LSTM models.

    Returns:
        DataFrame with columns: Country, SiteNumber, dt_utc, horizon, y_true, y_pred
    """
    import tensorflow as tf

    # Enable unsafe deserialization for Lambda layers (models trained locally, trusted)
    tf.keras.config.enable_unsafe_deserialization()

    # Load model and scalers
    model = tf.keras.models.load_model(model_path, compile=False)

    with open(scaler_path, 'r') as f:
        scaler_data = json.load(f)

    scalers = scaler_data['scalers']
    feature_cols_scaler = scaler_data['feature_cols']

    # Build sequences
    from dl_sampling_utils import compute_segments

    test_df = test_df.sort_values(['Country', 'SiteNumber', 'dt_utc']).reset_index(drop=True)
    test_df = make_station_key(test_df)

    # Create station ID mapping (for models with station embeddings)
    unique_stations = sorted(test_df['station_key'].unique())
    station_to_id = {station: idx for idx, station in enumerate(unique_stations)}

    # Check if model expects 2 inputs (sequence + station_id)
    model_expects_station_id = len(model.inputs) == 2

    # Get PM2.5 index for denormalization (model outputs normalized predictions)
    pm25_index = feature_cols_scaler.index('PM2.5')

    # Pre-filter stations: only process stations with sufficient continuous data
    print(f"  Pre-filtering stations for sufficient continuous data (>= {lookback_h + horizon + 1}h sequences)...")
    valid_stations = []
    for station_key, g in test_df.groupby('station_key', sort=False):
        if station_key not in scalers:
            continue
        segs = compute_segments(g, gap_break_hours=2)
        has_valid_segment = any(len(seg) >= lookback_h + horizon + 1 for seg in segs)
        if has_valid_segment:
            valid_stations.append(station_key)

    print(f"  Valid stations: {len(valid_stations)} / {test_df['station_key'].nunique()}")

    # Filter test_df to only valid stations
    test_df = test_df[test_df['station_key'].isin(valid_stations)].copy()
    print(f"  Filtered test set: {len(test_df):,} rows")

    predictions = []
    prediction_count = 0  # Track how many predictions we've made

    for station_key, g in test_df.groupby('station_key', sort=False):
        # Early stopping if we have enough predictions
        if max_predictions is not None and prediction_count >= max_predictions:
            print(f"  Reached max predictions limit ({max_predictions}), stopping early...")
            break
        # Get scaler for this station
        if station_key not in scalers:
            continue

        scaler_info = scalers[station_key]
        means = np.array(scaler_info['mean'])
        stds = np.array(scaler_info['std'])

        # Get PM2.5 normalization params for denormalization
        pm25_mean = means[pm25_index]
        pm25_std = stds[pm25_index]

        # Get station ID for embedding models
        station_id = station_to_id.get(station_key, 0)

        # Segment at gaps
        segs = compute_segments(g, gap_break_hours=2)

        for seg in segs:
            if len(seg) < lookback_h + horizon + 1:
                continue

            seg = seg.sort_values('dt_utc').reset_index(drop=True)

            # Extract features in correct order
            X_raw = seg[feature_cols_scaler].values.astype(np.float32)

            # Normalize
            X_norm = (X_raw - means) / (stds + 1e-8)

            # Get target
            if f'y_h{horizon}' not in seg.columns:
                continue
            y_vals = seg[f'y_h{horizon}'].values

            # Create sequences
            n = len(seg)
            start_t = lookback_h - 1
            end_t = n - horizon

            for t in range(start_t, end_t):
                # Check for NaN in lookback window and target
                x_win = X_norm[t - lookback_h + 1:t + 1, :]
                if np.isnan(x_win).any() or pd.isna(y_vals[t]):
                    continue

                # Make prediction with appropriate inputs
                x_batch = x_win[np.newaxis, :, :]  # Shape: (1, lookback_h, n_features)

                if model_expects_station_id:
                    # Model expects [sequence, station_id]
                    station_id_batch = np.array([[station_id]], dtype=np.int32)
                    y_pred_norm = model.predict([x_batch, station_id_batch], verbose=0)[0, 0]
                else:
                    # Model expects only sequence
                    y_pred_norm = model.predict(x_batch, verbose=0)[0, 0]

                # Denormalize prediction (model outputs normalized values)
                y_pred = y_pred_norm * pm25_std + pm25_mean

                predictions.append({
                    'Country': seg.iloc[t]['Country'],
                    'SiteNumber': seg.iloc[t]['SiteNumber'],
                    'dt_utc': seg.iloc[t]['dt_utc'],
                    'horizon': horizon,
                    'y_true': y_vals[t],
                    'y_pred': y_pred
                })

                prediction_count += 1

                # Progress update every 1000 predictions
                if prediction_count % 1000 == 0:
                    print(f"    Progress: {prediction_count:,} predictions generated...")

                # Early stopping within segment
                if max_predictions is not None and prediction_count >= max_predictions:
                    break

            # Early stopping within station
            if max_predictions is not None and prediction_count >= max_predictions:
                break

        # Early stopping across stations
        if max_predictions is not None and prediction_count >= max_predictions:
            break

    if len(predictions) == 0:
        return pd.DataFrame(columns=['Country', 'SiteNumber', 'dt_utc', 'horizon', 'y_true', 'y_pred'])

    return pd.DataFrame(predictions)


def merge_predictions(
    pred1: pd.DataFrame,
    pred2: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge predictions from two models on same samples.

    Returns:
        (y_true, y_pred1, y_pred2)
    """
    # Ensure dt_utc has consistent timezone (UTC) in both dataframes
    if 'dt_utc' in pred1.columns:
        pred1 = pred1.copy()
        pred1['dt_utc'] = pd.to_datetime(pred1['dt_utc'], utc=True)
    if 'dt_utc' in pred2.columns:
        pred2 = pred2.copy()
        pred2['dt_utc'] = pd.to_datetime(pred2['dt_utc'], utc=True)

    # Merge on (Country, SiteNumber, dt_utc, horizon)
    merged = pred1.merge(
        pred2,
        on=['Country', 'SiteNumber', 'dt_utc', 'horizon'],
        suffixes=('_1', '_2'),
        how='inner'
    )

    if len(merged) == 0:
        raise ValueError("No common samples found between predictions")

    y_true = merged['y_true_1'].values  # Should be same as y_true_2
    y_pred1 = merged['y_pred_1'].values
    y_pred2 = merged['y_pred_2'].values

    return y_true, y_pred1, y_pred2


# =========================
# Main Comparison Logic
# =========================

def run_protocol_a_comparisons():
    """Run Diebold-Mariano tests for Protocol A (Classical vs ML)."""
    print("\n" + "=" * 80)
    print("PROTOCOL A: Classical ML Models (Full Coverage)")
    print("=" * 80)

    results = []

    # Load dataset
    print(f"\nLoading dataset: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df['dt_utc'] = pd.to_datetime(df['dt_utc'], utc=True)
    df = normalize_season_column(df)

    # Build multi-horizon targets
    print("Building multi-horizon targets...")
    df = build_multi_horizon_targets_like_lstm(df, PROTOCOL_A_HORIZONS)

    # Split to test set
    test_df = df[(df['dt_utc'] >= TEST_START) & (df['dt_utc'] <= TEST_END)].copy()
    print(f"Test set size: {len(test_df):,} samples")

    # Get feature columns (same logic as training scripts)
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

        # Include raw NO2/PM10
        for c in ["NO2", "PM10"]:
            if c in df.columns:
                candidates.append(c)

        # dedup
        candidates = list(dict.fromkeys(candidates))
        return candidates

    feature_cols = select_feature_columns(test_df)
    print(f"Number of features: {len(feature_cols)}")

    # Run comparisons for each horizon
    for horizon in PROTOCOL_A_HORIZONS:
        print(f"\n{'='*60}")
        print(f"Horizon: {horizon}h")
        print(f"{'='*60}")

        # Identify best models
        try:
            model1, model2, rmse1, rmse2 = identify_best_models(
                PROTOCOL_A_MODELS,
                horizon,
                protocol='A'
            )
            print(f"Best model 1: {model1} (RMSE: {rmse1:.3f})")
            print(f"Best model 2: {model2} (RMSE: {rmse2:.3f})")
        except Exception as e:
            print(f"ERROR identifying best models: {e}")
            continue

        # Sample test set before generating predictions
        test_df_horizon = test_df.copy()
        if len(test_df_horizon) > MAX_PREDICTIONS_PER_HORIZON:
            print(f"\nSampling {MAX_PREDICTIONS_PER_HORIZON:,} samples from {len(test_df_horizon):,} test samples...")
            np.random.seed(42 + horizon)  # For reproducibility (different per horizon)
            sample_idx = np.random.choice(len(test_df_horizon), size=MAX_PREDICTIONS_PER_HORIZON, replace=False)
            test_df_horizon = test_df_horizon.iloc[sample_idx].copy()
            print(f"  Sampled test set size: {len(test_df_horizon):,}")

        # Generate predictions
        try:
            print(f"\nGenerating predictions for {model1}...")
            model1_path = f"../models/{model1}/{model1}_h{horizon}.pkl"
            if not os.path.exists(model1_path):
                print(f"  Model file not found: {model1_path}")
                continue

            pred1 = generate_predictions_classical(
                model1_path, test_df_horizon, feature_cols, horizon, model1
            )
            print(f"  Generated {len(pred1):,} predictions")

            print(f"\nGenerating predictions for {model2}...")
            model2_path = f"../models/{model2}/{model2}_h{horizon}.pkl"
            if not os.path.exists(model2_path):
                print(f"  Model file not found: {model2_path}")
                continue

            pred2 = generate_predictions_classical(
                model2_path, test_df_horizon, feature_cols, horizon, model2
            )
            print(f"  Generated {len(pred2):,} predictions")

        except Exception as e:
            print(f"ERROR generating predictions: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Merge predictions
        try:
            y_true, y_pred1, y_pred2 = merge_predictions(pred1, pred2)
            print(f"\nMerged predictions: {len(y_true):,} common samples")
        except Exception as e:
            print(f"ERROR merging predictions: {e}")
            continue

        # Compute errors
        errors1 = y_true - y_pred1
        errors2 = y_true - y_pred2

        # Compute DM test
        dm_stat, p_value = compute_dm_statistic(errors1, errors2, loss='squared')

        # Determine winner
        if p_value < 0.05:
            if dm_stat < 0:
                winner = model1
            else:
                winner = model2
        else:
            winner = 'tie'

        print(f"\nDiebold-Mariano Test Results:")
        print(f"  DM statistic: {dm_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
        print(f"  Winner: {winner}")

        # Store results
        results.append({
            'Protocol': 'A',
            'horizon': horizon,
            'model1': model1,
            'model2': model2,
            'DM_stat': dm_stat,
            'p_value': p_value,
            'model1_rmse': rmse1,
            'model2_rmse': rmse2,
            'n_samples': len(y_true),
            'winner': winner
        })

    return results


def run_protocol_b_comparisons():
    """Run Diebold-Mariano tests for Protocol B (ML vs DL)."""
    print("\n" + "=" * 80)
    print("PROTOCOL B: ML vs Deep Learning (LSTM-Eligible Subset)")
    print("=" * 80)

    results = []

    # Load dataset
    print(f"\nLoading dataset: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df['dt_utc'] = pd.to_datetime(df['dt_utc'], utc=True)
    df = normalize_season_column(df)
    df = make_station_key(df)
    df = recover_station_area_label(df)

    # Get feature columns
    feature_cols = select_feature_columns_like_lstm_script(df)
    print(f"Number of features: {len(feature_cols)}")

    # Build multi-horizon targets
    print("Building multi-horizon targets...")
    df = build_multi_horizon_targets_like_lstm(df, PROTOCOL_B_HORIZONS)

    # Apply LSTM-aligned anchor filtering
    print("Applying LSTM-aligned anchor filtering...")
    train_frames, test_frames, common_st, valid_st = build_lstm_aligned_anchors(
        df=df,
        feature_cols=feature_cols,
        horizons=PROTOCOL_B_HORIZONS,
        train_end=TRAIN_END,
        test_start=TEST_START,
        test_end=TEST_END,
        lookback_h=168,
        gap_break_hours=2
    )

    # Run comparisons for each horizon
    for horizon in PROTOCOL_B_HORIZONS:
        print(f"\n{'='*60}")
        print(f"Horizon: {horizon}h")
        print(f"{'='*60}")

        # Filter test set to valid anchors
        test_anchors_h = test_frames[horizon]
        test_df = filter_df_by_anchors(df, test_anchors_h)
        test_df = test_df[(test_df['dt_utc'] >= TEST_START) & (test_df['dt_utc'] <= TEST_END)].copy()
        print(f"Test set size (filtered): {len(test_df):,} samples")

        if len(test_df) == 0:
            print("No test samples available after filtering")
            continue

        # Don't sample here for Protocol B - LSTM needs continuous sequences
        # Early stopping will limit to MAX_PREDICTIONS_PER_HORIZON during generation

        # Identify best models (separate DL and ML)
        try:
            # Best DL model
            dl_models = [m for m in PROTOCOL_B_MODELS if m.startswith('lstm')]
            dl_model, _, dl_rmse, _ = identify_best_models(
                dl_models + [dl_models[0]],  # Hack to get at least 2 models
                horizon,
                protocol='B'
            )

            # Best ML model
            ml_models = [m for m in PROTOCOL_B_MODELS if not m.startswith('lstm')]
            ml_model, _, ml_rmse, _ = identify_best_models(
                ml_models + [ml_models[0]],  # Hack to get at least 2 models
                horizon,
                protocol='B'
            )

            print(f"Best DL model: {dl_model} (RMSE: {dl_rmse:.3f})")
            print(f"Best ML model: {ml_model} (RMSE: {ml_rmse:.3f})")

            model1 = dl_model
            model2 = ml_model
            rmse1 = dl_rmse
            rmse2 = ml_rmse

        except Exception as e:
            print(f"ERROR identifying best models: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Generate predictions
        try:
            # DL model predictions
            print(f"\nGenerating predictions for {model1} (LSTM)...")
            model1_path = f"../models/{model1}/{model1}_h{horizon}.keras"

            # Handle scaler file naming
            if model1 == 'lstm_residual':
                scaler_path = f"../train_models/scalers/lstm_residual_single_station_scalers.json"
            else:
                scaler_path = f"../train_models/scalers/{model1}_scalers.json"

            if not os.path.exists(model1_path):
                print(f"  Model file not found: {model1_path}")
                continue
            if not os.path.exists(scaler_path):
                print(f"  Scaler file not found: {scaler_path}")
                continue

            pred1 = generate_predictions_lstm(
                model1_path, scaler_path, test_df, feature_cols, horizon, model1,
                max_predictions=MAX_PREDICTIONS_PER_HORIZON
            )
            print(f"  Generated {len(pred1):,} predictions")

            # ML model predictions - filter to same samples as LSTM for efficiency
            print(f"\nGenerating predictions for {model2} (ML)...")
            print(f"  Filtering to same {len(pred1):,} samples as LSTM for efficiency...")

            # Create a filter based on LSTM predictions (Country, SiteNumber, dt_utc)
            lstm_samples = pred1[['Country', 'SiteNumber', 'dt_utc']].copy()
            test_df_filtered = test_df.merge(
                lstm_samples,
                on=['Country', 'SiteNumber', 'dt_utc'],
                how='inner'
            )
            print(f"  Filtered test set: {len(test_df_filtered):,} samples")

            # For classical models, use the same feature selection as training (not LSTM features)
            def select_feature_columns_classical(df: pd.DataFrame) -> list[str]:
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
                # Include raw NO2/PM10
                for c in ["NO2", "PM10"]:
                    if c in df.columns:
                        candidates.append(c)
                # dedup
                candidates = list(dict.fromkeys(candidates))
                return candidates

            feature_cols_classical = select_feature_columns_classical(test_df_filtered)
            print(f"  Number of features for classical model: {len(feature_cols_classical)}")

            # For DL models, files are named without _dl suffix
            # e.g., models/lgb_dl/lgb_h1.pkl (not lgb_dl_h1.pkl)
            model2_base = model2.replace('_dl', '')
            model2_path = f"../models/{model2}/{model2_base}_h{horizon}.pkl"

            if not os.path.exists(model2_path):
                print(f"  Model file not found: {model2_path}")
                continue

            pred2 = generate_predictions_classical(
                model2_path, test_df_filtered, feature_cols_classical, horizon, model2
            )
            print(f"  Generated {len(pred2):,} predictions")

        except Exception as e:
            print(f"ERROR generating predictions: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Merge predictions
        try:
            y_true, y_pred1, y_pred2 = merge_predictions(pred1, pred2)
            print(f"\nMerged predictions: {len(y_true):,} common samples")
        except Exception as e:
            print(f"ERROR merging predictions: {e}")
            continue

        # Compute errors
        errors1 = y_true - y_pred1
        errors2 = y_true - y_pred2

        # Compute DM test
        dm_stat, p_value = compute_dm_statistic(errors1, errors2, loss='squared')

        # Determine winner
        if p_value < 0.05:
            if dm_stat < 0:
                winner = model1
            else:
                winner = model2
        else:
            winner = 'tie'

        print(f"\nDiebold-Mariano Test Results:")
        print(f"  DM statistic: {dm_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
        print(f"  Winner: {winner}")

        # Store results
        results.append({
            'Protocol': 'B',
            'horizon': horizon,
            'model1': model1,
            'model2': model2,
            'DM_stat': dm_stat,
            'p_value': p_value,
            'model1_rmse': rmse1,
            'model2_rmse': rmse2,
            'n_samples': len(y_true),
            'winner': winner
        })

    return results


def standardize_dm_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize model comparisons:
    - Protocol A: alphabetical order
    - Protocol B: ML models (with _dl suffix) first, then DL models (pure LSTM variants)
    When flipping, negate DM_stat and swap RMSE values.
    """
    df = df.copy()

    for idx, row in df.iterrows():
        model1 = row['model1']
        model2 = row['model2']
        protocol = row['Protocol']

        should_swap = False

        if protocol == 'B':
            # For Protocol B: ML models (_dl suffix) should come first
            # Check if model1 is pure DL (lstm without _dl) and model2 is ML (_dl)
            is_model1_pure_dl = 'lstm' in model1.lower() and '_dl' not in model1
            is_model2_ml = '_dl' in model2 and 'lstm' not in model2

            is_model2_pure_dl = 'lstm' in model2.lower() and '_dl' not in model2
            is_model1_ml = '_dl' in model1 and 'lstm' not in model1

            if is_model1_pure_dl and is_model2_ml:
                # model1 is pure DL, model2 is ML -> swap to put ML first
                should_swap = True
            elif is_model2_pure_dl and is_model1_ml:
                # model1 is ML, model2 is pure DL -> correct order, no swap
                should_swap = False
            else:
                # Both are same type, use alphabetical
                should_swap = model1 > model2
        else:
            # Protocol A: alphabetical order
            should_swap = model1 > model2

        if should_swap:
            # Swap models
            df.at[idx, 'model1'] = model2
            df.at[idx, 'model2'] = model1

            # Negate DM_stat (because direction is reversed)
            df.at[idx, 'DM_stat'] = -row['DM_stat']

            # Swap RMSE values
            df.at[idx, 'model1_rmse'] = row['model2_rmse']
            df.at[idx, 'model2_rmse'] = row['model1_rmse']

            # Update winner field
            if row['winner'] == model1:
                df.at[idx, 'winner'] = model2
            elif row['winner'] == model2:
                df.at[idx, 'winner'] = model1
            # 'tie' stays as 'tie'

    return df


def plot_dm_results(results_df: pd.DataFrame):
    """
    Creates a publication-quality Bar Chart for DM test results
    with a clear legend explaining positive vs negative direction.
    """
    # Standardize comparisons first
    results_df = standardize_dm_comparisons(results_df)

    # 1. Setup Scaling
    y_max = max(results_df['DM_stat'].max(), 3.0) + 0.5
    y_min = min(results_df['DM_stat'].min(), -3.0) - 0.5
    y_limit = max(abs(y_max), abs(y_min))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    plt.subplots_adjust(wspace=0.15)

    protocols = [('A', 'Protocol A: Classical ML Models'),
                 ('B', 'Protocol B: ML vs Deep Learning')]

    for i, (prot_code, title) in enumerate(protocols):
        ax = axes[i]
        data = results_df[results_df['Protocol'] == prot_code].sort_values('horizon')

        if len(data) == 0:
            continue

        x = np.arange(len(data))
        y = data['DM_stat'].values
        horizons = data['horizon'].values
        p_values = data['p_value'].values

        # 2. Add "Significance Zone" (The Grey Band)
        ax.axhspan(-1.96, 1.96, color='#f0f0f0', alpha=1.0, zorder=0)
        ax.axhline(1.96, color='gray', linestyle=':', linewidth=0.8, zorder=1)
        ax.axhline(-1.96, color='gray', linestyle=':', linewidth=0.8, zorder=1)
        ax.axhline(0, color='black', linewidth=1, zorder=1)

        # 3. Plot Bars
        is_sig = p_values < 0.05
        colors = ['#D62728' if sig else '#555555' for sig in is_sig]

        bars = ax.bar(x, y, color=colors, alpha=0.8, width=0.5, edgecolor='black', linewidth=1, zorder=2)

        # 4. Clean Labels & Layout
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Forecast Horizon', fontsize=12, labelpad=10)

        # Set y-axis label for both plots
        ax.set_ylabel('DM Statistic (Z-Score)', fontsize=12)

        if i == 0:
            # --- CUSTOM LEGEND ---
            # Creates a small box in the top-left to explain direction
            legend_text = "↑ Positive: Model 1 Wins\n↓ Negative: Model 2 Wins"
            ax.text(0.03, 0.97, legend_text, transform=ax.transAxes,
                    fontsize=10, va='top', ha='left',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9, edgecolor='gray'))

        # Annotate "Significance Threshold" on both plots, centered
        center_x = len(data) / 2 - 0.5
        ax.text(center_x, 2.05, 'Sig. (±1.96)', color='gray', fontsize=8, ha='center', va='bottom')

        # X-Ticks
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h}h" for h in horizons], fontsize=11, fontweight='bold')

        # 5. Smart Model Annotations
        for bar, row in zip(bars, data.itertuples()):
            m1 = row.model1.replace('_dl', '').upper()
            m2 = row.model2.replace('_dl', '').upper()

            # For Protocol B, split underscores in model names for readability
            if prot_code == 'B':
                m1 = m1.replace('_', '\n')
                m2 = m2.replace('_', '\n')

            label_text = f"{m1}\nvs\n{m2}"

            height = bar.get_height()
            is_positive = height >= 0

            # Label placement logic
            if abs(height) > 1.2:
                # Place inside
                y_pos = height - (0.4 if is_positive else -0.4)
                text_color = 'white'
                va = 'top' if is_positive else 'bottom'
            else:
                # Place outside
                y_pos = height + (0.2 if is_positive else -0.2)
                text_color = 'black'
                va = 'bottom' if is_positive else 'top'

            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                    label_text,
                    ha='center', va=va,
                    fontsize=8, fontweight='bold', color=text_color)

        # Limits & Grid
        ax.set_ylim(-y_limit, y_limit)
        ax.grid(axis='y', linestyle='-', alpha=0.1, zorder=0)

    plt.tight_layout(pad=3.0)

    output_path = f"{FIGURES_DIR}/diebold_mariano_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


# =========================
# Main Execution
# =========================

def main():
    """Main execution function."""
    all_results = []

    # Run Protocol A comparisons
    try:
        results_a = run_protocol_a_comparisons()
        all_results.extend(results_a)
    except Exception as e:
        print(f"\nERROR in Protocol A: {e}")
        import traceback
        traceback.print_exc()

    # Run Protocol B comparisons
    try:
        results_b = run_protocol_b_comparisons()
        all_results.extend(results_b)
    except Exception as e:
        print(f"\nERROR in Protocol B: {e}")
        import traceback
        traceback.print_exc()

    # Save results
    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)
        output_csv = f"{OUTPUT_DIR}/diebold_mariano_results.csv"
        results_df.to_csv(output_csv, index=False)
        print(f"\n{'='*80}")
        print(f"Results saved to: {output_csv}")
        print(f"{'='*80}")

        # Display summary
        print("\nSummary:")
        print(results_df.to_string(index=False))

        # Create plots
        try:
            plot_dm_results(results_df)
        except Exception as e:
            print(f"\nERROR creating plots: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nNo results generated!")

    print("\n" + "=" * 80)
    print("Diebold-Mariano analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
