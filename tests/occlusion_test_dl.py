"""
Occlusion Test for Deep Learning Models

Tests impact of removing lookback window on LSTM model performance.
Compares baseline (full lookback) vs occluded (zero lookback) predictions.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
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
    recover_station_area_label,
    compute_segments
)

# =========================
# CONFIG
# =========================
CSV_PATH = "../ml_ready_dataset_full_realistic.csv"
OUTPUT_DIR = "../results"
FIGURES_DIR = "../figures"

HORIZONS = [1, 3, 6, 12, 24]
LOOKBACK_H = 168

TRAIN_END = pd.Timestamp("2022-12-31 23:00:00", tz="UTC")
TEST_START = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
TEST_END = pd.Timestamp("2024-12-31 23:00:00", tz="UTC")

SAMPLE_SIZE = 1000  # Number of sequences to test

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 80)
print("Occlusion Test for Deep Learning Models")
print("=" * 80)


# =========================
# Helper Functions
# =========================

def identify_best_dl_model(
    models: List[str],
    horizon: int = 1
) -> Tuple[str, float]:
    """
    Identify best deep learning model based on RMSE at given horizon.

    Returns:
        (model_name, rmse)
    """
    results = []

    for model_name in models:
        # Handle special naming for lstm_residual
        if model_name == 'lstm_residual':
            metrics_file = f"{OUTPUT_DIR}/lstm_residual_singleoutput_aligned_metrics.csv"
        else:
            metrics_file = f"{OUTPUT_DIR}/{model_name}_metrics.csv"

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

    if len(results) == 0:
        raise ValueError(f"No valid DL models found for horizon {horizon}")

    # Sort by RMSE (ascending) and return best
    results.sort(key=lambda x: x[1])
    best_model, best_rmse = results[0]

    return best_model, best_rmse


def load_lstm_model_and_scaler(
    model_name: str,
    horizon: int
) -> Tuple:
    """
    Load LSTM model and corresponding scalers.

    Returns:
        (model, scalers_dict, feature_cols)
    """
    import tensorflow as tf

    # Enable unsafe deserialization for Lambda layers (models trained locally, trusted)
    tf.keras.config.enable_unsafe_deserialization()

    # Load model
    model_path = f"../models/{model_name}/{model_name}_h{horizon}.keras"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"  Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    # Load scalers - handle naming variations
    if model_name == 'lstm_residual':
        scaler_path = f"../train_models/scalers/lstm_residual_single_station_scalers.json"
    else:
        scaler_path = f"../train_models/scalers/{model_name}_scalers.json"

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    print(f"  Loading scalers: {scaler_path}")
    with open(scaler_path, 'r') as f:
        scaler_data = json.load(f)

    scalers = scaler_data['scalers']
    feature_cols = scaler_data['feature_cols']

    return model, scalers, feature_cols


def create_test_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    horizon: int,
    lookback_h: int,
    scalers: Dict,
    sample_size: int,
    model
) -> Tuple:
    """
    Create test sequences from data.

    Returns:
        If model expects station IDs:
            (X_sequences, station_ids, y_targets, station_keys)
        Else:
            (X_sequences, None, y_targets, station_keys)

        - X_sequences: shape (n_sequences, lookback_h, n_features)
        - station_ids: shape (n_sequences,) if needed, else None
        - y_targets: shape (n_sequences,)
        - station_keys: list of station identifiers
    """
    df = df.sort_values(['Country', 'SiteNumber', 'dt_utc']).reset_index(drop=True)
    df = make_station_key(df)

    # Create station ID mapping (for models with station embeddings)
    unique_stations = sorted(df['station_key'].unique())
    station_to_id = {station: idx for idx, station in enumerate(unique_stations)}

    # Check if model expects 2 inputs (sequence + station_id)
    model_expects_station_id = len(model.inputs) == 2

    all_sequences = []
    all_targets = []
    all_station_keys = []
    all_station_ids = []

    for station_key, g in df.groupby('station_key', sort=False):
        # Get scaler for this station
        if station_key not in scalers:
            continue

        scaler_info = scalers[station_key]
        means = np.array(scaler_info['mean'])
        stds = np.array(scaler_info['std'])

        # Get station ID for embedding models
        station_id = station_to_id.get(station_key, 0)

        # Segment at gaps
        segs = compute_segments(g, gap_break_hours=2)

        for seg in segs:
            if len(seg) < lookback_h + horizon + 1:
                continue

            seg = seg.sort_values('dt_utc').reset_index(drop=True)

            # Extract features in correct order
            X_raw = seg[feature_cols].values.astype(np.float32)

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

                all_sequences.append(x_win)
                all_targets.append(y_vals[t])
                all_station_keys.append(station_key)
                all_station_ids.append(station_id)

                # Early stopping if we have enough samples
                if len(all_sequences) >= sample_size * 2:
                    break

            if len(all_sequences) >= sample_size * 2:
                break

        if len(all_sequences) >= sample_size * 2:
            break

    if len(all_sequences) == 0:
        raise ValueError("No valid sequences found")

    # Convert to arrays
    X_sequences = np.array(all_sequences)
    y_targets = np.array(all_targets)
    station_ids_array = np.array(all_station_ids) if model_expects_station_id else None

    # Sample if we have too many
    if len(X_sequences) > sample_size:
        indices = np.random.choice(len(X_sequences), sample_size, replace=False)
        X_sequences = X_sequences[indices]
        y_targets = y_targets[indices]
        station_keys_sampled = [all_station_keys[i] for i in indices]
        if station_ids_array is not None:
            station_ids_array = station_ids_array[indices]
    else:
        station_keys_sampled = all_station_keys

    print(f"  Created {len(X_sequences)} sequences")
    print(f"  Sequence shape: {X_sequences.shape}")
    if model_expects_station_id:
        print(f"  Station IDs shape: {station_ids_array.shape}")

    return X_sequences, station_ids_array, y_targets, station_keys_sampled


def occlude_lookback(
    X_sequences: np.ndarray,
    method: str = 'zero'
) -> np.ndarray:
    """
    Occlude lookback window by setting to zeros (after normalization).

    Args:
        X_sequences: shape (n_sequences, lookback_h, n_features)
        method: 'zero' to set to zeros

    Returns:
        Occluded sequences with same shape
    """
    X_occluded = X_sequences.copy()

    if method == 'zero':
        # Set all lookback values to zero (mean in normalized space)
        X_occluded[:, :, :] = 0.0
    else:
        raise ValueError(f"Unknown occlusion method: {method}")

    return X_occluded


def compute_performance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute performance metrics.

    Returns:
        Dictionary with MAE, RMSE, R2
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }


# =========================
# Main Analysis
# =========================

def run_occlusion_test():
    """Run occlusion test for best deep learning model."""
    print("\n" + "=" * 80)
    print("Running Occlusion Test")
    print("=" * 80)

    # Identify best DL model
    print("\nIdentifying best deep learning model...")
    candidate_models = ['lstm_attention', 'lstm_residual']
    best_model, best_rmse = identify_best_dl_model(candidate_models, horizon=1)
    print(f"Selected model: {best_model} (RMSE at h1: {best_rmse:.3f})")
    print(f"Will test on all horizons: {HORIZONS}")

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
    df = build_multi_horizon_targets_like_lstm(df, HORIZONS)

    # Apply LSTM-aligned anchor filtering
    print("Applying LSTM-aligned anchor filtering...")
    train_frames, test_frames, common_st, valid_st = build_lstm_aligned_anchors(
        df=df,
        feature_cols=feature_cols,
        horizons=HORIZONS,
        train_end=TRAIN_END,
        test_start=TEST_START,
        test_end=TEST_END,
        lookback_h=LOOKBACK_H,
        gap_break_hours=2
    )

    results = []

    # Test each horizon
    for horizon in HORIZONS:
        print(f"\n{'='*60}")
        print(f"Testing Horizon: {horizon}h")
        print(f"{'='*60}")

        # Load model and scalers
        try:
            model, scalers, feature_cols_scaler = load_lstm_model_and_scaler(
                best_model, horizon
            )
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            continue

        # Filter test set to valid anchors
        test_anchors_h = test_frames[horizon]
        test_df = filter_df_by_anchors(df, test_anchors_h)
        test_df = test_df[(test_df['dt_utc'] >= TEST_START) & (test_df['dt_utc'] <= TEST_END)].copy()
        print(f"Test set size (filtered): {len(test_df):,} samples")

        if len(test_df) == 0:
            print("No test samples available after filtering")
            continue

        # Create test sequences
        try:
            X_sequences, station_ids, y_targets, station_keys = create_test_sequences(
                test_df,
                feature_cols_scaler,
                horizon,
                LOOKBACK_H,
                scalers,
                SAMPLE_SIZE,
                model
            )
        except Exception as e:
            print(f"  ERROR creating sequences: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Check if model expects station IDs
        model_expects_station_id = station_ids is not None

        # Get PM2.5 normalization parameters for denormalization
        # The model outputs normalized predictions, we need to denormalize them
        pm25_index = feature_cols_scaler.index('PM2.5')

        # Get unique station keys for this batch and their normalization params
        unique_station_keys = list(set(station_keys))
        denorm_means = {}
        denorm_stds = {}
        for sk in unique_station_keys:
            if sk in scalers:
                denorm_means[sk] = scalers[sk]['mean'][pm25_index]
                denorm_stds[sk] = scalers[sk]['std'][pm25_index]

        # Baseline predictions (full lookback)
        print("\n  Computing baseline predictions (full lookback)...")
        if model_expects_station_id:
            y_pred_norm = model.predict([X_sequences, station_ids], verbose=0).flatten()
        else:
            y_pred_norm = model.predict(X_sequences, verbose=0).flatten()

        # Denormalize predictions
        y_pred_baseline = np.array([
            pred * denorm_stds.get(sk, 1.0) + denorm_means.get(sk, 0.0)
            for pred, sk in zip(y_pred_norm, station_keys)
        ])

        baseline_metrics = compute_performance_metrics(y_targets, y_pred_baseline)

        print(f"  Baseline MAE: {baseline_metrics['MAE']:.3f}")
        print(f"  Baseline RMSE: {baseline_metrics['RMSE']:.3f}")
        print(f"  Baseline R²: {baseline_metrics['R2']:.3f}")

        # Occluded predictions (zero lookback)
        print("\n  Computing occluded predictions (zero lookback)...")
        X_occluded = occlude_lookback(X_sequences, method='zero')
        if model_expects_station_id:
            y_pred_occluded_norm = model.predict([X_occluded, station_ids], verbose=0).flatten()
        else:
            y_pred_occluded_norm = model.predict(X_occluded, verbose=0).flatten()

        # Denormalize predictions
        y_pred_occluded = np.array([
            pred * denorm_stds.get(sk, 1.0) + denorm_means.get(sk, 0.0)
            for pred, sk in zip(y_pred_occluded_norm, station_keys)
        ])

        occluded_metrics = compute_performance_metrics(y_targets, y_pred_occluded)

        print(f"  Occluded MAE: {occluded_metrics['MAE']:.3f}")
        print(f"  Occluded RMSE: {occluded_metrics['RMSE']:.3f}")
        print(f"  Occluded R²: {occluded_metrics['R2']:.3f}")

        # Compute degradation
        delta_mae = occluded_metrics['MAE'] - baseline_metrics['MAE']
        delta_rmse = occluded_metrics['RMSE'] - baseline_metrics['RMSE']
        relative_degradation = (delta_rmse / baseline_metrics['RMSE']) * 100

        print(f"\n  Delta MAE: +{delta_mae:.3f} ({delta_mae/baseline_metrics['MAE']*100:.1f}%)")
        print(f"  Delta RMSE: +{delta_rmse:.3f} ({relative_degradation:.1f}%)")

        # Store results
        results.append({
            'model': best_model,
            'horizon': horizon,
            'baseline_mae': baseline_metrics['MAE'],
            'baseline_rmse': baseline_metrics['RMSE'],
            'baseline_r2': baseline_metrics['R2'],
            'occluded_mae': occluded_metrics['MAE'],
            'occluded_rmse': occluded_metrics['RMSE'],
            'occluded_r2': occluded_metrics['R2'],
            'delta_mae': delta_mae,
            'delta_rmse': delta_rmse,
            'relative_degradation_pct': relative_degradation,
            'n_sequences': len(X_sequences)
        })

    return results


def plot_degradation(results_df: pd.DataFrame):
    """Create visualization of occlusion test results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sort by horizon
    results_df = results_df.sort_values('horizon')

    model_name = results_df['model'].iloc[0]
    horizons = results_df['horizon'].values

    # MAE degradation
    ax = axes[0]
    delta_mae = results_df['delta_mae'].values
    relative_mae = (delta_mae / results_df['baseline_mae'].values) * 100

    ax.bar(horizons, relative_mae, alpha=0.7, edgecolor='black', color='steelblue')
    ax.set_xlabel('Forecast Horizon (hours)', fontsize=12)
    ax.set_ylabel('MAE Degradation (%)', fontsize=12)
    ax.set_title(f'MAE Impact of Zero Lookback\n({model_name})', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(horizons)

    # Add value labels
    for h, v in zip(horizons, relative_mae):
        ax.text(h, v + 1, f'{v:.1f}%', ha='center', fontsize=9)

    # RMSE degradation
    ax = axes[1]
    relative_rmse = results_df['relative_degradation_pct'].values

    ax.bar(horizons, relative_rmse, alpha=0.7, edgecolor='black', color='coral')
    ax.set_xlabel('Forecast Horizon (hours)', fontsize=12)
    ax.set_ylabel('RMSE Degradation (%)', fontsize=12)
    ax.set_title(f'RMSE Impact of Zero Lookback\n({model_name})', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(horizons)

    # Add value labels
    for h, v in zip(horizons, relative_rmse):
        ax.text(h, v + 1, f'{v:.1f}%', ha='center', fontsize=9)

    plt.tight_layout()

    output_path = f"{FIGURES_DIR}/occlusion_test_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


# =========================
# Main Execution
# =========================

def main():
    """Main execution function."""

    # Run occlusion test
    try:
        results = run_occlusion_test()

        if len(results) > 0:
            # Save results
            results_df = pd.DataFrame(results)
            output_csv = f"{OUTPUT_DIR}/occlusion_test_results.csv"
            results_df.to_csv(output_csv, index=False)

            print(f"\n{'='*80}")
            print(f"Results saved to: {output_csv}")
            print(f"{'='*80}")

            # Display summary
            print("\nSummary:")
            print(results_df.to_string(index=False))

            # Create plots
            print("\n" + "=" * 80)
            print("Creating visualizations...")
            print("=" * 80)
            plot_degradation(results_df)

        else:
            print("\nNo results generated!")

    except Exception as e:
        print(f"\nERROR in occlusion test: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Occlusion test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
