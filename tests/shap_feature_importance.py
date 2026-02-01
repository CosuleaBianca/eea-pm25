"""
SHAP Feature Importance Analysis

Computes SHAP values for tree-based models with grouped feature importance.
Identifies contributions of different feature groups (PM2.5 History, Meteorological, etc.).
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
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

HORIZONS_TO_ANALYZE = [1, 12]  # Analyze h1 and h12
SAMPLE_SIZE = 10000  # Number of samples for SHAP computation

TRAIN_END = pd.Timestamp("2022-12-31 23:00:00", tz="UTC")
TEST_START = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
TEST_END = pd.Timestamp("2024-12-31 23:00:00", tz="UTC")

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 80)
print("SHAP Feature Importance Analysis")
print("=" * 80)


# =========================
# Feature Groups Definition
# =========================

def define_feature_groups() -> Dict[str, List[str]]:
    """
    Define feature groups based on feature_config.py.

    Returns:
        Dictionary mapping group names to feature patterns/lists
    """
    groups = {
        'Meteorological': [
            'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
            'wind_u', 'wind_v', 'precipitation', 'surface_pressure'
        ],
        'PM2.5 History': [
            'PM2.5_lag_1h', 'PM2.5_lag_2h', 'PM2.5_lag_3h',
            'PM2.5_lag_6h', 'PM2.5_lag_12h', 'PM2.5_lag_24h', 'PM2.5_lag_168h',
            'PM2.5_rolling_mean_3h', 'PM2.5_rolling_mean_6h',
            'PM2.5_rolling_mean_12h', 'PM2.5_rolling_mean_24h',
            'PM2.5_rolling_std_3h', 'PM2.5_rolling_std_6h',
            'PM2.5_rolling_std_12h', 'PM2.5_rolling_std_24h', 'PM2.5'
        ],
        'Other Pollutants': [
            'NO2', 'NO2_lag_1h', 'NO2_lag_2h', 'NO2_lag_3h', 'NO2_lag_6h',
            'NO2_lag_12h', 'NO2_lag_24h', 'NO2_lag_168h',
            'NO2_rolling_mean_3h', 'NO2_rolling_mean_6h',
            'NO2_rolling_mean_12h', 'NO2_rolling_mean_24h',
            'NO2_rolling_std_3h', 'NO2_rolling_std_6h',
            'NO2_rolling_std_12h', 'NO2_rolling_std_24h',
            'PM10', 'PM10_lag_1h', 'PM10_lag_2h', 'PM10_lag_3h', 'PM10_lag_6h',
            'PM10_lag_12h', 'PM10_lag_24h', 'PM10_lag_168h',
            'PM10_rolling_mean_3h', 'PM10_rolling_mean_6h',
            'PM10_rolling_mean_12h', 'PM10_rolling_mean_24h',
            'PM10_rolling_std_3h', 'PM10_rolling_std_6h',
            'PM10_rolling_std_12h', 'PM10_rolling_std_24h'
        ],
        'Temporal': [
            'hour', 'hour_sin', 'hour_cos',
            'day_of_week', 'day_of_month', 'month', 'month_sin', 'month_cos',
            'year', 'is_weekend', 'season'
        ],
        'Metadata': [
            'Latitude', 'Longitude', 'Altitude',
            'StationType_background', 'StationType_industrial', 'StationType_traffic',
            'StationArea_rural', 'StationArea_rural-nearcity',
            'StationArea_suburban', 'StationArea_urban',
            'station_encoded'
        ]
    }

    return groups


def assign_features_to_groups(
    feature_cols: List[str],
    groups: Dict[str, List[str]]
) -> Dict[str, str]:
    """
    Assign each feature to a group.

    Returns:
        Dictionary mapping feature name to group name
    """
    feature_to_group = {}

    for feature in feature_cols:
        assigned = False
        for group_name, group_features in groups.items():
            if feature in group_features:
                feature_to_group[feature] = group_name
                assigned = True
                break

        if not assigned:
            # Check for pattern matching (e.g., StationType_*)
            if feature.startswith('StationType_') or feature.startswith('StationArea_'):
                feature_to_group[feature] = 'Metadata'
            elif 'PM2.5' in feature:
                feature_to_group[feature] = 'PM2.5 History'
            elif 'NO2' in feature or 'PM10' in feature:
                feature_to_group[feature] = 'Other Pollutants'
            else:
                feature_to_group[feature] = 'Other'

    return feature_to_group


# =========================
# Model Selection
# =========================

def identify_best_model(
    models: List[str],
    horizon: int,
    protocol: str
) -> Tuple[str, float]:
    """
    Identify best tree-based model for given horizon.

    Returns:
        (model_name, rmse)
    """
    results = []

    for model_name in models:
        # Construct metrics file path
        if protocol == 'A':
            if model_name == 'gam':
                metrics_file = f"{OUTPUT_DIR}/gam_metrics_tuned.csv"
            else:
                metrics_file = f"{OUTPUT_DIR}/{model_name}_metrics_tuned.csv"
        else:  # Protocol B
            metrics_file = f"{OUTPUT_DIR}/{model_name}_dl_metrics_tuned.csv"

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
        raise ValueError(f"No valid models found for horizon {horizon}")

    # Sort by RMSE (ascending) and return best
    results.sort(key=lambda x: x[1])
    best_model, best_rmse = results[0]

    return best_model, best_rmse


# =========================
# SHAP Computation
# =========================

def compute_shap_values(
    model,
    X_sample: np.ndarray,
    model_type: str
) -> np.ndarray:
    """
    Compute SHAP values for given model and data sample.

    Returns:
        SHAP values array of shape (n_samples, n_features)
    """
    print(f"  Computing SHAP values (model type: {model_type})...")

    try:
        if model_type in ['xgb', 'lgb']:
            # Use TreeExplainer for tree-based models
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            except (ValueError, TypeError) as e:
                # Fallback for XGBoost compatibility issues
                if 'base_score' in str(e) or 'could not convert' in str(e):
                    print(f"  WARNING: TreeExplainer failed ({str(e)[:100]})")
                    print(f"  Falling back to KernelExplainer (slower but more compatible)...")
                    background_size = min(100, len(X_sample))
                    background = shap.sample(X_sample, background_size)
                    explainer = shap.KernelExplainer(model.predict, background)
                    shap_values = explainer.shap_values(X_sample[:1000])  # Limit for speed
                else:
                    raise

            # Handle multi-output case
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

        elif model_type == 'gam':
            # Use KernelExplainer for GAM
            # Sample background data for efficiency
            background_size = min(100, len(X_sample))
            background = shap.sample(X_sample, background_size)

            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_sample[:1000])  # Limit for speed

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        print(f"  SHAP values computed. Shape: {shap_values.shape}")
        return shap_values

    except Exception as e:
        print(f"  ERROR computing SHAP values: {e}")
        import traceback
        traceback.print_exc()
        raise


def compute_grouped_shap(
    shap_values: np.ndarray,
    feature_cols: List[str],
    feature_to_group: Dict[str, str]
) -> pd.DataFrame:
    """
    Aggregate SHAP values by feature groups.

    Returns:
        DataFrame with columns: group, n_features, importance_mean, importance_std
    """
    # Compute absolute SHAP values (importance)
    abs_shap = np.abs(shap_values)

    # Aggregate by group
    group_importance = {}

    for group_name in set(feature_to_group.values()):
        # Get features in this group
        group_features = [f for f in feature_cols if feature_to_group.get(f) == group_name]
        group_indices = [i for i, f in enumerate(feature_cols) if f in group_features]

        if len(group_indices) == 0:
            continue

        # Sum absolute SHAP values across features in group
        group_shap = abs_shap[:, group_indices].sum(axis=1)

        group_importance[group_name] = {
            'n_features': len(group_features),
            'importance_mean': group_shap.mean(),
            'importance_std': group_shap.std()
        }

    # Convert to DataFrame
    results = []
    for group_name, stats in group_importance.items():
        results.append({
            'group': group_name,
            'n_features': stats['n_features'],
            'importance_mean': stats['importance_mean'],
            'importance_std': stats['importance_std']
        })

    df_grouped = pd.DataFrame(results)
    df_grouped = df_grouped.sort_values('importance_mean', ascending=False)

    return df_grouped


# =========================
# Visualization
# =========================

def plot_grouped_importance(results: Dict[int, Tuple[pd.DataFrame, pd.DataFrame, str]]):
    """
    Create grouped feature importance plots for multiple horizons.

    Args:
        results: Dictionary mapping horizon to (individual_df, grouped_df, model_name)
    """
    n_horizons = len(results)
    fig, axes = plt.subplots(1, n_horizons, figsize=(7 * n_horizons, 6))

    if n_horizons == 1:
        axes = [axes]

    for i, (horizon, (_, df_grouped, model_name)) in enumerate(sorted(results.items())):
        ax = axes[i]

        # Sort by importance
        df_plot = df_grouped.sort_values('importance_mean', ascending=True)

        # Create horizontal bar plot
        y_pos = np.arange(len(df_plot))
        ax.barh(y_pos, df_plot['importance_mean'], xerr=df_plot['importance_std'],
                alpha=0.7, edgecolor='black', capsize=5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_plot['group'])
        ax.set_xlabel('Mean |SHAP| (Feature Importance)', fontsize=12)
        ax.set_title(f'Grouped Feature Importance\nHorizon: {horizon}h ({model_name})',
                    fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add feature counts as text
        for j, (idx, row) in enumerate(df_plot.iterrows()):
            ax.text(row['importance_mean'] + row['importance_std'] + 0.1,
                   j, f"n={row['n_features']}", va='center', fontsize=9)

    plt.tight_layout()

    output_path = f"{FIGURES_DIR}/shap_grouped_importance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


# =========================
# Main Analysis
# =========================

def analyze_protocol_a():
    """Analyze Protocol A models (full coverage)."""
    print("\n" + "=" * 80)
    print("PROTOCOL A: Classical ML Models")
    print("=" * 80)

    results = {}

    # Load dataset
    print(f"\nLoading dataset: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df['dt_utc'] = pd.to_datetime(df['dt_utc'], utc=True)
    df = normalize_season_column(df)

    # Build multi-horizon targets
    print("Building multi-horizon targets...")
    df = build_multi_horizon_targets_like_lstm(df, HORIZONS_TO_ANALYZE)

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

    # Define feature groups
    groups = define_feature_groups()
    feature_to_group = assign_features_to_groups(feature_cols, groups)

    # Identify best model (use h1 to select model, then apply to all horizons)
    print("\nIdentifying best tree-based model...")
    # Use LightGBM preferentially due to better SHAP compatibility
    candidate_models = ['lgb', 'xgb']  # Try LightGBM first
    best_model, best_rmse = identify_best_model(candidate_models, horizon=1, protocol='A')

    # Force LightGBM if XGBoost was selected (due to SHAP compatibility issues)
    if best_model == 'xgb':
        print(f"  Note: XGBoost selected (RMSE: {best_rmse:.3f}) but using LightGBM for SHAP compatibility")
        best_model = 'lgb'
        # Get LightGBM RMSE
        lgb_metrics = pd.read_csv(f"{OUTPUT_DIR}/lgb_metrics_tuned.csv")
        mask = (lgb_metrics['Scope'] == 'Global') & (lgb_metrics['Group'] == 'All') & (lgb_metrics['horizon'] == 1)
        best_rmse = lgb_metrics.loc[mask, 'RMSE'].values[0]

    print(f"Selected model: {best_model} (RMSE at h1: {best_rmse:.3f})")
    print(f"Will use {best_model} for all horizons: {HORIZONS_TO_ANALYZE}")

    # Analyze each horizon
    for horizon in HORIZONS_TO_ANALYZE:
        print(f"\n{'='*60}")
        print(f"Analyzing Horizon: {horizon}h")
        print(f"{'='*60}")

        # Filter test data for this horizon
        test_h = test_df.dropna(subset=[f'y_h{horizon}']).copy()

        # Sample for SHAP computation
        if len(test_h) > SAMPLE_SIZE:
            test_sample = test_h.sample(n=SAMPLE_SIZE, random_state=42)
        else:
            test_sample = test_h

        print(f"Sample size: {len(test_sample):,}")

        # Load model
        model_path = f"../models/{best_model}/{best_model}_h{horizon}.pkl"
        if not os.path.exists(model_path):
            print(f"  Model file not found: {model_path}")
            continue

        print(f"Loading model: {model_path}")
        model = joblib.load(model_path)

        # Prepare data
        X_sample = test_sample[feature_cols].values

        # Compute SHAP values
        try:
            shap_values = compute_shap_values(model, X_sample, best_model)
        except Exception as e:
            print(f"  Failed to compute SHAP values: {e}")
            continue

        # Individual feature importance
        abs_shap = np.abs(shap_values)
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'shap_mean': abs_shap.mean(axis=0),
            'shap_std': abs_shap.std(axis=0)
        })
        feature_importance = feature_importance.sort_values('shap_mean', ascending=False)

        # Save individual importance
        output_individual = f"{OUTPUT_DIR}/shap_importance_h{horizon}.csv"
        feature_importance.to_csv(output_individual, index=False)
        print(f"\nIndividual feature importance saved to: {output_individual}")

        print("\nTop 10 features:")
        print(feature_importance.head(10).to_string(index=False))

        # Grouped importance
        df_grouped = compute_grouped_shap(shap_values, feature_cols, feature_to_group)

        # Save grouped importance
        output_grouped = f"{OUTPUT_DIR}/shap_grouped_importance_h{horizon}.csv"
        df_grouped.to_csv(output_grouped, index=False)
        print(f"\nGrouped feature importance saved to: {output_grouped}")

        print("\nGrouped importance:")
        print(df_grouped.to_string(index=False))

        # Store results for plotting
        results[horizon] = (feature_importance, df_grouped, best_model)

    return results


def main():
    """Main execution function."""

    # Run Protocol A analysis
    try:
        results = analyze_protocol_a()

        if len(results) > 0:
            # Create plots
            print("\n" + "=" * 80)
            print("Creating visualizations...")
            print("=" * 80)
            plot_grouped_importance(results)
        else:
            print("\nNo results to plot!")

    except Exception as e:
        print(f"\nERROR in analysis: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("SHAP feature importance analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
