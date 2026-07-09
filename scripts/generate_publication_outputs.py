#!/usr/bin/env python3
"""
Generate publication-ready tables and figures from PM2.5 model evaluation metrics.

This script creates:
- Table 1: Dataset overview (countries, splits, station counts)
- Table 2: Protocol A performance (classical/ML models on full coverage)
- Table 3: Protocol B performance (all models on sequence-eligible subset)
- Figure 1: Spatial coverage map with insets
- Figure 2: Performance vs horizon comparison
- Figure 3: Country robustness heatmap
- Figure 4: Station area robustness heatmap
- Figure 5: Bias analysis (optional)
- Figure 6: Model skill comparison vs persistence baseline
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_CONFIGS = {
    'protocol_a': {
        'persistence': {'file': 'persistence_metrics.csv', 'label': 'Persistence', 'color': '#1f77b4'},
        'lr': {'file': 'lr_metrics.csv', 'label': 'Linear Regression', 'color': '#ff7f0e'},
        'gam': {'file': 'gam_metrics_tuned.csv', 'label': 'GAM', 'color': '#2ca02c'},
        'rf': {'file': 'rf_metrics_tuned.csv', 'label': 'Random Forest', 'color': '#d62728'},
        'xgb': {'file': 'xgb_metrics_tuned.csv', 'label': 'XGBoost', 'color': '#9467bd'},
        'lgb': {'file': 'lgb_metrics_tuned.csv', 'label': 'LightGBM', 'color': '#8c564b'},
    },
    'protocol_b': {
        'persistence': {'file': 'persistence_dl_metrics.csv', 'label': 'Persistence', 'color': '#1f77b4'},
        'lr': {'file': 'lr_dl_metrics.csv', 'label': 'Linear Regression', 'color': '#ff7f0e'},
        'gam': {'file': 'gam_dl_metrics_tuned.csv', 'label': 'GAM', 'color': '#2ca02c'},
        'rf': {'file': 'rf_dl_metrics_tuned.csv', 'label': 'Random Forest', 'color': '#d62728'},
        'xgb': {'file': 'xgb_dl_metrics_tuned.csv', 'label': 'XGBoost', 'color': '#9467bd'},
        'lgb': {'file': 'lgb_dl_metrics_tuned.csv', 'label': 'LightGBM', 'color': '#8c564b'},
        'lstm_global': {'file': 'lstm_global_metrics.csv', 'label': 'LSTM-Global', 'color': '#e377c2'},
        'lstm_residual': {'file': 'lstm_residual_singleoutput_aligned_metrics.csv', 'label': 'LSTM-Residual', 'color': '#7f7f7f'},
        'lstm_attention': {'file': 'lstm_attention_metrics.csv', 'label': 'LSTM-Attention', 'color': '#bcbd22'},
        'lstm_cnn': {'file': 'lstm_cnn_metrics.csv', 'label': 'LSTM-CNN', 'color': '#17becf'},
        'itransformer': {'file': 'itransformer_metrics.csv', 'label': 'iTransformer', 'color': '#393b79'},
    }
}

# Model family grouping for family-distinct figure styling.
MODEL_FAMILY = {
    'persistence': 'baseline', 'lr': 'baseline',
    'gam': 'statistical',
    'rf': 'tree', 'xgb': 'tree', 'lgb': 'tree',
    'lstm_global': 'dl', 'lstm_residual': 'dl', 'lstm_attention': 'dl',
    'lstm_cnn': 'dl', 'itransformer': 'dl',
}
# One line style per family so families are distinguishable even in grayscale.
FAMILY_LINESTYLE = {
    'baseline': ':', 'statistical': '-.', 'tree': '--', 'dl': '-',
}
FAMILY_MARKER = {
    'baseline': 'x', 'statistical': 'D', 'tree': 's', 'dl': 'o',
}

STYLE_CONFIG = {
    'figure_dpi': 300,
    'font_size': 10,
    'figure_format': 'png',
    'color_palette': 'tab10',
    'heatmap_cmap': 'RdYlGn_r',
    'europe_bbox': {'lat': [37, 63], 'lon': [-10, 30]},
}

# Metrics columns expected in all metrics CSV files
METRICS_COLUMNS = ['Scope', 'Group', 'horizon', 'MAE', 'RMSE', 'R2', 'Bias', 'Count']

# Horizons to evaluate
HORIZONS = [1, 3, 6, 12, 24]

# Countries and station areas
COUNTRIES = ['AT', 'BE', 'ES', 'FI', 'FR']
STATION_AREAS = ['urban', 'suburban', 'rural', 'rural-nearcity']

# Country display names
COUNTRY_NAMES = {
    'AT': 'Austria',
    'BE': 'Belgium',
    'ES': 'Spain',
    'FI': 'Finland',
    'FR': 'France',
}

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_metrics(protocol: str = 'A', models: Optional[List[str]] = None,
                 results_dir: str = '../results', verbose: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Load metrics CSV files for specified protocol.

    Parameters
    ----------
    protocol : str
        'A' for full-coverage evaluation, 'B' for sequence-eligible subset
    models : list of str, optional
        List of model names to load. If None, loads all models for the protocol.
    results_dir : str
        Directory containing results CSV files
    verbose : bool
        Print loading progress

    Returns
    -------
    dict
        Dictionary mapping model names to DataFrames
    """
    protocol_key = f'protocol_{protocol.lower()}'
    if protocol_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown protocol: {protocol}. Must be 'A' or 'B'.")

    config = MODEL_CONFIGS[protocol_key]
    if models is None:
        models = list(config.keys())

    metrics_dict = {}
    results_path = Path(results_dir)

    for model in models:
        if model not in config:
            warnings.warn(f"Model '{model}' not found in {protocol_key} configuration. Skipping.")
            continue

        file_name = config[model]['file']
        file_path = results_path / file_name

        if not file_path.exists():
            warnings.warn(f"Metrics file not found: {file_path}. Skipping {model}.")
            continue

        if verbose:
            print(f"Loading {model}: {file_name}")

        df = pd.read_csv(file_path)

        # Handle baseline column if present (persistence has it)
        if 'baseline' in df.columns:
            df = df.drop(columns=['baseline'])

        # Validate structure
        missing_cols = set(METRICS_COLUMNS) - set(df.columns)
        if missing_cols:
            warnings.warn(f"Missing columns in {file_name}: {missing_cols}")

        metrics_dict[model] = df

    if verbose:
        print(f"\nLoaded {len(metrics_dict)} models for Protocol {protocol}")

    return metrics_dict


def load_metrics_with_skill(protocol: str = 'A', models: Optional[List[str]] = None,
                             results_with_skill_dir: str = '../results_with_skill',
                             verbose: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Load metrics CSV files with skill scores from results_with_skill directory.

    Parameters
    ----------
    protocol : str
        'A' for full-coverage evaluation, 'B' for sequence-eligible subset
    models : list of str, optional
        List of model names to load. If None, loads all models for the protocol.
    results_with_skill_dir : str
        Directory containing results CSV files with skill columns
    verbose : bool
        Print loading progress

    Returns
    -------
    dict
        Dictionary mapping model names to DataFrames with skill columns
    """
    protocol_key = f'protocol_{protocol.lower()}'
    if protocol_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown protocol: {protocol}. Must be 'A' or 'B'.")

    config = MODEL_CONFIGS[protocol_key]
    if models is None:
        models = list(config.keys())

    metrics_dict = {}
    results_path = Path(results_with_skill_dir)

    for model in models:
        if model not in config:
            warnings.warn(f"Model '{model}' not found in {protocol_key} configuration. Skipping.")
            continue

        # Skip persistence as it has no skill scores (it IS the baseline)
        if model == 'persistence':
            continue

        file_name = config[model]['file']
        file_path = results_path / file_name

        if not file_path.exists():
            warnings.warn(f"Skill metrics file not found: {file_path}. Skipping {model}.")
            continue

        if verbose:
            print(f"Loading {model} (with skill): {file_name}")

        df = pd.read_csv(file_path)

        # Validate skill columns exist
        skill_cols = ['MAE_skill', 'RMSE_skill']
        missing_cols = [col for col in skill_cols if col not in df.columns]
        if missing_cols:
            warnings.warn(f"Missing skill columns in {file_name}: {missing_cols}. Skipping.")
            continue

        metrics_dict[model] = df

    if verbose:
        print(f"\nLoaded {len(metrics_dict)} models with skill scores for Protocol {protocol}")

    return metrics_dict


def load_station_metadata(dataset_path: str = '../ml_ready_dataset_full_realistic.csv',
                          verbose: bool = False) -> pd.DataFrame:
    """
    Load station metadata from ML dataset.

    Parameters
    ----------
    dataset_path : str
        Path to ML dataset CSV file
    verbose : bool
        Print loading progress

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Country, SiteNumber, Latitude, Longitude, StationArea_Label
    """
    if verbose:
        print(f"Loading station metadata from {dataset_path}")

    df = pd.read_csv(dataset_path, low_memory=False)

    # Get unique stations
    station_cols = ['Country', 'SiteNumber', 'Latitude', 'Longitude']
    area_cols = [col for col in df.columns if col.startswith('StationArea_')]

    stations = df[station_cols + area_cols].drop_duplicates(subset=['Country', 'SiteNumber'])

    # Convert one-hot encoded station areas to single label column
    def get_station_area_label(row):
        """Convert one-hot encoded station area to label."""
        for area in STATION_AREAS:
            col_name = f'StationArea_{area}'
            if col_name in row.index and row[col_name] == 1:
                return area
        return 'unknown'

    stations['StationArea_Label'] = stations.apply(get_station_area_label, axis=1)

    # Drop one-hot columns
    stations = stations.drop(columns=area_cols)

    if verbose:
        print(f"Found {len(stations)} unique stations")
        print(f"Countries: {sorted(stations['Country'].unique())}")
        print(f"Station areas: {stations['StationArea_Label'].value_counts().to_dict()}")

    return stations


def load_coverage_info(coverage_path: str = '../dataset_build/station_train_test_coverage.csv',
                       verbose: bool = False) -> pd.DataFrame:
    """
    Load station coverage statistics.

    Parameters
    ----------
    coverage_path : str
        Path to coverage CSV file
    verbose : bool
        Print loading progress

    Returns
    -------
    pd.DataFrame
        Coverage statistics per station
    """
    if verbose:
        print(f"Loading coverage info from {coverage_path}")

    if not Path(coverage_path).exists():
        # Coverage CSV is a Phase-3 artifact (gitignored) used only by Table 1 (dataset
        # overview). Degrade gracefully so the rest of the outputs can still be generated.
        print(f"  NOTE: coverage file not found ({coverage_path}); Table 1 will be skipped.")
        return None

    coverage = pd.read_csv(coverage_path)

    if verbose:
        print(f"Loaded coverage info for {len(coverage)} stations")

    return coverage


def validate_count_consistency(metrics_dict: Dict[str, pd.DataFrame],
                               protocol: str, verbose: bool = False) -> Dict[str, List[str]]:
    """
    Check that all models have consistent Count values for same scope/group/horizon.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary of model DataFrames
    protocol : str
        Protocol identifier for logging
    verbose : bool
        Print detailed warnings

    Returns
    -------
    dict
        Dictionary of inconsistencies found (empty if all consistent)
    """
    if verbose:
        print(f"\nValidating count consistency for Protocol {protocol}...")

    inconsistencies = {}

    # Get all unique (Scope, Group, horizon) combinations
    all_keys = set()
    for model, df in metrics_dict.items():
        keys = df[['Scope', 'Group', 'horizon']].apply(tuple, axis=1)
        all_keys.update(keys)

    # Check each combination
    for scope, group, horizon in sorted(all_keys):
        counts = {}
        for model, df in metrics_dict.items():
            mask = (df['Scope'] == scope) & (df['Group'] == group) & (df['horizon'] == horizon)
            rows = df[mask]
            if len(rows) > 0:
                counts[model] = rows['Count'].iloc[0]

        # Check if all counts are the same
        if len(set(counts.values())) > 1:
            key = f"{scope}/{group}/h={horizon}"
            inconsistencies[key] = [f"{model}={count}" for model, count in counts.items()]
            if verbose:
                print(f"  WARNING: Inconsistent counts for {key}")
                print(f"    {', '.join(inconsistencies[key])}")

    if not inconsistencies:
        if verbose:
            print("  [OK] All counts are consistent")
    else:
        print(f"\n  Found {len(inconsistencies)} inconsistencies in Protocol {protocol}")

    return inconsistencies


# ============================================================================
# TABLE GENERATION FUNCTIONS
# ============================================================================

def format_latex_table(df: pd.DataFrame, caption: str, label: str,
                       bold_best: Optional[Dict[str, str]] = None) -> str:
    """
    Convert DataFrame to LaTeX table with booktabs style (without jinja2 dependency).

    Parameters
    ----------
    df : pd.DataFrame
        Table data
    caption : str
        Table caption
    label : str
        LaTeX label for referencing
    bold_best : dict, optional
        Dictionary mapping column names to 'min' or 'max' for bolding best values

    Returns
    -------
    str
        LaTeX table code
    """
    # Build LaTeX manually to avoid jinja2 dependency
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{' + caption + '}')
    lines.append(r'\label{' + label + '}')

    # Column format
    ncols = len(df.columns) + 1  # +1 for index
    col_format = 'l' + 'r' * len(df.columns)
    lines.append(r'\begin{tabular}{' + col_format + '}')
    lines.append(r'\toprule')

    # Header row
    header = [str(df.index.name) if df.index.name else ''] + [str(c) for c in df.columns]
    lines.append(' & '.join(header) + r' \\')
    lines.append(r'\midrule')

    # Data rows
    for idx, row in df.iterrows():
        row_data = [str(idx)]
        for val in row:
            if pd.isna(val):
                row_data.append('--')
            elif isinstance(val, (int, float)):
                row_data.append(f'{val:.2f}')
            else:
                row_data.append(str(val))
        lines.append(' & '.join(row_data) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


def generate_table1_dataset_overview(
    station_metadata: pd.DataFrame,
    coverage_info: pd.DataFrame,
    metrics_a: Dict[str, pd.DataFrame],
    metrics_b: Dict[str, pd.DataFrame],
    output_dir: str = 'tables',
    verbose: bool = False
) -> None:
    """
    Generate Table 1: Dataset overview with coverage, splits, and station counts.

    Parameters
    ----------
    station_metadata : pd.DataFrame
        Station information
    coverage_info : pd.DataFrame
        Coverage statistics
    metrics_a : dict
        Protocol A metrics
    metrics_b : dict
        Protocol B metrics
    output_dir : str
        Output directory for tables
    verbose : bool
        Print progress
    """
    if verbose:
        print("\nGenerating Table 1: Dataset Overview...")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Section 1: Coverage information
    coverage_data = {
        'Attribute': ['Countries', 'Time Period', 'Frequency', 'Total Stations', 'Train Period', 'Test Period'],
        'Protocol_A': [
            ', '.join(COUNTRIES),
            '2018-01-01 to 2024-12-31',
            'Hourly',
            str(len(station_metadata)),
            '2018-01-08 to 2022-12-31',
            '2023-01-01 to 2024-12-31'
        ],
        'Protocol_B': [
            ', '.join(COUNTRIES),
            '2018-01-01 to 2024-12-31',
            'Hourly',
            str(len(station_metadata)),
            '2018-01-08 to 2022-12-31',
            '2023-01-08 to 2024-12-31'  # Different test period (starts 7 days later due to 168h lookback)
        ]
    }

    # Section 2: Station counts by country
    country_counts_a = station_metadata['Country'].value_counts().reindex(COUNTRIES, fill_value=0)
    country_counts_b = country_counts_a  # Same stations, different samples

    # Section 3: Station counts by area
    area_counts = station_metadata['StationArea_Label'].value_counts().reindex(STATION_AREAS, fill_value=0)

    # Section 4: Test sample counts (from Global/All metrics)
    sample_counts_a = {}
    sample_counts_b = {}

    # Get counts from persistence model (arbitrary choice, all should be same)
    if 'persistence' in metrics_a:
        df_a = metrics_a['persistence']
        mask_a = (df_a['Scope'] == 'Global') & (df_a['Group'] == 'All')
        for h in HORIZONS:
            row = df_a[mask_a & (df_a['horizon'] == h)]
            if len(row) > 0:
                sample_counts_a[h] = int(row['Count'].iloc[0])

    if 'persistence' in metrics_b:
        df_b = metrics_b['persistence']
        mask_b = (df_b['Scope'] == 'Global') & (df_b['Group'] == 'All')
        for h in HORIZONS:
            row = df_b[mask_b & (df_b['horizon'] == h)]
            if len(row) > 0:
                sample_counts_b[h] = int(row['Count'].iloc[0])

    # Create combined table
    table_data = []

    # Coverage section
    table_data.append(['Dataset Coverage', '', ''])
    for i, attr in enumerate(coverage_data['Attribute']):
        table_data.append([attr, coverage_data['Protocol_A'][i], coverage_data['Protocol_B'][i]])

    table_data.append(['', '', ''])

    # Station counts by country
    table_data.append(['Stations by Country', 'Protocol A', 'Protocol B'])
    for country in COUNTRIES:
        table_data.append([
            COUNTRY_NAMES[country],
            str(country_counts_a[country]),
            str(country_counts_b[country])
        ])
    table_data.append(['Total', str(country_counts_a.sum()), str(country_counts_b.sum())])

    table_data.append(['', '', ''])

    # Station counts by area
    table_data.append(['Stations by Area', 'Protocol A', 'Protocol B'])
    for area in STATION_AREAS:
        table_data.append([area.capitalize(), str(area_counts[area]), str(area_counts[area])])

    table_data.append(['', '', ''])

    # Test sample counts by horizon
    table_data.append(['Test Samples by Horizon', 'Protocol A', 'Protocol B'])
    for h in HORIZONS:
        count_a = sample_counts_a.get(h, 'N/A')
        count_b = sample_counts_b.get(h, 'N/A')
        if isinstance(count_a, int):
            count_a = f'{count_a:,}'
        if isinstance(count_b, int):
            count_b = f'{count_b:,}'
        table_data.append([f'{h}h ahead', count_a, count_b])

    # Create DataFrame
    df = pd.DataFrame(table_data, columns=['Metric', 'Protocol A', 'Protocol B'])

    # Save CSV
    csv_path = output_path / 'table1_dataset_overview.csv'
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"  Saved CSV: {csv_path}")

    # Save LaTeX
    caption = (
        "Dataset overview showing geographic coverage, temporal splits, and station counts. "
        "Protocol A uses full coverage (all stations with valid features), while Protocol B "
        "uses a sequence-eligible subset (stations with continuous 168h windows, gap≤2h)."
    )
    latex_code = format_latex_table(df, caption, 'tab:dataset_overview')

    latex_path = output_path / 'table1_dataset_overview.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_code)
    if verbose:
        print(f"  Saved LaTeX: {latex_path}")


def _generate_table2_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """
    Generate custom LaTeX table for Table 2 with multirow model blocks and bolded best RMSE.

    Parameters
    ----------
    df : pd.DataFrame
        Table data with columns: Model, Metric, h=1, h=3, h=6, h=12, h=24
    caption : str
        Table caption
    label : str
        LaTeX label for referencing

    Returns
    -------
    str
        LaTeX table code
    """
    # Find best (minimum) RMSE for each horizon
    rmse_rows = df[df['Metric'] == 'RMSE']
    best_rmse = {}
    for h in [1, 3, 6, 12, 24]:
        col = f'h={h}'
        if col in rmse_rows.columns:
            valid_values = rmse_rows[col].dropna()
            if len(valid_values) > 0:
                best_rmse[h] = valid_values.min()

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{' + caption + '}')
    lines.append(r'\label{' + label + '}')
    lines.append(r'\begin{tabular}{llrrrrr}')  # Left-align Model & Metric, right-align horizons
    lines.append(r'\toprule')

    # Header row
    lines.append(r'Model & Metric & h=1 & h=3 & h=6 & h=12 & h=24 \\')
    lines.append(r'\midrule')

    # Data rows
    current_model = None
    metric_count_in_block = 0

    for idx, row in df.iterrows():
        model = row['Model']
        metric = row['Metric']

        # Check if this is the Count row (last row)
        if model == 'Count (N)':
            lines.append(r'\midrule')
            # Count row uses multicolumn for Model + Metric cells
            count_values = []
            for h in [1, 3, 6, 12, 24]:
                val = row[f'h={h}']
                if pd.isna(val):
                    count_values.append('--')
                else:
                    # Format with thousands separator
                    count_values.append(f'{int(val):,}')
            lines.append(r'\multicolumn{2}{l}{Count (N)} & ' + ' & '.join(count_values) + r' \\')
            continue

        # For model blocks: use multirow for Model column
        if model != '':  # New model block starts
            current_model = model
            metric_count_in_block = 0

        metric_count_in_block += 1

        # Build row
        if metric_count_in_block == 1:
            # First row of model block: use multirow
            row_parts = [r'\multirow{4}{*}{' + current_model + '}']
        else:
            # Subsequent rows: empty cell
            row_parts = ['']

        row_parts.append(metric)

        # Add metric values for each horizon
        for h in [1, 3, 6, 12, 24]:
            val = row[f'h={h}']
            if pd.isna(val):
                row_parts.append('--')
            else:
                # Format based on metric type
                if metric == 'R²':
                    formatted = f'{val:.3f}'
                else:  # MAE, RMSE, Bias
                    formatted = f'{val:.2f}'

                # Bold if this is RMSE and it's the best value for this horizon
                if metric == 'RMSE' and h in best_rmse:
                    if abs(val - best_rmse[h]) < 1e-6:  # Handle floating point comparison
                        formatted = r'\textbf{' + formatted + '}'

                row_parts.append(formatted)

        lines.append(' & '.join(row_parts) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


def generate_table2_protocol_a_performance(
    metrics_a: Dict[str, pd.DataFrame],
    output_dir: str = 'tables',
    verbose: bool = False
) -> None:
    """
    Generate Table 2: Protocol A global performance across horizons.

    Restructured format: Metrics stacked as rows (MAE, RMSE, R², Bias) within each model block.

    Parameters
    ----------
    metrics_a : dict
        Protocol A metrics
    output_dir : str
        Output directory
    verbose : bool
        Print progress
    """
    if verbose:
        print("\nGenerating Table 2: Protocol A Performance...")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Models to include in order
    models_order = ['persistence', 'lr', 'gam', 'rf', 'xgb', 'lgb']

    # Metric names in display order
    metric_names = ['MAE', 'RMSE', 'R²', 'Bias']

    # Build table with stacked metrics
    rows = []
    count_values = {}  # To store count values (should be identical across models)

    for model_key in models_order:
        if model_key not in metrics_a:
            warnings.warn(f"Model {model_key} not found in Protocol A metrics. Skipping.")
            continue

        df = metrics_a[model_key]
        config = MODEL_CONFIGS['protocol_a'][model_key]

        # Filter to Global/All
        mask = (df['Scope'] == 'Global') & (df['Group'] == 'All')
        global_df = df[mask].sort_values('horizon')

        # Collect data for all horizons
        h_data = {}
        for h in HORIZONS:
            h_subset = global_df[global_df['horizon'] == h]
            if len(h_subset) > 0:
                h_data[h] = {
                    'MAE': h_subset['MAE'].iloc[0],
                    'RMSE': h_subset['RMSE'].iloc[0],
                    'R2': h_subset['R2'].iloc[0],
                    'Bias': h_subset['Bias'].iloc[0],
                    'Count': int(h_subset['Count'].iloc[0])
                }
                # Store count values (should be identical across models)
                if h not in count_values:
                    count_values[h] = h_data[h]['Count']
            else:
                h_data[h] = {
                    'MAE': np.nan,
                    'RMSE': np.nan,
                    'R2': np.nan,
                    'Bias': np.nan,
                    'Count': 0
                }

        # Create 4 rows for this model (one per metric)
        for i, metric_name in enumerate(metric_names):
            row_data = {
                'Model': config['label'] if i == 0 else '',  # Only first row gets model name
                'Metric': metric_name
            }
            # Map metric display name to data key
            metric_key = 'R2' if metric_name == 'R²' else metric_name
            for h in HORIZONS:
                row_data[f'h={h}'] = h_data[h][metric_key]
            rows.append(row_data)

    # Add Count row at the end
    count_row = {'Model': 'Count (N)', 'Metric': ''}
    for h in HORIZONS:
        count_row[f'h={h}'] = count_values.get(h, 0)
    rows.append(count_row)

    # Create DataFrame
    df_table = pd.DataFrame(rows)
    column_order = ['Model', 'Metric'] + [f'h={h}' for h in HORIZONS]
    df_table = df_table[column_order]

    # Format DataFrame for CSV output (apply formatting based on Metric type)
    df_formatted = df_table.copy()
    for h in HORIZONS:
        col = f'h={h}'
        # pandas>=3.0 no longer silently upcasts a float column on string assignment;
        # cast to object first so the formatted strings can be written in place.
        df_formatted[col] = df_formatted[col].astype(object)
        # Apply formatting row by row based on Metric
        for idx in df_formatted.index:
            metric = df_formatted.loc[idx, 'Metric']
            val = df_formatted.loc[idx, col]

            if pd.isna(val):
                df_formatted.loc[idx, col] = ''
            elif metric == '':  # Count row
                df_formatted.loc[idx, col] = f'{int(val):,}'
            elif metric == 'R²':
                df_formatted.loc[idx, col] = f'{val:.3f}'
            else:  # MAE, RMSE, Bias
                df_formatted.loc[idx, col] = f'{val:.2f}'

    # Save CSV with formatted values
    csv_path = output_path / 'table2_protocol_a_performance.csv'
    df_formatted.to_csv(csv_path, index=False)
    if verbose:
        print(f"  Saved CSV: {csv_path}")

    # Generate custom LaTeX with multirow and custom formatting
    caption = (
        "Protocol A: Global performance metrics across prediction horizons. "
        "Lower MAE/RMSE and higher R² indicate better performance. "
        "Bias shows systematic over/under-prediction (positive = over-prediction). "
        "Note: Under Protocol A, sample counts are identical across models for each horizon; "
        "therefore Count is reported once at the bottom."
    )
    latex_code = _generate_table2_latex(df_table, caption, 'tab:protocol_a_performance')

    latex_path = output_path / 'table2_protocol_a_performance.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_code)
    if verbose:
        print(f"  Saved LaTeX: {latex_path}")


def _generate_table3_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """
    Generate custom LaTeX table for Table 3 with multirow model blocks and bolded best RMSE.

    Parameters
    ----------
    df : pd.DataFrame
        Table data with columns: Model, Metric, h=1, h=3, h=6, h=12, h=24
    caption : str
        Table caption
    label : str
        LaTeX label for referencing

    Returns
    -------
    str
        LaTeX table code
    """
    # Find best (minimum) RMSE for each horizon
    rmse_rows = df[df['Metric'] == 'RMSE']
    best_rmse = {}
    for h in [1, 3, 6, 12, 24]:
        col = f'h={h}'
        if col in rmse_rows.columns:
            valid_values = rmse_rows[col].dropna()
            if len(valid_values) > 0:
                best_rmse[h] = valid_values.min()

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{' + caption + '}')
    lines.append(r'\label{' + label + '}')
    lines.append(r'\begin{tabular}{llrrrrr}')  # Left-align Model & Metric, right-align horizons
    lines.append(r'\toprule')

    # Header row
    lines.append(r'Model & Metric & h=1 & h=3 & h=6 & h=12 & h=24 \\')
    lines.append(r'\midrule')

    # Data rows
    current_model = None
    metric_count_in_block = 0

    for idx, row in df.iterrows():
        model = row['Model']
        metric = row['Metric']

        # Check if this is the Count row (last row)
        if model == 'Count (N)':
            lines.append(r'\midrule')
            # Count row uses multicolumn for Model + Metric cells
            count_values = []
            for h in [1, 3, 6, 12, 24]:
                val = row[f'h={h}']
                if pd.isna(val):
                    count_values.append('--')
                else:
                    # Format with thousands separator
                    count_values.append(f'{int(val):,}')
            lines.append(r'\multicolumn{2}{l}{Count (N)} & ' + ' & '.join(count_values) + r' \\')
            continue

        # For model blocks: use multirow for Model column
        if model != '':  # New model block starts
            current_model = model
            metric_count_in_block = 0

        metric_count_in_block += 1

        # Build row
        if metric_count_in_block == 1:
            # First row of model block: use multirow
            row_parts = [r'\multirow{4}{*}{' + current_model + '}']
        else:
            # Subsequent rows: empty cell
            row_parts = ['']

        row_parts.append(metric)

        # Add metric values for each horizon
        for h in [1, 3, 6, 12, 24]:
            val = row[f'h={h}']
            if pd.isna(val):
                row_parts.append('--')
            else:
                # Format based on metric type
                if metric == 'R²':
                    formatted = f'{val:.3f}'
                else:  # MAE, RMSE, Bias
                    formatted = f'{val:.2f}'

                # Bold if this is RMSE and it's the best value for this horizon
                if metric == 'RMSE' and h in best_rmse:
                    if abs(val - best_rmse[h]) < 1e-6:  # Handle floating point comparison
                        formatted = r'\textbf{' + formatted + '}'

                row_parts.append(formatted)

        lines.append(' & '.join(row_parts) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)


def generate_table3_protocol_b_performance(
    metrics_b: Dict[str, pd.DataFrame],
    output_dir: str = 'tables',
    verbose: bool = False
) -> None:
    """
    Generate Table 3: Protocol B global performance including deep learning models.

    Restructured format: Metrics stacked as rows (MAE, RMSE, R², Bias) within each model block.

    Parameters
    ----------
    metrics_b : dict
        Protocol B metrics
    output_dir : str
        Output directory
    verbose : bool
        Print progress
    """
    if verbose:
        print("\nGenerating Table 3: Protocol B Performance...")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Models to include in order
    models_order = ['persistence', 'lr', 'gam', 'rf', 'xgb', 'lgb',
                    'lstm_global', 'lstm_residual', 'lstm_attention', 'lstm_cnn',
                    'itransformer']

    # Metric names in display order
    metric_names = ['MAE', 'RMSE', 'R²', 'Bias']

    # Build table with stacked metrics
    rows = []
    count_values = {}  # To store count values (should be identical across models)

    for model_key in models_order:
        if model_key not in metrics_b:
            warnings.warn(f"Model {model_key} not found in Protocol B metrics. Skipping.")
            continue

        df = metrics_b[model_key]
        config = MODEL_CONFIGS['protocol_b'][model_key]

        # Filter to Global/All
        mask = (df['Scope'] == 'Global') & (df['Group'] == 'All')
        global_df = df[mask].sort_values('horizon')

        # Collect data for all horizons
        h_data = {}
        for h in HORIZONS:
            h_subset = global_df[global_df['horizon'] == h]
            if len(h_subset) > 0:
                h_data[h] = {
                    'MAE': h_subset['MAE'].iloc[0],
                    'RMSE': h_subset['RMSE'].iloc[0],
                    'R2': h_subset['R2'].iloc[0],
                    'Bias': h_subset['Bias'].iloc[0],
                    'Count': int(h_subset['Count'].iloc[0])
                }
                # Store count values (should be identical across models)
                if h not in count_values:
                    count_values[h] = h_data[h]['Count']
            else:
                h_data[h] = {
                    'MAE': np.nan,
                    'RMSE': np.nan,
                    'R2': np.nan,
                    'Bias': np.nan,
                    'Count': 0
                }

        # Create 4 rows for this model (one per metric)
        for i, metric_name in enumerate(metric_names):
            row_data = {
                'Model': config['label'] if i == 0 else '',  # Only first row gets model name
                'Metric': metric_name
            }
            # Map metric display name to data key
            metric_key = 'R2' if metric_name == 'R²' else metric_name
            for h in HORIZONS:
                row_data[f'h={h}'] = h_data[h][metric_key]
            rows.append(row_data)

    # Add Count row at the end
    count_row = {'Model': 'Count (N)', 'Metric': ''}
    for h in HORIZONS:
        count_row[f'h={h}'] = count_values.get(h, 0)
    rows.append(count_row)

    # Create DataFrame
    df_table = pd.DataFrame(rows)
    column_order = ['Model', 'Metric'] + [f'h={h}' for h in HORIZONS]
    df_table = df_table[column_order]

    # Format DataFrame for CSV output (apply formatting based on Metric type)
    df_formatted = df_table.copy()
    for h in HORIZONS:
        col = f'h={h}'
        # pandas>=3.0 no longer silently upcasts a float column on string assignment;
        # cast to object first so the formatted strings can be written in place.
        df_formatted[col] = df_formatted[col].astype(object)
        # Apply formatting row by row based on Metric
        for idx in df_formatted.index:
            metric = df_formatted.loc[idx, 'Metric']
            val = df_formatted.loc[idx, col]

            if pd.isna(val):
                df_formatted.loc[idx, col] = ''
            elif metric == '':  # Count row
                df_formatted.loc[idx, col] = f'{int(val):,}'
            elif metric == 'R²':
                df_formatted.loc[idx, col] = f'{val:.3f}'
            else:  # MAE, RMSE, Bias
                df_formatted.loc[idx, col] = f'{val:.2f}'

    # Save CSV with formatted values
    csv_path = output_path / 'table3_protocol_b_performance.csv'
    df_formatted.to_csv(csv_path, index=False)
    if verbose:
        print(f"  Saved CSV: {csv_path}")

    # Generate custom LaTeX with multirow and custom formatting
    caption = (
        "Protocol B: Global performance metrics across prediction horizons for all models "
        "(classical, machine learning, and deep learning). Evaluated on sequence-eligible "
        "subset with continuous 168h windows. Lower MAE/RMSE and higher R² indicate better "
        "performance. Note: Under Protocol B, sample counts are identical across models for "
        "each horizon; therefore Count is reported once at the bottom."
    )
    latex_code = _generate_table3_latex(df_table, caption, 'tab:protocol_b_performance')

    latex_path = output_path / 'table3_protocol_b_performance.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_code)
    if verbose:
        print(f"  Saved LaTeX: {latex_path}")


# ============================================================================
# FIGURE GENERATION FUNCTIONS
# ============================================================================

def generate_fig1_spatial_coverage(
    station_metadata: pd.DataFrame,
    metrics_a: Dict[str, pd.DataFrame],
    metrics_b: Dict[str, pd.DataFrame],
    output_dir: str = 'figures',
    dpi: int = 300,
    verbose: bool = False
) -> None:
    """
    Generate Figure 1: Spatial coverage map with station distribution.

    Parameters
    ----------
    station_metadata : pd.DataFrame
        Station coordinates and metadata
    metrics_a : dict
        Protocol A metrics
    metrics_b : dict
        Protocol B metrics
    output_dir : str
        Output directory
    dpi : int
        Figure DPI
    verbose : bool
        Print progress
    """
    if verbose:
        print("\nGenerating Figure 1: Spatial Coverage...")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_context("paper", font_scale=1.2)

    # Create figure with main panel and 3 insets
    fig = plt.figure(figsize=(14, 10))

    # Main panel: Map of stations
    ax_map = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)

    # Inset 1: Station counts by country
    ax_country = plt.subplot2grid((3, 3), (0, 2))

    # Inset 2: Test sample counts by country
    ax_samples = plt.subplot2grid((3, 3), (1, 2))

    # Inset 3: Station counts by area type
    ax_area = plt.subplot2grid((3, 3), (2, 2))

    # --- Main Map ---
    bbox = STYLE_CONFIG['europe_bbox']

    # Plot stations colored by country
    for country in COUNTRIES:
        country_data = station_metadata[station_metadata['Country'] == country]
        config = MODEL_CONFIGS['protocol_a']['persistence']  # Use persistence color as base
        ax_map.scatter(
            country_data['Longitude'],
            country_data['Latitude'],
            s=100,
            alpha=0.7,
            label=COUNTRY_NAMES[country],
            edgecolors='black',
            linewidths=0.5
        )

    ax_map.set_xlim(bbox['lon'])
    ax_map.set_ylim(bbox['lat'])
    ax_map.set_xlabel('Longitude')
    ax_map.set_ylabel('Latitude')
    ax_map.set_title('Air Quality Monitoring Stations Across Europe', fontsize=14, fontweight='bold')
    ax_map.legend(loc='lower left', framealpha=0.9)
    ax_map.grid(True, alpha=0.3)

    # Add country labels at centroids
    for country in COUNTRIES:
        country_data = station_metadata[station_metadata['Country'] == country]
        if len(country_data) > 0:
            centroid_lon = country_data['Longitude'].mean()
            centroid_lat = country_data['Latitude'].mean()
            ax_map.annotate(
                country,
                xy=(centroid_lon, centroid_lat),
                fontsize=12,
                fontweight='bold',
                ha='center',
                va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )

    # --- Inset 1: Station counts by country ---
    country_counts = station_metadata['Country'].value_counts().reindex(COUNTRIES, fill_value=0)
    bars = ax_country.bar(range(len(COUNTRIES)), country_counts.values, color='steelblue', edgecolor='black')
    ax_country.set_xticks(range(len(COUNTRIES)))
    ax_country.set_xticklabels(COUNTRIES, rotation=0)
    ax_country.set_ylabel('Station Count')
    ax_country.set_title('Stations by Country', fontsize=10, fontweight='bold')
    ax_country.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax_country.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=8)

    # --- Inset 2: Test sample counts (Protocol A vs B) ---
    # Get h=1 counts from persistence model
    sample_counts_a = {}
    sample_counts_b = {}

    if 'persistence' in metrics_a:
        df_a = metrics_a['persistence']
        for country in COUNTRIES:
            mask = (df_a['Scope'] == 'Country') & (df_a['Group'] == country) & (df_a['horizon'] == 1)
            row = df_a[mask]
            if len(row) > 0:
                sample_counts_a[country] = int(row['Count'].iloc[0])

    if 'persistence' in metrics_b:
        df_b = metrics_b['persistence']
        for country in COUNTRIES:
            mask = (df_b['Scope'] == 'Country') & (df_b['Group'] == country) & (df_b['horizon'] == 1)
            row = df_b[mask]
            if len(row) > 0:
                sample_counts_b[country] = int(row['Count'].iloc[0])

    x = np.arange(len(COUNTRIES))
    width = 0.35

    counts_a_list = [sample_counts_a.get(c, 0) for c in COUNTRIES]
    counts_b_list = [sample_counts_b.get(c, 0) for c in COUNTRIES]

    ax_samples.bar(x - width/2, counts_a_list, width, label='Protocol A', color='lightcoral', edgecolor='black')
    ax_samples.bar(x + width/2, counts_b_list, width, label='Protocol B', color='lightblue', edgecolor='black')

    ax_samples.set_xticks(x)
    ax_samples.set_xticklabels(COUNTRIES, rotation=0)
    ax_samples.set_ylabel('Test Samples (h=1)')
    ax_samples.set_title('Test Samples by Country', fontsize=10, fontweight='bold')
    ax_samples.legend(fontsize=8)
    ax_samples.grid(axis='y', alpha=0.3)
    ax_samples.ticklabel_format(style='plain', axis='y')

    # --- Inset 3: Station counts by area type ---
    area_counts = station_metadata['StationArea_Label'].value_counts().reindex(STATION_AREAS, fill_value=0)
    bars = ax_area.bar(range(len(STATION_AREAS)), area_counts.values, color='mediumseagreen', edgecolor='black')
    ax_area.set_xticks(range(len(STATION_AREAS)))
    ax_area.set_xticklabels([a.replace('-', '-\n') for a in STATION_AREAS], rotation=0, fontsize=8)
    ax_area.set_ylabel('Station Count')
    ax_area.set_title('Stations by Area Type', fontsize=10, fontweight='bold')
    ax_area.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax_area.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # Save figure
    fig_path = output_path / f'fig1_spatial_coverage.{STYLE_CONFIG["figure_format"]}'
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved figure: {fig_path}")


def generate_fig2_performance_vs_horizon(
    metrics_a: Dict[str, pd.DataFrame],
    metrics_b: Dict[str, pd.DataFrame],
    output_dir: str = 'figures',
    dpi: int = 300,
    verbose: bool = False
) -> None:
    """
    Generate Figure 2: Performance metrics vs prediction horizon.

    Parameters
    ----------
    metrics_a : dict
        Protocol A metrics
    metrics_b : dict
        Protocol B metrics
    output_dir : str
        Output directory
    dpi : int
        Figure DPI
    verbose : bool
        Print progress
    """
    if verbose:
        print("\nGenerating Figure 2: Performance vs Horizon...")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.1)

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Models to show
    models_a = ['persistence', 'gam', 'xgb', 'lgb'] # , 'rf', 'lr'
    models_b = ['persistence', 'gam', 'xgb', 'lgb', 'lstm_attention', 'lstm_cnn', 'itransformer'] # 'rf', 'lr', 'lstm_global', 'lstm_residual'

    # Panel A: Protocol A RMSE
    ax = axes[0, 0]
    for model_key in models_a:
        if model_key not in metrics_a:
            continue
        df = metrics_a[model_key]
        config = MODEL_CONFIGS['protocol_a'][model_key]
        mask = (df['Scope'] == 'Global') & (df['Group'] == 'All')
        global_df = df[mask].sort_values('horizon')
        fam = MODEL_FAMILY.get(model_key, 'dl')
        ax.plot(global_df['horizon'], global_df['RMSE'],
               color=config['color'], linestyle=FAMILY_LINESTYLE[fam],
               marker=FAMILY_MARKER[fam], label=config['label'],
               linewidth=2, markersize=6)

    ax.set_xlabel('Prediction Horizon (hours)')
    ax.set_ylabel('RMSE (μg/m³)')
    ax.set_title('A) Protocol A: RMSE vs Horizon', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(HORIZONS)

    # Panel B: Protocol B RMSE
    ax = axes[0, 1]
    for model_key in models_b:
        if model_key not in metrics_b:
            continue
        df = metrics_b[model_key]
        config = MODEL_CONFIGS['protocol_b'][model_key]
        mask = (df['Scope'] == 'Global') & (df['Group'] == 'All')
        global_df = df[mask].sort_values('horizon')
        fam = MODEL_FAMILY.get(model_key, 'dl')
        ax.plot(global_df['horizon'], global_df['RMSE'],
               color=config['color'], linestyle=FAMILY_LINESTYLE[fam],
               marker=FAMILY_MARKER[fam], label=config['label'],
               linewidth=2, markersize=6)

    ax.set_xlabel('Prediction Horizon (hours)')
    ax.set_ylabel('RMSE (μg/m³)')
    ax.set_title('B) Protocol B: RMSE vs Horizon', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(HORIZONS)

    # Panel C: Protocol A MAE
    ax = axes[1, 0]
    for model_key in models_a:
        if model_key not in metrics_a:
            continue
        df = metrics_a[model_key]
        config = MODEL_CONFIGS['protocol_a'][model_key]
        mask = (df['Scope'] == 'Global') & (df['Group'] == 'All')
        global_df = df[mask].sort_values('horizon')
        fam = MODEL_FAMILY.get(model_key, 'dl')
        ax.plot(global_df['horizon'], global_df['MAE'],
               color=config['color'], linestyle=FAMILY_LINESTYLE[fam],
               marker=FAMILY_MARKER[fam], label=config['label'],
               linewidth=2, markersize=6)

    ax.set_xlabel('Prediction Horizon (hours)')
    ax.set_ylabel('MAE (μg/m³)')
    ax.set_title('C) Protocol A: MAE vs Horizon', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(HORIZONS)

    # Panel D: Protocol B MAE
    ax = axes[1, 1]
    for model_key in models_b:
        if model_key not in metrics_b:
            continue
        df = metrics_b[model_key]
        config = MODEL_CONFIGS['protocol_b'][model_key]
        mask = (df['Scope'] == 'Global') & (df['Group'] == 'All')
        global_df = df[mask].sort_values('horizon')
        fam = MODEL_FAMILY.get(model_key, 'dl')
        ax.plot(global_df['horizon'], global_df['MAE'],
               color=config['color'], linestyle=FAMILY_LINESTYLE[fam],
               marker=FAMILY_MARKER[fam], label=config['label'],
               linewidth=2, markersize=6)

    ax.set_xlabel('Prediction Horizon (hours)')
    ax.set_ylabel('MAE (μg/m³)')
    ax.set_title('D) Protocol B: MAE vs Horizon', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(HORIZONS)

    plt.tight_layout()

    # Save figure
    fig_path = output_path / f'fig2_performance_vs_horizon.{STYLE_CONFIG["figure_format"]}'
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved figure: {fig_path}")


def generate_fig3_country_heatmap(
    metrics_b: Dict[str, pd.DataFrame],
    output_dir: str = 'figures',
    dpi: int = 300,
    verbose: bool = False
) -> None:
    """
    Generate Figure 3: Country robustness heatmap showing best RMSE per country/horizon.

    Parameters
    ----------
    metrics_b : dict
        Protocol B metrics
    output_dir : str
        Output directory
    dpi : int
        Figure DPI
    verbose : bool
        Print progress
    """
    if verbose:
        print("\nGenerating Figure 3: Country Robustness Heatmap...")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Build matrix: rows=countries, cols=horizons, values=min RMSE
    rmse_matrix = np.zeros((len(COUNTRIES), len(HORIZONS)))
    best_model_matrix = np.empty((len(COUNTRIES), len(HORIZONS)), dtype=object)

    for i, country in enumerate(COUNTRIES):
        for j, horizon in enumerate(HORIZONS):
            best_rmse = np.inf
            best_model = ''

            for model_key, df in metrics_b.items():
                mask = (df['Scope'] == 'Country') & (df['Group'] == country) & (df['horizon'] == horizon)
                rows = df[mask]
                if len(rows) > 0:
                    rmse = rows['RMSE'].iloc[0]
                    if rmse < best_rmse:
                        best_rmse = rmse
                        # Create short label
                        label = MODEL_CONFIGS['protocol_b'][model_key]['label']
                        if 'LSTM' in label:
                            best_model = label.replace('LSTM-', 'L-')
                        elif label == 'Persistence':
                            best_model = 'Pers'
                        else:
                            best_model = label

            rmse_matrix[i, j] = best_rmse if best_rmse < np.inf else np.nan
            best_model_matrix[i, j] = best_model

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        rmse_matrix,
        annot=best_model_matrix,
        fmt='',
        cmap=STYLE_CONFIG['heatmap_cmap'],
        cbar_kws={'label': 'Best RMSE (μg/m³)'},
        xticklabels=[f'{h}h' for h in HORIZONS],
        yticklabels=[COUNTRY_NAMES[c] for c in COUNTRIES],
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )

    ax.set_title('Country Robustness: Best RMSE per Horizon (Protocol B)',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Prediction Horizon', fontsize=12)
    ax.set_ylabel('Country', fontsize=12)

    plt.tight_layout()

    # Save figure
    fig_path = output_path / f'fig3_country_robustness.{STYLE_CONFIG["figure_format"]}'
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved figure: {fig_path}")


def generate_fig4_area_heatmap(
    metrics_b: Dict[str, pd.DataFrame],
    output_dir: str = 'figures',
    dpi: int = 300,
    verbose: bool = False
) -> None:
    """
    Generate Figure 4: Station area robustness heatmap showing best RMSE per area/horizon.

    Parameters
    ----------
    metrics_b : dict
        Protocol B metrics
    output_dir : str
        Output directory
    dpi : int
        Figure DPI
    verbose : bool
        Print progress
    """
    if verbose:
        print("\nGenerating Figure 4: Station Area Robustness Heatmap...")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Build matrix: rows=areas, cols=horizons, values=min RMSE
    rmse_matrix = np.zeros((len(STATION_AREAS), len(HORIZONS)))
    best_model_matrix = np.empty((len(STATION_AREAS), len(HORIZONS)), dtype=object)

    for i, area in enumerate(STATION_AREAS):
        for j, horizon in enumerate(HORIZONS):
            best_rmse = np.inf
            best_model = ''

            for model_key, df in metrics_b.items():
                mask = (df['Scope'] == 'StationArea') & (df['Group'] == area) & (df['horizon'] == horizon)
                rows = df[mask]
                if len(rows) > 0:
                    rmse = rows['RMSE'].iloc[0]
                    if rmse < best_rmse:
                        best_rmse = rmse
                        # Create short label
                        label = MODEL_CONFIGS['protocol_b'][model_key]['label']
                        if 'LSTM' in label:
                            best_model = label.replace('LSTM-', 'L-')
                        elif label == 'Persistence':
                            best_model = 'Pers'
                        else:
                            best_model = label

            rmse_matrix[i, j] = best_rmse if best_rmse < np.inf else np.nan
            best_model_matrix[i, j] = best_model

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        rmse_matrix,
        annot=best_model_matrix,
        fmt='',
        cmap=STYLE_CONFIG['heatmap_cmap'],
        cbar_kws={'label': 'Best RMSE (μg/m³)'},
        xticklabels=[f'{h}h' for h in HORIZONS],
        yticklabels=[area.capitalize() for area in STATION_AREAS],
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )

    ax.set_title('Station Area Robustness: Best RMSE per Horizon (Protocol B)',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Prediction Horizon', fontsize=12)
    ax.set_ylabel('Station Area Type', fontsize=12)

    plt.tight_layout()

    # Save figure
    fig_path = output_path / f'fig4_area_robustness.{STYLE_CONFIG["figure_format"]}'
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved figure: {fig_path}")


def generate_fig5_bias_analysis(
    metrics_b: Dict[str, pd.DataFrame],
    output_dir: str = 'figures',
    dpi: int = 300,
    verbose: bool = False
) -> None:
    """
    Generate Figure 5: Bias progression with prediction horizon.

    Parameters
    ----------
    metrics_b : dict
        Protocol B metrics
    output_dir : str
        Output directory
    dpi : int
        Figure DPI
    verbose : bool
        Print progress
    """
    if verbose:
        print("\nGenerating Figure 5: Bias Analysis...")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.1)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Models to show
    models_to_show = ['lstm_global', 'lstm_residual', 'lstm_attention', 'lstm_cnn']

    for model_key in models_to_show:
        if model_key not in metrics_b:
            continue

        df = metrics_b[model_key]
        config = MODEL_CONFIGS['protocol_b'][model_key]

        # Get global metrics
        mask = (df['Scope'] == 'Global') & (df['Group'] == 'All')
        global_df = df[mask].sort_values('horizon')

        ax.plot(
            global_df['horizon'],
            global_df['Bias'],
            marker='o',
            label=config['label'],
            linewidth=2,
            markersize=6,
            color=config['color']
        )

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Prediction Horizon (hours)', fontsize=12)
    ax.set_ylabel('Bias (μg/m³)', fontsize=12)
    ax.set_title('Prediction Bias vs Horizon (Protocol B - DL family)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(HORIZONS)

    # Add text annotation
    ax.text(
        0.98, 0.02,
        'Positive bias = over-prediction\nNegative bias = under-prediction',
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()

    # Save figure
    fig_path = output_path / f'fig5_bias_analysis.{STYLE_CONFIG["figure_format"]}'
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved figure: {fig_path}")


def generate_fig6_skill_comparison(
    metrics_a_skill: Dict[str, pd.DataFrame],
    metrics_b_skill: Dict[str, pd.DataFrame],
    output_dir: str = 'figures',
    dpi: int = 300,
    verbose: bool = False
) -> None:
    """
    Generate Figure 6: Model skill comparison vs persistence baseline.

    Shows RMSE_skill and MAE_skill progression across horizons for both protocols.
    Negative values indicate improvement over persistence baseline.

    Parameters
    ----------
    metrics_a_skill : dict
        Protocol A metrics with skill scores
    metrics_b_skill : dict
        Protocol B metrics with skill scores
    output_dir : str
        Output directory
    dpi : int
        Figure DPI
    verbose : bool
        Print progress
    """
    if verbose:
        print("\nGenerating Figure 6: Skill Comparison...")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.1)

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Models to show
    models_a = ['lr', 'gam', 'rf', 'xgb', 'lgb']
    models_b = ['lr', 'gam', 'rf', 'xgb', 'lgb', 'lstm_global', 'lstm_residual', 'lstm_attention', 'lstm_cnn']

    # Panel A: Protocol A RMSE_skill
    ax = axes[0, 0]
    # Add horizontal reference line at y=0 (persistence baseline)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Persistence')

    for model_key in models_a:
        if model_key not in metrics_a_skill:
            continue
        df = metrics_a_skill[model_key]
        config = MODEL_CONFIGS['protocol_a'][model_key]
        mask = (df['Scope'] == 'Global') & (df['Group'] == 'All')
        global_df = df[mask].sort_values('horizon')
        ax.plot(global_df['horizon'], global_df['RMSE_skill'],
               marker='o', label=config['label'], linewidth=2, markersize=6,
               color=config['color'])

    ax.set_xlabel('Prediction Horizon (hours)')
    ax.set_ylabel('RMSE Skill Score (μg/m³)')
    ax.set_title('A) Protocol A: RMSE Skill vs Horizon', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(HORIZONS)
    # Add annotation
    ax.text(0.98, 0.98, 'Negative = Better than Persistence',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel B: Protocol B RMSE_skill
    ax = axes[0, 1]
    # Add horizontal reference line at y=0 (persistence baseline)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Persistence')

    for model_key in models_b:
        if model_key not in metrics_b_skill:
            continue
        df = metrics_b_skill[model_key]
        config = MODEL_CONFIGS['protocol_b'][model_key]
        mask = (df['Scope'] == 'Global') & (df['Group'] == 'All')
        global_df = df[mask].sort_values('horizon')
        ax.plot(global_df['horizon'], global_df['RMSE_skill'],
               marker='o', label=config['label'], linewidth=2, markersize=6,
               color=config['color'])

    ax.set_xlabel('Prediction Horizon (hours)')
    ax.set_ylabel('RMSE Skill Score (μg/m³)')
    ax.set_title('B) Protocol B: RMSE Skill vs Horizon', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(HORIZONS)
    # Add annotation
    ax.text(0.98, 0.98, 'Negative = Better than Persistence',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel C: Protocol A MAE_skill
    ax = axes[1, 0]
    # Add horizontal reference line at y=0 (persistence baseline)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Persistence')

    for model_key in models_a:
        if model_key not in metrics_a_skill:
            continue
        df = metrics_a_skill[model_key]
        config = MODEL_CONFIGS['protocol_a'][model_key]
        mask = (df['Scope'] == 'Global') & (df['Group'] == 'All')
        global_df = df[mask].sort_values('horizon')
        ax.plot(global_df['horizon'], global_df['MAE_skill'],
               marker='o', label=config['label'], linewidth=2, markersize=6,
               color=config['color'])

    ax.set_xlabel('Prediction Horizon (hours)')
    ax.set_ylabel('MAE Skill Score (μg/m³)')
    ax.set_title('C) Protocol A: MAE Skill vs Horizon', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(HORIZONS)
    # Add annotation
    ax.text(0.98, 0.98, 'Negative = Better than Persistence',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel D: Protocol B MAE_skill
    ax = axes[1, 1]
    # Add horizontal reference line at y=0 (persistence baseline)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Persistence')

    for model_key in models_b:
        if model_key not in metrics_b_skill:
            continue
        df = metrics_b_skill[model_key]
        config = MODEL_CONFIGS['protocol_b'][model_key]
        mask = (df['Scope'] == 'Global') & (df['Group'] == 'All')
        global_df = df[mask].sort_values('horizon')
        ax.plot(global_df['horizon'], global_df['MAE_skill'],
               marker='o', label=config['label'], linewidth=2, markersize=6,
               color=config['color'])

    ax.set_xlabel('Prediction Horizon (hours)')
    ax.set_ylabel('MAE Skill Score (μg/m³)')
    ax.set_title('D) Protocol B: MAE Skill vs Horizon', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(HORIZONS)
    # Add annotation
    ax.text(0.98, 0.98, 'Negative = Better than Persistence',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save figure
    fig_path = output_path / f'fig6_skill_comparison.{STYLE_CONFIG["figure_format"]}'
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved figure: {fig_path}")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate publication-ready tables and figures from PM2.5 model evaluation metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate everything
  python generate_publication_outputs.py

  # Generate only tables
  python generate_publication_outputs.py --outputs tables

  # Generate specific figures
  python generate_publication_outputs.py --outputs figures --figures fig1 fig2

  # With validation
  python generate_publication_outputs.py --validate -v
        """
    )

    parser.add_argument(
        '--outputs',
        nargs='+',
        choices=['tables', 'figures', 'all'],
        default=['all'],
        help='Type of outputs to generate (default: all)'
    )

    parser.add_argument(
        '--tables',
        nargs='+',
        choices=['table1', 'table2', 'table3', 'all'],
        default=['all'],
        help='Specific tables to generate (default: all)'
    )

    parser.add_argument(
        '--figures',
        nargs='+',
        choices=['fig1', 'fig2', 'fig3', 'fig4', 'fig5', 'fig6', 'all'],
        default=['all'],
        help='Specific figures to generate (default: all)'
    )

    parser.add_argument(
        '--table-dir',
        default='../tables',
        help='Output directory for tables (default: ../tables)'
    )

    parser.add_argument(
        '--figure-dir',
        default='../figures',
        help='Output directory for figures (default: ../figures)'
    )

    parser.add_argument(
        '--results-dir',
        default='../results',
        help='Input directory with metrics CSV files (default: ../results)'
    )

    parser.add_argument(
        '--dataset-path',
        default='../ml_ready_dataset_full_realistic.csv',
        help='Path to ML dataset for station metadata (default: ../ml_ready_dataset_full_realistic.csv)'
    )

    parser.add_argument(
        '--coverage-path',
        default='../dataset_build/station_train_test_coverage.csv',
        help='Path to coverage statistics CSV (default: ../dataset_build/station_train_test_coverage.csv)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Figure DPI (default: 300)'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate data consistency before generating outputs'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    return parser.parse_args()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    args = parse_args()

    # Print header
    print("=" * 80)
    print("Publication-Ready Tables & Figures Generator")
    print("PM2.5 Air Quality Forecasting Project")
    print("=" * 80)

    # Create output directories
    Path(args.table_dir).mkdir(exist_ok=True)
    Path(args.figure_dir).mkdir(exist_ok=True)

    # Load data
    if args.verbose:
        print("\n" + "=" * 80)
        print("LOADING DATA")
        print("=" * 80)

    metrics_a = load_metrics(protocol='A', results_dir=args.results_dir, verbose=args.verbose)
    metrics_b = load_metrics(protocol='B', results_dir=args.results_dir, verbose=args.verbose)
    station_metadata = load_station_metadata(dataset_path=args.dataset_path, verbose=args.verbose)
    coverage_info = load_coverage_info(coverage_path=args.coverage_path, verbose=args.verbose)

    # Validate if requested
    if args.validate:
        if args.verbose:
            print("\n" + "=" * 80)
            print("VALIDATION")
            print("=" * 80)

        inconsistencies_a = validate_count_consistency(metrics_a, 'A', verbose=args.verbose)
        inconsistencies_b = validate_count_consistency(metrics_b, 'B', verbose=args.verbose)

        if inconsistencies_a or inconsistencies_b:
            print("\n⚠ Warning: Found count inconsistencies (see above)")

    # Generate tables
    if 'all' in args.outputs or 'tables' in args.outputs:
        if args.verbose:
            print("\n" + "=" * 80)
            print("GENERATING TABLES")
            print("=" * 80)

        if 'all' in args.tables or 'table1' in args.tables:
            generate_table1_dataset_overview(
                station_metadata, coverage_info, metrics_a, metrics_b,
                output_dir=args.table_dir, verbose=args.verbose
            )

        if 'all' in args.tables or 'table2' in args.tables:
            generate_table2_protocol_a_performance(
                metrics_a, output_dir=args.table_dir, verbose=args.verbose
            )

        if 'all' in args.tables or 'table3' in args.tables:
            generate_table3_protocol_b_performance(
                metrics_b, output_dir=args.table_dir, verbose=args.verbose
            )

    # Generate figures
    if 'all' in args.outputs or 'figures' in args.outputs:
        if args.verbose:
            print("\n" + "=" * 80)
            print("GENERATING FIGURES")
            print("=" * 80)

        if 'all' in args.figures or 'fig1' in args.figures:
            generate_fig1_spatial_coverage(
                station_metadata, metrics_a, metrics_b,
                output_dir=args.figure_dir, dpi=args.dpi, verbose=args.verbose
            )

        if 'all' in args.figures or 'fig2' in args.figures:
            generate_fig2_performance_vs_horizon(
                metrics_a, metrics_b,
                output_dir=args.figure_dir, dpi=args.dpi, verbose=args.verbose
            )

        if 'all' in args.figures or 'fig3' in args.figures:
            generate_fig3_country_heatmap(
                metrics_b, output_dir=args.figure_dir, dpi=args.dpi, verbose=args.verbose
            )

        if 'all' in args.figures or 'fig4' in args.figures:
            generate_fig4_area_heatmap(
                metrics_b, output_dir=args.figure_dir, dpi=args.dpi, verbose=args.verbose
            )

        if 'all' in args.figures or 'fig5' in args.figures:
            generate_fig5_bias_analysis(
                metrics_b, output_dir=args.figure_dir, dpi=args.dpi, verbose=args.verbose
            )

        if 'all' in args.figures or 'fig6' in args.figures:
            # Load metrics with skill scores for fig6
            metrics_a_skill = load_metrics_with_skill(protocol='A', results_with_skill_dir='../results_with_skill', verbose=args.verbose)
            metrics_b_skill = load_metrics_with_skill(protocol='B', results_with_skill_dir='../results_with_skill', verbose=args.verbose)
            generate_fig6_skill_comparison(
                metrics_a_skill, metrics_b_skill,
                output_dir=args.figure_dir, dpi=args.dpi, verbose=args.verbose
            )

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"[OK] Tables saved to: {args.table_dir}/")
    print(f"[OK] Figures saved to: {args.figure_dir}/")
    print("\n[OK] Publication outputs generated successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
