# PM2.5 Air Quality Prediction Dataset

This repository contains a complete pipeline for PM2.5 air quality prediction, from raw data acquisition through model training, evaluation, statistical analysis, and publication-ready outputs. The pipeline downloads European Environment Agency (EEA) air quality data, trains multiple forecasting models (statistical, ML, and deep learning), and generates comprehensive evaluation results for research publication.

## Data & Models

Pre-trained models and processed datasets are available on Hugging Face:

- **Dataset**: [huggingface.co/datasets/cosuleabianca/eea-pm25-dataset](https://huggingface.co/datasets/cosuleabianca/eea-pm25-forecasting)
- **Models**: [huggingface.co/cosuleabianca/eea-pm25-models](https://huggingface.co/cosuleabianca/eea-pm25)

## Table of Contents

- [Data & Models](#data--models)
- [Quick Start: Reproduce Paper Results](#quick-start-reproduce-paper-results)
- [Workflow Visualization](#workflow-visualization)
- [Overview](#overview)
- [Dataset Creation Pipeline](#dataset-creation-pipeline)
  - [Step 1: Raw Data Acquisition](#step-1-raw-data-acquisition)
  - [Step 2: Site Filtering](#step-2-site-filtering)
  - [Step 3: Data Processing](#step-3-data-processing)
  - [Step 4: ML Dataset Preparation](#step-4-ml-dataset-preparation)
  - [Step 5: Data Quality Filtering](#step-5-data-quality-filtering)
- [Model Training](#model-training)
  - [Training Architecture: Global Models](#training-architecture-global-models)
  - [Multi-Horizon Forecasting](#multi-horizon-forecasting)
  - [Available Models](#available-models)
  - [Feature Configuration](#feature-configuration)
  - [Training Output Structure](#training-output-structure)
- [Evaluation Protocols](#evaluation-protocols)
  - [Protocol A: Full-Coverage Evaluation](#protocol-a-full-coverage-evaluation)
  - [Protocol B: Sequence-Eligible Subset](#protocol-b-sequence-eligible-subset)
  - [Metrics File Structure](#metrics-file-structure)
- [Statistical Analysis and Interpretability](#statistical-analysis-and-interpretability)
  - [Diebold-Mariano Test](#1-diebold-mariano-test)
  - [SHAP Feature Importance](#2-shap-feature-importance)
  - [Occlusion Test](#3-occlusion-test)
- [Publication-Ready Outputs](#publication-ready-outputs)
  - [Model Skill Computation](#model-skill-computation)
  - [Publication Tables and Figures](#publication-tables-and-figures)
- [Output Structure](#output-structure)
- [Requirements](#requirements)
- [Reproducibility](#reproducibility)
- [Dataset Characteristics](#dataset-characteristics)
- [Citation](#citation)

---

## Quick Start: Reproduce Paper Results

For researchers wanting to reproduce the paper results, run these commands in sequence:

**⚠️ TIME WARNING**: The complete pipeline with all models takes **~6 days** due to deep learning training time (136 hours for LSTM models).

### Prerequisites
- Python 3.12+
- 30+ GB free disk space
- Internet connection for API downloads
- GPU recommended for deep learning models (LSTM training)

### Complete Pipeline (Data → Models → Publication)

**Phase 1: Data Acquisition (2-4 hours first run)**
```bash
# Download raw pollutant data from EEA API
python dataset_build/src/download_pollutants.py

# Filter to sites with PM2.5 measurements
python dataset_build/src/filter_pm25_sites.py

# Merge pollutants and add site metadata
python dataset_build/src/process_data.py
```

**Phase 2: Feature Engineering (30-60 min first run, 5-10 min cached)**
```bash
# Create ML dataset with lag/rolling features and weather data
python dataset_build/src/prepare_ml_dataset.py
```

**Phase 3: Data Quality Filtering (5-10 min)**
```bash
# Compute station coverage statistics
python dataset_build/src/coverage_only_v6.py

# Create final filtered dataset (moderate filtering)
python dataset_build/src/dataset_full_realistic_v6.py
```

**Phase 4: Model Training (140+ hours for all models, 3 hours for classical only)**
```bash
# Baseline models (~2 min each)
python train_models/global/persistence.py
python train_models/global/lr_global.py

# Statistical and tree-based models (3-98 min each)
python train_models/global/gam_global.py        # ~98 min
python train_models/global/rf_global.py         # ~36 min
python train_models/global/xgb_global.py        # ~3 min (with cached hyperparams)
python train_models/global/lgb_global.py        # ~52 min

# Deep learning models (13-46 hours each)
python train_models/global/lstm_global.py                # ~13 hours
python train_models/global/lstm_global_residual.py      # ~46 hours
python train_models/global/lstm_global_attention.py     # ~34 hours
python train_models/global/lstm_cnn.py                  # ~43 hours
```

**Phase 5: Statistical Analysis (30-60 min total)**
```bash
# Test statistical significance of model differences
python tests/diebold_mariano_tests.py

# Feature importance analysis
python tests/shap_feature_importance.py

# LSTM interpretability analysis
python tests/occlusion_test_dl.py
```

**Phase 6: Publication Outputs (5-15 min)**
```bash
# Compute model skill vs persistence baseline
python scripts/compute_model_skill.py

# Generate tables and figures
python scripts/generate_publication_outputs.py
```

### Expected Runtime
- Data pipeline: 2-4 hours (first run with API downloads)
- Model training: **~140 hours** (all 10 models, all horizons)
  - Classical models: ~3 hours (Persistence, LR, GAM, RF, XGBoost, LightGBM)
  - Deep learning models: ~136 hours (4 LSTM variants)
- Statistical analysis: 30-60 minutes
- Publication outputs: 5-15 minutes
- **Total**: **~143-147 hours** (~6 days)

### Output Verification

After each phase, verify outputs:

```bash
# Phase 1-3: Check data files
ls -lh data/*.parquet
ls -lh ml_ready_dataset_full_realistic.csv

# Phase 4: Check model files and metrics
ls -lh models/*/
ls -lh results/*_metrics*.csv

# Phase 5: Check statistical analysis results
ls -lh results/diebold_mariano_results.csv
ls -lh results/shap_*.csv
ls -lh results/occlusion_test_results.csv

# Phase 6: Check publication outputs
ls -lh tables/*.csv tables/*.tex
ls -lh figures/*.png
ls -lh results_with_skill/*_metrics*.csv
```

---

## Workflow Visualization

### Complete Pipeline Flowchart

```
┌─────────────────────────────────────────────────────────────────┐
│                 PHASE 1: DATA ACQUISITION (2-4h)                 │
└─────────────────────────────────────────────────────────────────┘
  │
  ├─► download_pollutants.py ──► data/PM2.5.parquet (raw)
  ├─► download_pollutants.py ──► data/NO2.parquet (raw)
  ├─► download_pollutants.py ──► data/PM10.parquet (raw)
  │
  ├─► filter_pm25_sites.py ──► data/*_filtered.parquet
  │
  └─► process_data.py ──► processed_air_quality_data.csv
  │
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 2: FEATURE ENGINEERING (30-60min)             │
└─────────────────────────────────────────────────────────────────┘
  │
  └─► prepare_ml_dataset.py ──► ml_ready_dataset_v6.csv
                                 (+ weather data from Open-Meteo API)
  │
┌─────────────────────────────────────────────────────────────────┐
│            PHASE 3: DATA QUALITY FILTERING (5-10min)             │
└─────────────────────────────────────────────────────────────────┘
  │
  ├─► coverage_only_v6.py ──► station_train_test_coverage.csv
  │
  └─► dataset_full_realistic_v6.py ──► ml_ready_dataset_full_realistic.csv
  │
┌─────────────────────────────────────────────────────────────────┐
│               PHASE 4: MODEL TRAINING (8-12h)                    │
└─────────────────────────────────────────────────────────────────┘
  │
  ├─► Baselines (persistence, lr_global)
  ├─► Statistical/ML (gam, rf, xgb, lgb)
  └─► Deep Learning (4 LSTM variants)
  │
  ├─► models/{model_name}_models/*.{pkl,keras}
  ├─► results/{model}_metrics*.csv
  └─► results/{model}_predictions_sample.csv
  │
┌─────────────────────────────────────────────────────────────────┐
│           PHASE 5: STATISTICAL ANALYSIS (30-60min)               │
└─────────────────────────────────────────────────────────────────┘
  │
  ├─► diebold_mariano_tests.py ──► results/diebold_mariano_results.csv
  ├─► shap_feature_importance.py ──► results/shap_*.csv
  └─► occlusion_test_dl.py ──► results/occlusion_test_results.csv
  │
┌─────────────────────────────────────────────────────────────────┐
│            PHASE 6: PUBLICATION OUTPUTS (5-15min)                │
└─────────────────────────────────────────────────────────────────┘
  │
  ├─► compute_model_skill.py ──► results_with_skill/*_metrics*.csv
  │
  └─► generate_publication_outputs.py ──► tables/*.{csv,tex}
                                      └──► figures/*.png
```

### File Dependency Graph

```
EEA API
  │
  └─► data/PM2.5.parquet, data/NO2.parquet, data/PM10.parquet
        │
        └─► data/*_filtered.parquet
              │
              └─► processed_air_quality_data.csv
                    │
                    └─► ml_ready_dataset_v6.csv ◄─── Open-Meteo API (weather)
                          │
                          └─► station_train_test_coverage.csv
                                │
                                └─► ml_ready_dataset_full_realistic.csv
                                      │
                                      ├─► Train: persistence, lr, gam, rf, xgb, lgb
                                      │     │
                                      │     └─► results/{model}_metrics.csv (Protocol A)
                                      │
                                      ├─► Train: lstm_global, lstm_residual, lstm_attention, lstm_cnn
                                      │     │
                                      │     ├─► results/lstm_*_metrics.csv (Protocol B)
                                      │     └─► Re-evaluate classical models on DL subset
                                      │           │
                                      │           └─► results/{model}_dl_metrics.csv (Protocol B)
                                      │
                                      └─► Statistical Analysis
                                            │
                                            ├─► diebold_mariano_results.csv
                                            ├─► shap_*.csv
                                            └─► occlusion_test_results.csv
                                                  │
                                                  └─► Publication Outputs
                                                        │
                                                        ├─► results_with_skill/*
                                                        ├─► tables/*.csv, tables/*.tex
                                                        └─► figures/*.png
```

---

## Overview

This is a complete pipeline for PM2.5 air quality prediction research, from raw data acquisition through publication-ready outputs. The pipeline:

1. **Downloads and processes** air quality data from the European Environment Agency (EEA) covering 5 countries and 7 years (2018-2024)
2. **Engineers comprehensive features** including lag features, rolling statistics, weather data, and temporal encodings
3. **Trains 10 forecasting models** across three categories: baseline (Persistence, Linear Regression), tree-based ML (GAM, Random Forest, XGBoost, LightGBM), and deep learning (4 LSTM variants)
4. **Predicts PM2.5 concentrations** at 5 time horizons (1h, 3h, 6h, 12h, 24h ahead)
5. **Evaluates models** using two protocols: Protocol A (full geographic coverage) and Protocol B (sequence-eligible subset for fair DL comparison)
6. **Performs statistical analysis** including significance testing, feature importance, and interpretability studies
7. **Generates publication outputs** including LaTeX tables, high-resolution figures, and skill scores

The dataset creation process involves five main steps:

1. **Raw Data Acquisition**: Downloading hourly air quality measurements from the EEA API
2. **Site Filtering**: Retaining only monitoring sites with PM2.5 measurements
3. **Data Processing**: Merging pollutants and enriching with site metadata
4. **ML Dataset Preparation**: Feature engineering with weather data integration
5. **Data Quality Filtering**: Selecting stations with sufficient completeness

This README provides both a quick-start guide for reproducing results and detailed documentation of each pipeline component.

---

## Dataset Creation Pipeline

### Directory Structure

```
eea-pm25_v2/
├── dataset_build/
│   ├── src/
│   │   ├── download_pollutants.py     # Step 1: Download raw data
│   │   ├── filter_pm25_sites.py       # Step 2: Filter sites
│   │   ├── process_data.py            # Step 3: Process data
│   │   └── prepare_ml_dataset.py      # Step 4: Feature engineering
│   ├── processed_air_quality_data.csv # Output of Step 3
│   ├── ml_features.txt                # Output of Step 4
│   └── ml_dataset_summary.txt         # Output of Step 4
├── utils/
│   ├── config.yaml                    # Configuration file
│   └── config_utils.py                # Configuration utilities
├── data/                              # Raw and filtered data
└── ml_ready_dataset_v6.csv            # Final ML dataset
```

### Step 1: Raw Data Acquisition

**Script:** `dataset_build/src/download_pollutants.py`

This script downloads air quality data from the EEA API for specified cities, pollutants, and date ranges.

#### Configuration

All parameters are specified in `utils/config.yaml`:

```yaml
# Cities to download (exact names as in EEA API)
cities:
  - country: AT
    city: Wien
  - country: FR
    city: Paris (greater city)
  - country: ES
    city: Madrid
  - country: BE
    city: Antwerpen
  - country: FI
    city: Helsinki / Helsingfors (greater city)

# Date range
date_range:
  start: "2018-01-01"
  end: "2024-12-31"

# Pollutants
pollutants:
  - name: PM2.5
    code: 6001
    uri: http://dd.eionet.europa.eu/vocabulary/aq/pollutant/6001
    required: true
  - name: NO2
    code: 8
    uri: http://dd.eionet.europa.eu/vocabulary/aq/pollutant/8
  - name: PM10
    code: 5
    uri: http://dd.eionet.europa.eu/vocabulary/aq/pollutant/5
```

#### Data Source

- **API Endpoint**: EEA Downloads API (`https://eeadmz1-downloads-api-appservice.azurewebsites.net/`)
- **Dataset**: E1a (real-time verified data, dataset ID: 2)
- **Aggregation**: Hourly
- **Format**: Parquet (compressed columnar format)

#### Download Strategy

To avoid API rate limits, data is downloaded in **6-month chunks**:
- 2018-H1 (Jan-Jun), 2018-H2 (Jul-Dec)
- 2019-H1, 2019-H2
- ... and so on through 2024

Each pollutant is downloaded separately and saved to `data/{POLLUTANT_NAME}.parquet`.

#### Output Files

```
data/
├── PM2.5.parquet
├── NO2.parquet
└── PM10.parquet
```

#### Execution

```bash
python dataset_build/src/download_pollutants.py
```

**Note**: If files already exist and `skip_existing_downloads: true` is set in `utils/config.yaml`, the script will skip re-downloading.

---

### Step 2: Site Filtering

**Script:** `dataset_build/src/filter_pm25_sites.py`

This script identifies all monitoring sites that measure PM2.5 and filters data from other pollutants to include only those sites.

#### Rationale

Since PM2.5 is our target variable, we only need data from sites that measure it. Other pollutants (NO2, PM10) are only useful at PM2.5 monitoring sites.

#### Process

1. **Identify PM2.5 Sites**
   - Load `data/PM2.5.parquet`
   - Extract unique site identifiers (Country/SiteNumber)
   - Parse sampling point strings using country-specific regex patterns

2. **Site Identification Patterns**

   Each country uses a different sampling point naming convention:

   | Country | Pattern | Example |
   |---------|---------|---------|
   | Austria (AT) | `.09.{CODE}.` | `AT/SPO.09.A23.65516.6001.1` → A23 |
   | Belgium (BE) | `BE{LETTERS+DIGITS}` | `BE/SPO-BETR817_06001_100` → BETR817 |
   | Spain (ES) | `SP_{CITY_CODE}` | `ES/SP_28079056_9_47` → 28079056 |
   | Finland (FI) | `FI{5 DIGITS}` | `FI/SPO-FI00208_06001_100` → FI00208 |
   | France (FR) | `FR{5 DIGITS}` | `FR/SPO-FR04024_6001` → FR04024 |

3. **Filter Other Pollutants**
   - For each pollutant (NO2, PM10), load the raw parquet file
   - Extract site identifiers using the same patterns
   - Filter to only include sites present in the PM2.5 site list
   - Save filtered data to `data/{POLLUTANT_NAME}_filtered.parquet`

#### Output Files

```
data/
├── pm25_sites.txt                  # List of all PM2.5 monitoring sites
├── PM2.5_filtered.parquet          # Same as PM2.5.parquet (all sites included)
├── NO2_filtered.parquet            # NO2 data only for PM2.5 sites
└── PM10_filtered.parquet           # PM10 data only for PM2.5 sites
```

#### Execution

```bash
python dataset_build/src/filter_pm25_sites.py
```

**Prerequisites**: Must run `dataset_build/src/download_pollutants.py` first.

---

### Step 3: Data Processing

**Script:** `dataset_build/src/process_data.py`

This script combines filtered pollutant data and enriches it with site metadata.

#### Process

1. **Load Filtered Pollutants**
   - Read `data/{POLLUTANT}_filtered.parquet` for each pollutant
   - Each file contains: `Samplingpoint`, `Start` (timestamp), `Value`, `PollutantName`

2. **Combine Pollutants (Long Format)**
   - Concatenate all pollutant dataframes vertically
   - Result: One row per measurement with columns: `Samplingpoint`, `Start`, `Value`, `PollutantName`

3. **Load and Merge Site Metadata**

   Site metadata is loaded from `site_metadata.csv` (obtained from EEA):

   **Metadata Attributes:**
   - **Geographic**: Latitude, Longitude, Altitude
   - **Station Type**: Background, Traffic, Industrial (one-hot encoded)
   - **Station Area**: Urban, Suburban, Rural (one-hot encoded)

   **Matching Strategy:**

   Metadata matching uses country-specific approaches due to varying sampling point formats:

   - **FI, FR, BE**: Extract EoI code directly from sampling point string
   - **AT**: Extract numeric code and match to metadata via `Sampling Point Id`
   - **ES**: Extract city code (Nat Code) and match to metadata

   Example for Austria:
   ```python
   # Samplingpoint: AT/SPO.09.A23.65516.6001.1
   # Extract: 65516 (numeric code before pollutant code)
   # Match to metadata: Sampling Point Id contains ".65516."
   # Result: EoI code like "AT00001"
   ```

4. **One-Hot Encode Categorical Features**
   - `Station_Type` → `StationType_Background`, `StationType_Traffic`, `StationType_Industrial`
   - `Station_Area` → `StationArea_urban`, `StationArea_suburban`, `StationArea_rural`

5. **Pivot to Wide Format**

   Transform from long format (one row per measurement) to wide format (one row per time/location):

   **Before (Long):**
   ```
   Start               Country  SiteNumber  PollutantName  Value
   2018-01-01 00:00:00 AT       A23         PM2.5          12.3
   2018-01-01 00:00:00 AT       A23         NO2            45.6
   2018-01-01 00:00:00 AT       A23         PM10           23.4
   ```

   **After (Wide):**
   ```
   Start               Country  SiteNumber  Latitude  Longitude  PM2.5  NO2   PM10
   2018-01-01 00:00:00 AT       A23         48.2      16.4       12.3   45.6  23.4
   ```

6. **Column Ordering**

   Final column order:
   1. Temporal: `Start`, `Country`, `SiteNumber`
   2. Geographic metadata: `Latitude`, `Longitude`, `Altitude`
   3. Categorical metadata: `StationType_*`, `StationArea_*`
   4. Pollutants: `PM2.5`, `NO2`, `PM10`

#### Output Files

```
dataset_build/
└── processed_air_quality_data.csv
```

Columns: `Start`, `Country`, `SiteNumber`, `Latitude`, `Longitude`, `Altitude`, `StationType_*`, `StationArea_*`, `PM2.5`, `NO2`, `PM10`

#### Execution

```bash
python dataset_build/src/process_data.py
```

**Prerequisites**: Must run `dataset_build/src/filter_pm25_sites.py` first. Requires `dataset_build/site_metadata.csv` for metadata enrichment.

---

### Step 4: ML Dataset Preparation

**Script:** `dataset_build/src/prepare_ml_dataset.py`

This is the most critical step, performing comprehensive feature engineering to create the final machine learning dataset.

#### Overview

This script transforms processed air quality data into a ML-ready dataset with:
- **Target variable**: PM2.5 concentration (create prediction target in training scripts as needed)
- **Temporal features**: Hour, day, month, cyclical encodings
- **Lag features**: Historical values (1h, 2h, 3h, 6h, 12h, 24h, 168h ago)
- **Rolling features**: Moving averages and standard deviations
- **Weather features**: Temperature, humidity, wind, pressure, precipitation
- **Metadata features**: Geographic location, station type/area

#### Critical Design Decision: Time-Based vs Row-Based Shifting

**IMPORTANT**: All lag, rolling, and target features use **time-based shifting** (not row-based) to correctly handle data gaps.

**Problem with Row-Based Shifting:**
```python
# WRONG - shifts by 1 row, which may be >1 hour if there are gaps
df['PM2.5_lag_1h'] = df.groupby(['Country', 'SiteNumber'])['PM2.5'].shift(1)
```

If there's a 3-hour gap in the data, `shift(1)` would give you a value from 3 hours ago, not 1 hour ago.

**Solution - Time-Based Shifting:**
```python
# CORRECT - shifts by exactly 1 hour in real time
# 1. Create complete hourly grid
full_range = pd.date_range(start, end, freq='h')
# 2. Reindex to hourly (fills gaps with NaN)
pm25_hourly = df['PM2.5'].reindex(full_range)
# 3. Shift on complete grid (now 1 row = 1 hour)
pm25_lag_1h = pm25_hourly.shift(1)
# 4. Reindex back to original timestamps
pm25_lag_1h = pm25_lag_1h.reindex(original_timestamps)
```

This ensures that `lag_1h` always refers to exactly 1 hour ago in real time, regardless of data gaps.

#### Process Steps

##### Step 1: Replace Sentinel Values

Air quality measurements cannot be negative. Replace sentinel values with NaN:
- Values < 0 (includes -999, -9999, -5499)

```python
df.loc[df['PM2.5'] < 0, 'PM2.5'] = np.nan
df.loc[df['NO2'] < 0, 'NO2'] = np.nan
df.loc[df['PM10'] < 0, 'PM10'] = np.nan
```

##### Step 2: Filter Missing Target Variable

Remove rows where PM2.5 (target variable) is missing:

```python
df = df[df['PM2.5'].notna()]
```

##### Step 3: Impute Missing Feature Values

Impute missing NO2 and PM10 using **time-based interpolation** within each site:

1. Sort by `Country`, `SiteNumber`, `Start`
2. Interpolate linearly within each site group
3. Fill remaining edge NaNs with site mean

```python
df['NO2'] = df.groupby(['Country', 'SiteNumber'])['NO2'].transform(
    lambda x: x.interpolate(method='linear', limit_direction='both')
)
df['NO2'] = df.groupby(['Country', 'SiteNumber'])['NO2'].transform(
    lambda x: x.fillna(x.mean())
)
```

##### Step 4: Add Timezone-Aware Datetime Columns

**Rationale**: Air pollution patterns follow local time (rush hour, weekday/weekend behavior), not UTC time. Different European cities have different timezones and DST transitions.

**Process:**

1. **Convert Start to UTC**
   - Original `Start` column is in UTC+1 (CET without DST)
   - Convert to UTC by subtracting 1 hour
   - Create `dt_utc` column

2. **Convert UTC to Local Timezone**
   - Map each country to its IANA timezone:
     - Austria: `Europe/Vienna` (CET/CEST, UTC+1/+2)
     - France: `Europe/Paris` (CET/CEST, UTC+1/+2)
     - Spain: `Europe/Madrid` (CET/CEST, UTC+1/+2)
     - Belgium: `Europe/Brussels` (CET/CEST, UTC+1/+2)
     - Finland: `Europe/Helsinki` (EET/EEST, UTC+2/+3)
   - Convert `dt_utc` to local timezone for each row
   - Create `dt_local` column (timezone-aware)

**Why This Matters:**

Example: Rush hour in Helsinki (8 AM local) occurs at different UTC times depending on season:
- Winter (EET): 8 AM = 6 AM UTC
- Summer (EEST): 8 AM = 5 AM UTC

By using local time for temporal features, we ensure the model learns that "8 AM" has high pollution regardless of the UTC hour.

##### Step 5: Fetch and Merge Weather Data

Weather data is obtained from the **Open-Meteo Archive API** for all unique site locations.

**API Details:**
- Endpoint: `https://archive-api.open-meteo.com/v1/archive`
- Coverage: 2018-01-01 to 2024-12-31
- Temporal resolution: Hourly
- Spatial resolution: ~4 decimal places (≈11 meters)

**Weather Variables:**

| Variable | Description | Units |
|----------|-------------|-------|
| `temperature_2m` | Temperature at 2 meters | °C |
| `relative_humidity_2m` | Relative humidity at 2 meters | % |
| `dew_point_2m` | Dew point temperature | °C |
| `wind_speed_10m` | Wind speed at 10 meters | m/s |
| `wind_direction_10m` | Wind direction | degrees |
| `precipitation` | Total precipitation | mm |
| `surface_pressure` | Surface pressure | hPa |

**Wind Component Transformation:**

Wind speed and direction are converted to **u/v components** for better ML performance:
- `wind_u = wind_speed * cos(wind_direction)` (east-west component)
- `wind_v = wind_speed * sin(wind_direction)` (north-south component)

This avoids the circular discontinuity problem (0° = 360°).

**Caching Strategy:**

To avoid repeated API calls, responses are cached in `weather_cache.db` (SQLite):
- First run: Fetches from API (adds 5-second delay between requests)
- Subsequent runs: Loads from cache (instant)

**Merge Strategy:**

1. Round site coordinates to 4 decimal places (API grid alignment)
2. Fetch weather for each unique rounded location
3. Merge with air quality data on `Lat_rounded`, `Lon_rounded`, `dt_utc`

##### Step 6: Impute Missing Weather Features

Weather data may have gaps. Impute using the same strategy as NO2/PM10:
1. Interpolate linearly within each site
2. Fill remaining NaNs with site mean

##### Step 7: Create Temporal Features

Extract temporal features from **local time** (`dt_local`):

| Feature | Description | Type |
|---------|-------------|------|
| `hour` | Hour of day (0-23) | Integer |
| `day_of_week` | Day of week (0=Monday, 6=Sunday) | Integer |
| `day_of_month` | Day of month (1-31) | Integer |
| `month` | Month (1-12) | Integer |
| `year` | Year (2018-2024) | Integer |
| `is_weekend` | Weekend flag (Sat/Sun) | Binary |
| `season` | Season (Winter/Spring/Summer/Fall) | String |
| `hour_sin` | Cyclical encoding: sin(2π × hour/24) | Float |
| `hour_cos` | Cyclical encoding: cos(2π × hour/24) | Float |
| `month_sin` | Cyclical encoding: sin(2π × month/12) | Float |
| `month_cos` | Cyclical encoding: cos(2π × month/12) | Float |

**Cyclical Encoding Rationale:**

For cyclical features (hour, month), using sin/cos encoding prevents discontinuities:
- Hour 23 and Hour 0 are close in time but far apart numerically
- `sin(hour)` and `cos(hour)` create a continuous circular representation

##### Step 8: Create Lag Features (Time-Based)

Create historical values for PM2.5, NO2, and PM10 at specific lags.

**Lag Periods:**
- `1h`: Previous hour
- `2h`: 2 hours ago
- `3h`: 3 hours ago
- `6h`: 6 hours ago
- `12h`: 12 hours ago
- `24h`: Same hour yesterday
- `168h`: Same hour last week

**Algorithm:**

For each pollutant and each site:
1. Create complete hourly grid
2. Reindex pollutant to hourly grid
3. Shift by lag period: `lag_Nh = hourly.shift(N)`
4. Reindex back to original timestamps
5. Assign to `{POLLUTANT}_lag_{N}h` column

**Result**: 3 pollutants × 7 lags = **21 lag features**

**Note**: The target variable (y) for prediction should be created in your model training scripts. This allows flexibility in defining different prediction horizons or multi-step forecasting as needed.

##### Step 9: Create Rolling Features (Time-Based)

Create moving averages and standard deviations over time windows.

**Window Sizes:**
- `3h`: Past 3 hours
- `6h`: Past 6 hours
- `12h`: Past 12 hours
- `24h`: Past 24 hours

**Features:**
- **Rolling mean**: Average value over window (captures trend)
- **Rolling std**: Standard deviation over window (captures volatility)

**Algorithm:**

For each pollutant, window size, and site:
1. Create complete hourly grid
2. Reindex pollutant to hourly grid
3. Apply rolling window: `rolling_mean = hourly.rolling(window=N, min_periods=1).mean()`
4. Reindex back to original timestamps
5. Assign to `{POLLUTANT}_rolling_mean_{N}h` column

**Result**: 3 pollutants × 4 windows × 2 metrics = **24 rolling features**

**Note**: `min_periods=1` allows computation even with incomplete windows (e.g., at the start of data).

##### Step 10: Remove Incomplete Rows

Remove rows where any lag feature is NaN. This typically occurs:
- At the start of each site's data (no historical values for lags)
- After large data gaps (e.g., sensor outage for >168 hours)

This ensures the final dataset has complete features for all rows.

##### Step 11: Save ML Dataset

Save the final dataset and metadata:

**Output Files:**
- `ml_ready_dataset_v6.csv`: Main dataset
- `ml_features.txt`: List of all features grouped by category
- `ml_dataset_summary.txt`: Dataset statistics (rows, columns, date range, missing values)

#### Feature Summary

The final dataset contains the following feature groups:

| Category | Features | Count |
|----------|----------|-------|
| **Identifiers** | Start, Country, SiteNumber, dt_utc, dt_local | 5 |
| **Geographic Metadata** | Latitude, Longitude, Altitude | 3 |
| **Station Metadata** | StationType_*, StationArea_* (one-hot) | Variable |
| **Target** | PM2.5 | 1 |
| **Original Pollutants** | NO2, PM10 | 2 |
| **Weather** | temperature_2m, relative_humidity_2m, dew_point_2m, wind_u, wind_v, precipitation, surface_pressure | 7 |
| **Temporal** | hour, day_of_week, month, year, is_weekend, season, hour_sin, hour_cos, month_sin, month_cos | 11 |
| **Lag Features** | PM2.5/NO2/PM10 × 7 lags | 21 |
| **Rolling Mean** | PM2.5/NO2/PM10 × 4 windows | 12 |
| **Rolling Std** | PM2.5/NO2/PM10 × 4 windows | 12 |
| **TOTAL** | | **75+** |

**Key Feature Insights** (based on Beijing AQI analysis):
- `PM2.5_rolling_mean_3h` contributes 87-90% of predictive power
- Short-interval lags (1h, 2h, 3h) capture recent trends
- Rolling std features capture volatility/stability

#### Execution

```bash
python dataset_build/src/prepare_ml_dataset.py
```

**Prerequisites**: Must run `dataset_build/src/process_data.py` first. Requires internet connection for weather API.

**Runtime**:
- First run: ~30-60 minutes (fetching weather data with 5s delays)
- Subsequent runs: ~5-10 minutes (weather data cached)

---

### Step 5: Data Quality Filtering (Station Selection)

After creating the initial ML dataset, additional filtering steps ensure data quality by selecting only stations with sufficient completeness and manageable data gaps.

#### Substep 5.1: Compute Station Coverage Statistics

**Script:** `dataset_build/src/coverage_only_v6.py`

This script analyzes each station's data completeness in both train and test periods.

**Metrics Computed:**
- **Train Completeness**: Percentage of expected hourly measurements present in training period (2018-01-08 to 2022-12-31)
- **Test Completeness**: Percentage of expected hourly measurements present in test period (2023-01-01 to 2024-12-31)
- **Max Gap Hours**: Largest time gap between consecutive measurements (used to identify prolonged sensor outages)

**Output:**
```
dataset_build/
└── station_train_test_coverage.csv
```

Contains: `Country`, `SiteNumber`, `train_completeness`, `test_completeness`, `train_max_gap_hours`, `test_max_gap_hours`, etc.

**Execution:**
```bash
python dataset_build/src/coverage_only_v6.py
```

**Prerequisites**: Must run `dataset_build/src/prepare_ml_dataset.py` first.

#### Substep 5.2: Create Final Filtered Dataset

After computing coverage statistics, choose **one** of the following filtering strategies to create the final dataset:

##### Option 1: Full Realistic Dataset (Moderate Filtering)

**Script:** `dataset_build/src/dataset_full_realistic_v6.py`

Creates a dataset with moderate quality requirements, suitable for realistic forecasting scenarios.

**Filtering Criteria:**
- Minimum train completeness: **50%**
- Minimum test completeness: **50%**
- Maximum gap between measurements: **168 hours** (1 week)

**Output Files:**
```
ml_ready_dataset_full_realistic.csv              # Final filtered dataset
dataset_build/stations_full_realistic.csv        # Selected stations list
dataset_build/ml_dataset_realistic_summary.txt   # Dataset statistics
```

**Execution:**
```bash
python dataset_build/src/dataset_full_realistic_v6.py
```

**Use Case**: Retains more stations for broader geographic coverage; tolerates occasional week-long sensor outages.

##### Option 2: Best Station Dataset (Strict Filtering)

**Script:** `dataset_build/src/dataset_best_station.py`

Creates a high-quality dataset with strict requirements, suitable for optimal model performance.

**Filtering Criteria:**
- Minimum train completeness: **80%**
- Minimum test completeness: **80%**
- Maximum gap between measurements: **72 hours** (3 days)

**Output Files:**
```
ml_ready_dataset_best_station.csv    # Final filtered dataset
dataset_build/best_station.csv       # Selected stations list
```

**Execution:**
```bash
python dataset_build/src/dataset_best_station.py
```

**Use Case**: Ensures highest data quality; stations with consistent, reliable measurements; suitable for model benchmarking.

**Prerequisites**: Must run `dataset_build/src/coverage_only_v6.py` first.

---

## Model Training

After creating the final filtered dataset, the pipeline trains 10 forecasting models to predict PM2.5 concentrations at multiple time horizons.

### Training Architecture: Global Models

All models use a **global architecture**: a single model is trained on all cities/sites together, with station encoding (Country + SiteNumber) to capture site-specific patterns. This approach enables:
- **Cross-site learning**: The model learns patterns from all locations simultaneously
- **Better generalization**: Knowledge from high-data stations helps improve predictions for lower-data stations
- **Computational efficiency**: One model for all sites vs hundreds of separate per-site models

### Multi-Horizon Forecasting

Each model trains **5 separate predictors** for different forecasting horizons:

| Horizon | Description | Use Case |
|---------|-------------|----------|
| **1h** | Next hour | Immediate short-term forecasting |
| **3h** | 3 hours ahead | Short-term planning |
| **6h** | 6 hours ahead | Medium-term alerts |
| **12h** | 12 hours ahead | Half-day forecasting |
| **24h** | 24 hours ahead | Long-term daily planning |

Target variables are created using **time-based shifting** (not row-based) to correctly handle data gaps. For each station, a complete hourly time grid is created, targets are shifted by the exact number of hours, and then reindexed back to available timestamps.

### Available Models

#### Baseline Models

**Persistence** (`train_models/global/persistence.py`)
- Predicts that PM2.5 will remain at its current value
- Benchmark for all other models
- Runtime: ~2 minutes

**Linear Regression** (`train_models/global/lr_global.py`)
- Simple linear model with all features
- No hyperparameter tuning
- Useful baseline for understanding feature relationships
- Runtime: ~2 minutes

#### Statistical and Tree-Based Models

All tree-based models use **hyperparameter optimization** with RandomizedSearchCV (5-fold cross-validation):

**Generalized Additive Model (GAM)** (`train_models/global/gam_global.py`)
- Non-linear additive model with spline smoothing
- Tuned parameters: number of splines, spline order, regularization strength
- Best params cached to `train_models/global/gam_best_params_by_horizon.json`
- Runtime: ~98 minutes (~1.6 hours)

**Random Forest** (`train_models/global/rf_global.py`)
- Ensemble of decision trees
- Tuned parameters: n_estimators, max_depth, min_samples_split, min_samples_leaf
- Best params cached to `train_models/global/rf_best_params_by_horizon.json`
- Runtime: ~36 minutes

**XGBoost** (`train_models/global/xgb_global.py`)
- Gradient boosting with tree learners
- Tuned parameters: max_depth, learning_rate, n_estimators, subsample, colsample_bytree
- Early stopping on validation set (within train period)
- Best params cached to `train_models/global/xgb_best_params_by_horizon.json`
- Runtime: ~3 minutes (with cached hyperparameters from prior tuning)
- Note: Initial hyperparameter tuning adds significant time if cache doesn't exist

**LightGBM** (`train_models/global/lgb_global.py`)
- Efficient gradient boosting with histogram-based splitting
- Tuned parameters: num_leaves, learning_rate, n_estimators, subsample, colsample_bytree
- Early stopping on validation set (within train period)
- Best params cached to `train_models/global/lgb_best_params_by_horizon.json`
- Runtime: ~52 minutes

**Note on HPO caching**: If the `*_best_params_by_horizon.json` file exists, the script will load cached parameters and skip expensive hyperparameter search. To force re-tuning, delete the JSON file.

#### Deep Learning Models

All LSTM models require **continuous sequences** of length 168 hours (1 week lookback). During training:
1. Data is segmented at gaps larger than 2 hours
2. Only segments with at least 194 hours (168 + 24 + 2) are kept
3. Features are normalized per-station using StandardScaler
4. Early stopping on validation set (2022-07 to 2022-12)

**LSTM-Global** (`train_models/global/lstm_global.py`)
- Basic LSTM architecture with 2 layers
- 168-hour lookback window
- Runtime: ~13 hours (actual training time varies by early stopping)

**LSTM-Residual** (`train_models/global/lstm_global_residual.py`)
- LSTM with residual connections (skip connections)
- Helps with gradient flow in deep networks
- 168-hour lookback window
- Runtime: **~46 hours** (longest training time due to deeper architecture)

**LSTM-Attention** (`train_models/global/lstm_global_attention.py`)
- LSTM with global attention mechanism
- Learns to focus on important time steps in the lookback window
- 168-hour lookback window
- Runtime: ~34 hours

**LSTM-CNN** (`train_models/global/lstm_cnn.py`)
- Hybrid architecture combining LSTM and 1D CNN layers
- CNN extracts local patterns, LSTM models temporal dependencies
- 168-hour lookback window
- Runtime: ~43 hours

### Feature Configuration

All models use the same comprehensive feature set defined in `train_models/feature_config.py`:

| Feature Group | Features | Count |
|---------------|----------|-------|
| **PM2.5 History** | lag_1h, lag_2h, lag_3h, lag_6h, lag_12h, lag_24h, lag_168h, rolling_mean_3h/6h/12h/24h, rolling_std_3h/6h/12h/24h | 13 |
| **NO2 Features** | current value, 7 lags, 4 rolling_mean, 4 rolling_std | 16 |
| **PM10 Features** | current value, 7 lags, 4 rolling_mean, 4 rolling_std | 16 |
| **Weather** | temperature_2m, relative_humidity_2m, dew_point_2m, wind_u_component, wind_v_component, precipitation, surface_pressure | 7 |
| **Temporal** | hour_sin, hour_cos, month_sin, month_cos, day_of_week, is_weekend, day_of_month, year, season | 11 |
| **Station Encoding** | Country, SiteNumber (encoded) | 2 |
| **Metadata** | Latitude, Longitude, Altitude, StationType_*, StationArea_* | 10 |
| **TOTAL** | | **75+** |

**Key insight from Beijing AQI analysis**: PM2.5_rolling_mean_3h contributes 87-90% of predictive power. Other features capture additional patterns and improve robustness.

### Training Output Structure

Each training script produces standardized outputs to ensure consistency:

```
models/{model_name}_models/          # Trained models (.pkl or .keras)
├── {model}_h1.{pkl|keras}           # Model for horizon 1
├── {model}_h3.{pkl|keras}           # Model for horizon 3
├── {model}_h6.{pkl|keras}           # Model for horizon 6
├── {model}_h12.{pkl|keras}          # Model for horizon 12
└── {model}_h24.{pkl|keras}          # Model for horizon 24

results/
├── {model}_metrics*.csv             # Performance metrics (Scope, Group, horizon, MAE, RMSE, R2, Bias, Count)
└── {model}_predictions_sample.csv   # Sample predictions (10-20K rows for visualization)

train_models/global/runs_info/
└── {model}_run_info.json            # Training metadata (timing, sample sizes, hyperparams)

train_models/global/
└── {model}_best_params_by_horizon.json  # Cached HPO results (RF, XGB, LGB, GAM only)

train_models/scalers/                # LSTM only
└── {model}_scalers.json             # Per-station StandardScaler parameters

train_models/history/                # LSTM only
└── {model}_histories/
    ├── history_h1.csv               # Epoch-by-epoch training metrics
    ├── history_h3.csv
    ├── history_h6.csv
    ├── history_h12.csv
    └── history_h24.csv
```

### Training Individual Models

```bash
# Baseline models
python train_models/global/persistence.py
python train_models/global/lr_global.py

# Statistical and tree-based models
python train_models/global/gam_global.py
python train_models/global/rf_global.py
python train_models/global/xgb_global.py
python train_models/global/lgb_global.py

# Deep learning models
python train_models/global/lstm_global.py
python train_models/global/lstm_global_residual.py
python train_models/global/lstm_global_attention.py
python train_models/global/lstm_cnn.py
```

### Training All Models

**WARNING**: Training all models takes **~140 hours** (~6 days) due to LSTM training time.

```bash
# Sequential training (all models: ~140 hours)
# Baseline models (~4 minutes)
python train_models/global/persistence.py
python train_models/global/lr_global.py

# Statistical and tree-based models (~3 hours)
python train_models/global/gam_global.py
python train_models/global/rf_global.py
python train_models/global/xgb_global.py
python train_models/global/lgb_global.py

# Deep learning models (~136 hours)
python train_models/global/lstm_global.py            # 13 hours
python train_models/global/lstm_global_residual.py  # 46 hours
python train_models/global/lstm_global_attention.py # 34 hours
python train_models/global/lstm_cnn.py              # 43 hours
```

Or train classical models in parallel (requires sufficient CPU/memory):
```bash
# Train classical models in parallel (~2 hours with parallel execution)
python train_models/global/persistence.py &
python train_models/global/lr_global.py &
python train_models/global/gam_global.py &
python train_models/global/rf_global.py &
python train_models/global/xgb_global.py &
python train_models/global/lgb_global.py &
wait

# Train LSTM models sequentially (GPU memory limitations prevent parallel execution)
python train_models/global/lstm_global.py
python train_models/global/lstm_global_residual.py
python train_models/global/lstm_global_attention.py
python train_models/global/lstm_cnn.py
```

---

## Evaluation Protocols

The project uses two evaluation protocols to assess model performance fairly and comprehensively.

### Protocol A: Full-Coverage Evaluation

**Purpose**: Evaluate classical and ML models on maximum geographic coverage

**Dataset**: All stations passing completeness filters
- Minimum train completeness: **50%**
- Minimum test completeness: **50%**
- Maximum gap: **168 hours** (1 week)
- Dataset file: `ml_ready_dataset_full_realistic.csv`

**Models evaluated**:
- Baseline: Persistence, Linear Regression
- Statistical/ML: GAM, Random Forest, XGBoost, LightGBM

**Test sample size**: ~606,635 hourly observations (h=1), varying by horizon

**Metrics files**: `results/{model}_metrics.csv` (no "_dl_" suffix)

**Rationale**: Classical models don't require continuous sequences, so we can use all available data for maximum geographic coverage and robustness testing.

### Protocol B: Sequence-Eligible Subset

**Purpose**: Fair comparison between classical and deep learning models on identical data

**Dataset**: Subset with continuous sequences ≥194 hours
- Required for LSTM models (168h lookback + 24h max horizon + 2h buffer)
- Data breaks at gaps >2 hours
- Segments shorter than 194 hours are excluded
- ~38% reduction in test samples vs Protocol A

**Models evaluated**:
- All Protocol A models re-evaluated on sequence-eligible subset
- Deep learning: LSTM-Global, LSTM-Residual, LSTM-Attention, LSTM-CNN

**Test sample size**: ~375,906 hourly observations (h=1), varying by horizon

**Metrics files**:
- Classical/ML: `results/{model}_dl_metrics.csv` (with "_dl_" suffix)
- Deep learning: `results/lstm_{variant}_metrics.csv`

**Rationale**: Deep learning models require continuous time series without large gaps. Re-evaluating classical models on the same subset enables fair comparison by eliminating dataset bias.

### Metrics File Structure

All metrics CSV files share a standardized format:

```csv
Scope,Group,horizon,MAE,RMSE,R2,Bias,Count
Global,All,1,4.52,6.89,0.67,0.12,606635
Global,All,3,5.01,7.42,0.62,0.18,601349
Country,AT,1,4.21,6.45,0.71,0.08,45230
Country,FR,1,4.89,7.21,0.65,0.14,120450
StationArea,urban,1,4.89,7.21,0.65,0.14,200450
StationArea,suburban,1,4.21,6.45,0.71,0.08,180230
```

**Columns**:
- `Scope`: Aggregation level (`Global`, `Country`, `StationArea`)
- `Group`: Grouping value (`All`, country code like `AT`, or area type like `urban`)
- `horizon`: Prediction horizon (1, 3, 6, 12, or 24 hours)
- `MAE`: Mean Absolute Error (μg/m³)
- `RMSE`: Root Mean Squared Error (μg/m³)
- `R2`: R-squared score (coefficient of determination)
- `Bias`: Mean prediction bias (y_pred - y_true, μg/m³)
- `Count`: Number of test samples

**Scopes**:
- **Global**: Aggregated across all stations (provides overall performance)
- **Country**: Per-country breakdown (AT, BE, ES, FI, FR) - tests geographic robustness
- **StationArea**: Per-area breakdown (urban, suburban, rural) - tests performance by pollution level

### Train/Test Split

All models use a **temporal split** (not random) to simulate real-world forecasting:

- **Train**: 2018-01-08 to 2022-12-31 (~5 years)
- **Test**: 2023-01-01 to 2024-12-31 (~2 years)

**Note**: LSTM models use an additional validation split within the training period (2022-07 to 2022-12) for early stopping.

---

## Statistical Analysis and Interpretability

Three advanced analysis tools are provided in the `tests/` directory for model evaluation beyond standard metrics.

### 1. Diebold-Mariano Test

**Script**: `tests/diebold_mariano_tests.py`

**Purpose**: Statistical significance testing of performance differences between model pairs

**Method**: Diebold-Mariano test with Newey-West HAC variance estimator to account for autocorrelation in time series forecasting errors

**Comparisons**:
- **Protocol A**: XGBoost vs LightGBM (horizons 1, 12, 24) - Which tree-based model is significantly better?
- **Protocol B**: ML models vs Deep Learning models (all 5 horizons) - Is the complexity of LSTM warranted?

**Sample size**: 1,000 predictions per horizon (statistically sufficient for DM test)

**Output**:
- `results/diebold_mariano_results.csv` - DM statistics, p-values, and significance flags
- `figures/diebold_mariano_comparison.png` - Visualization of test results

**Interpretation**:
- p-value < 0.05: Performance difference is statistically significant
- p-value ≥ 0.05: No significant difference (simpler model may be preferred)

**Runtime**: 10-30 minutes

**Execution**:
```bash
python tests/diebold_mariano_tests.py
```

### 2. SHAP Feature Importance

**Script**: `tests/shap_feature_importance.py`

**Purpose**: Explain feature contributions using SHAP (SHapley Additive exPlanations) values

**Analyzed model**: Best tree-based model (XGBoost or LightGBM based on results)

**Horizons**: h=1 (short-term) and h=12 (medium-term) to compare feature importance across time scales

**Sample size**: 10,000 test samples for robust SHAP value estimation

**Feature grouping**: Features are grouped into 5 categories for interpretability:

1. **PM2.5 History** (13 features): lag_1h through lag_168h, rolling_mean_3h/6h/12h/24h, rolling_std_3h/6h/12h/24h
2. **Other Pollutants** (20 features): NO2 and PM10 current values, lags, and rolling statistics
3. **Meteorological** (7 features): temperature_2m, relative_humidity_2m, dew_point_2m, wind_u_component, wind_v_component, precipitation, surface_pressure
4. **Temporal** (11 features): hour_sin/cos, month_sin/cos, day_of_week, is_weekend, day_of_month, year, season
5. **Metadata** (10 features): Latitude, Longitude, Altitude, Country, SiteNumber, StationType_*, StationArea_*

**Output**:
- `results/shap_importance_h1.csv` - Individual feature importance (h=1)
- `results/shap_importance_h12.csv` - Individual feature importance (h=12)
- `results/shap_grouped_importance_h1.csv` - Grouped feature importance (h=1)
- `results/shap_grouped_importance_h12.csv` - Grouped feature importance (h=12)
- `figures/shap_grouped_importance.png` - Bar plots showing group contributions

**Key insight**: PM2.5 History group typically contributes 80-90% of predictive power, validating the importance of temporal patterns in air quality forecasting.

**Runtime**: 5-15 minutes

**Execution**:
```bash
python tests/shap_feature_importance.py
```

### 3. Occlusion Test

**Script**: `tests/occlusion_test_dl.py`

**Purpose**: Measure importance of LSTM's 168-hour lookback window through ablation study

**Analyzed model**: Best deep learning model (LSTM-Attention or LSTM-Residual based on results)

**Method**: Compare two prediction modes:
- **Baseline**: Full 168-hour historical sequence (normal operation)
- **Occluded**: Zero out entire lookback window (only current features available)

**Horizons**: All 5 horizons (1, 3, 6, 12, 24) to analyze how lookback importance changes with prediction distance

**Sample size**: 1,000 test sequences

**Output**:
- `results/occlusion_test_results.csv` - Baseline vs occluded metrics with degradation percentages
- `figures/occlusion_test_comparison.png` - Performance degradation plots across horizons

**Interpretation**:
- Large degradation (>20% MAE increase): Lookback window is critical for accurate predictions
- Small degradation (<10% MAE increase): Model relies more on current features than historical patterns

**Expected finding**: Degradation should increase with longer horizons as recent patterns become more important for distant predictions.

**Runtime**: 5-15 minutes

**Execution**:
```bash
python tests/occlusion_test_dl.py
```

---

## Publication-Ready Outputs

Two scripts in `scripts/` generate publication-ready tables and figures from model evaluation results.

### Model Skill Computation

**Script**: `scripts/compute_model_skill.py`

**Purpose**: Calculate skill scores comparing each model's performance to the persistence baseline

**Formula**:
- `MAE_skill = model_MAE - persistence_MAE` (negative is better)
- `RMSE_skill = model_RMSE - persistence_RMSE` (negative is better)

Skill score interpretation:
- Negative value: Model outperforms persistence (desirable)
- Zero: Model performs same as persistence
- Positive value: Model underperforms persistence (undesirable)

**Two baselines**:
- `results/persistence_metrics.csv` for Protocol A models
- `results/persistence_dl_metrics.csv` for Protocol B models

**Process**:
1. Loads all `*_metrics*.csv` files from `results/` directory
2. Matches each model's metrics with appropriate persistence baseline
3. Computes skill scores: `skill = model_metric - persistence_metric`
4. Saves augmented metrics to `results_with_skill/` directory

**Output**: `results_with_skill/` directory
- Creates augmented versions of all metrics files
- Adds `MAE_skill` and `RMSE_skill` columns
- Preserves original metrics (MAE, RMSE, R², Bias, Count)
- File structure mirrors `results/` directory

**Runtime**: <1 minute

**Execution**:
```bash
python scripts/compute_model_skill.py
```

### Publication Tables and Figures

**Script**: `scripts/generate_publication_outputs.py`

**Purpose**: Generate publication-ready tables and figures from evaluation metrics with skill scores

**Output Structure**:

#### Tables (3 tables, 6 files in `tables/` directory):

**1. table1_dataset_overview.csv/tex**
- Geographic coverage: 5 countries (AT, BE, ES, FI, FR), station distribution by country
- Temporal coverage: train period (2018-2022), test period (2023-2024)
- Station counts by area type (urban, suburban, rural)
- Test sample counts by horizon for both protocols
- Dataset statistics and train/test split information

**2. table2_protocol_a_performance.csv/tex**
- **Models**: Persistence, Linear Regression, GAM, Random Forest, XGBoost, LightGBM
- **Metrics**: MAE, RMSE, R², Bias (averaged across all stations)
- **Horizons**: 1h, 3h, 6h, 12h, 24h
- **Format**: Publication-ready with proper number formatting (2-3 decimal places)
- **Highlight**: Best performing model per metric per horizon

**3. table3_protocol_b_performance.csv/tex**
- **Models**: All Protocol A models + LSTM-Global, LSTM-Residual, LSTM-Attention, LSTM-CNN
- **Metrics**: MAE, RMSE, R², Bias (averaged across all stations)
- **Horizons**: 1h, 3h, 6h, 12h, 24h
- **Format**: Publication-ready with proper number formatting
- **Highlight**: Comparison between classical and deep learning approaches

**LaTeX format**: Tables are exported in both CSV and LaTeX (.tex) formats with:
- `\toprule`, `\midrule`, `\bottomrule` for professional appearance
- Bold formatting for best values
- Proper column alignment
- Caption and label ready for insertion into papers

#### Figures (5+ high-res PNG in `figures/` directory):

**1. fig1_spatial_coverage.png**
- Map showing station locations across 5 countries
- Color-coded by country or station area type
- Highlights geographic distribution and coverage

**2. fig2_performance_vs_horizon.png**
- 4-panel comparison plot (2x2 grid)
  - Top-left: Protocol A RMSE vs horizon
  - Top-right: Protocol A MAE vs horizon
  - Bottom-left: Protocol B RMSE vs horizon
  - Bottom-right: Protocol B MAE vs horizon
- Line plot for each model showing performance degradation with longer horizons
- Useful for understanding how each model handles different prediction distances

**3. fig3_country_robustness.png**
- Heatmap showing best RMSE per country × horizon
- Rows: 5 countries (AT, BE, ES, FI, FR)
- Columns: 5 horizons (1, 3, 6, 12, 24)
- Color intensity: RMSE magnitude (lighter = better)
- Identifies which countries/horizons are hardest to predict

**4. fig4_area_robustness.png**
- Heatmap showing best RMSE per station area × horizon
- Rows: 3 area types (urban, suburban, rural)
- Columns: 5 horizons (1, 3, 6, 12, 24)
- Color intensity: RMSE magnitude (lighter = better)
- Reveals whether urban/suburban/rural areas have different predictability

**5. fig5_bias_analysis.png**
- Line plot of prediction bias vs horizon for all models
- Separate panels for Protocol A and Protocol B
- Zero line indicates perfect calibration
- Positive bias: systematic overprediction
- Negative bias: systematic underprediction

**6. fig6_skill_comparison.png** (optional)
- Bar plot comparing model skill scores (MAE_skill or RMSE_skill)
- Separate panels per horizon or grouped bar chart
- Baseline (persistence) at zero line
- Negative bars: model outperforms persistence

**Additional figures** (from statistical analysis):
- `shap_grouped_importance.png` - Feature importance analysis
- `diebold_mariano_comparison.png` - Statistical significance tests
- `occlusion_test_comparison.png` - LSTM interpretability

**Configuration**:
- **DPI**: 300 (publication quality)
- **Format**: PNG (easily convertible to PDF/EPS for journals)
- **Color schemes**: Colorblind-friendly palettes (viridis, colorblind10)
- **Font sizes**: Optimized for 2-column journal format (10-12pt base)
- **Legend placement**: Automatic best location to avoid overlap

**Runtime**: 5-15 minutes

**Execution**:
```bash
# Generate all tables and figures
python scripts/generate_publication_outputs.py

# Generate only tables (faster)
python scripts/generate_publication_outputs.py --tables-only

# Generate only figures
python scripts/generate_publication_outputs.py --figures-only

# Custom output directories
python scripts/generate_publication_outputs.py --tables-dir my_tables --figures-dir my_figures

# Validation mode (check data consistency without generating outputs)
python scripts/generate_publication_outputs.py --validate

# Verbose output (show progress and diagnostics)
python scripts/generate_publication_outputs.py --verbose
```

---

## Output Structure

After running the complete pipeline, the following directory structure is created:

### Data Files
```
data/
├── PM2.5.parquet                      # Raw PM2.5 data (Step 1) ~1-2 GB
├── NO2.parquet                        # Raw NO2 data (Step 1) ~1-2 GB
├── PM10.parquet                       # Raw PM10 data (Step 1) ~1-2 GB
├── PM2.5_filtered.parquet             # PM2.5 from sites with data (Step 2)
├── NO2_filtered.parquet               # NO2 from PM2.5 sites only (Step 2)
├── PM10_filtered.parquet              # PM10 from PM2.5 sites only (Step 2)
└── pm25_sites.txt                     # List of all PM2.5 monitoring sites (Step 2)
```

### Processed Datasets
```
dataset_build/
├── processed_air_quality_data.csv     # Merged pollutants with metadata (Step 3) ~500 MB
├── station_train_test_coverage.csv    # Station quality metrics (Step 5) ~50 KB
├── stations_full_realistic.csv        # Selected stations list (Step 6) ~10 KB
└── ml_dataset_realistic_summary.txt   # Dataset statistics (Step 6)

ml_ready_dataset_v6.csv                # Feature-engineered pre-filtering (Step 4) ~2 GB
ml_ready_dataset_full_realistic.csv    # Final filtered dataset (Step 6) ~1.6 GB
weather_cache.db                       # Weather API cache (Step 4) ~200 MB
```

### Model Artifacts
```
models/
├── persistence.pkl                    # Persistence baseline ~1 KB
├── lr_models/                         # Linear regression models ~5 MB total
│   ├── lr_h1.pkl
│   ├── lr_h3.pkl
│   ├── lr_h6.pkl
│   ├── lr_h12.pkl
│   └── lr_h24.pkl
├── gam_models/                        # GAM models ~50 MB total
├── rf_models/                         # Random forest models ~200 MB total
├── xgb_models/                        # XGBoost models ~100 MB total
├── lgb_models/                        # LightGBM models ~50 MB total
├── lstm_global_models/                # LSTM basic models ~200 MB total
├── lstm_residual_single_models/       # LSTM residual models ~200 MB total
├── lstm_attention_models/             # LSTM attention models ~300 MB total
└── lstm_cnn_models/                   # LSTM-CNN models ~300 MB total

train_models/
├── global/
│   ├── runs_info/                     # Training metadata (JSON)
│   │   ├── lr_run_info.json           # Timing, sample counts, hyperparams
│   │   ├── gam_run_info.json
│   │   ├── rf_run_info.json
│   │   ├── xgb_run_info.json
│   │   ├── lgb_run_info.json
│   │   ├── lstm_global_run_info.json
│   │   ├── lstm_residual_run_info.json
│   │   ├── lstm_attention_run_info.json
│   │   └── lstm_cnn_run_info.json
│   ├── gam_best_params_by_horizon.json  # Cached HPO results
│   ├── rf_best_params_by_horizon.json
│   ├── xgb_best_params_by_horizon.json
│   └── lgb_best_params_by_horizon.json
├── scalers/                           # Per-station StandardScalers (LSTM only)
│   ├── lstm_global_scalers.json
│   ├── lstm_residual_scalers.json
│   ├── lstm_attention_scalers.json
│   └── lstm_cnn_scalers.json
└── history/                           # Training history (LSTM only)
    ├── lstm_global_histories/
    │   ├── history_h1.csv             # Epoch-by-epoch metrics
    │   ├── history_h3.csv
    │   ├── history_h6.csv
    │   ├── history_h12.csv
    │   └── history_h24.csv
    ├── lstm_residual_histories/
    ├── lstm_attention_histories/
    └── lstm_cnn_histories/
```

### Evaluation Results
```
results/
├── persistence_metrics.csv            # Protocol A metrics ~50 KB
├── persistence_dl_metrics.csv         # Protocol B metrics ~50 KB
├── lr_metrics.csv                     # Protocol A ~50 KB
├── lr_dl_metrics.csv                  # Protocol B ~50 KB
├── gam_metrics_tuned.csv              # Protocol A (tuned) ~50 KB
├── gam_dl_metrics_tuned.csv           # Protocol B (tuned) ~50 KB
├── rf_metrics_tuned.csv               # Protocol A (tuned) ~50 KB
├── rf_dl_metrics_tuned.csv            # Protocol B (tuned) ~50 KB
├── xgb_metrics_tuned.csv              # Protocol A (tuned) ~50 KB
├── xgb_dl_metrics_tuned.csv           # Protocol B (tuned) ~50 KB
├── lgb_metrics_tuned.csv              # Protocol A (tuned) ~50 KB
├── lgb_dl_metrics_tuned.csv           # Protocol B (tuned) ~50 KB
├── lstm_global_metrics.csv            # Protocol B (DL) ~50 KB
├── lstm_residual_singleoutput_aligned_metrics.csv  # Protocol B (DL) ~50 KB
├── lstm_attention_metrics.csv         # Protocol B (DL) ~50 KB
├── lstm_cnn_metrics.csv               # Protocol B (DL) ~50 KB
├── {model}_predictions_sample.csv     # Sample predictions ~5-10 MB per model
├── diebold_mariano_results.csv        # Statistical tests ~10 KB
├── shap_importance_h1.csv             # Feature importance (h=1) ~5 KB
├── shap_importance_h12.csv            # Feature importance (h=12) ~5 KB
├── shap_grouped_importance_h1.csv     # Grouped feature importance (h=1) ~1 KB
├── shap_grouped_importance_h12.csv    # Grouped feature importance (h=12) ~1 KB
└── occlusion_test_results.csv         # LSTM lookback test ~5 KB

results_with_skill/
└── {model}_metrics*.csv               # Metrics with skill scores (all models) ~50 KB each
```

### Publication Outputs
```
tables/
├── table1_dataset_overview.csv        # Dataset statistics (CSV) ~2 KB
├── table1_dataset_overview.tex        # Dataset statistics (LaTeX) ~3 KB
├── table2_protocol_a_performance.csv  # Protocol A results (CSV) ~5 KB
├── table2_protocol_a_performance.tex  # Protocol A results (LaTeX) ~8 KB
├── table3_protocol_b_performance.csv  # Protocol B results (CSV) ~8 KB
└── table3_protocol_b_performance.tex  # Protocol B results (LaTeX) ~12 KB

figures/
├── fig1_spatial_coverage.png          # Station map (300 DPI) ~500 KB
├── fig2_performance_vs_horizon.png    # RMSE/MAE comparison (300 DPI) ~400 KB
├── fig3_country_robustness.png        # Country-wise heatmap (300 DPI) ~300 KB
├── fig4_area_robustness.png           # Area-wise heatmap (300 DPI) ~300 KB
├── fig5_bias_analysis.png             # Bias vs horizon (300 DPI) ~350 KB
├── fig6_skill_comparison.png          # Model skill (300 DPI) ~400 KB
├── shap_grouped_importance.png        # Feature importance (300 DPI) ~350 KB
├── diebold_mariano_comparison.png     # Statistical tests (300 DPI) ~300 KB
└── occlusion_test_comparison.png      # LSTM lookback (300 DPI) ~300 KB
```

### Total Disk Space Requirements
- **Raw data** (`data/`): ~5 GB
- **Processed datasets**: ~3 GB (including ml_ready_dataset_full_realistic.csv at 1.6 GB)
- **Models**: ~1.5 GB (LSTM models are largest)
- **Results**: ~500 MB
- **Figures/tables**: ~50 MB
- **Cache/metadata**: ~250 MB
- **Total**: ~10-12 GB

---

## Requirements

### Python Version
- Python 3.12 or higher

### Dependencies

Install all dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Core Packages:**
- `pandas >= 2.0.0` - Data manipulation
- `numpy >= 1.24.0` - Numerical operations
- `pyarrow >= 14.0.0` - Parquet file support
- `pyyaml >= 6.0` - Configuration parsing
- `pytz >= 2024.1` - Timezone handling
- `requests >= 2.31.0` - API requests
- `openmeteo-requests >= 1.1.0` - Open-Meteo API client
- `requests-cache >= 1.1.0` - API response caching
- `retry-requests >= 2.0.0` - Retry logic for API requests

**ML Packages (for training):**
- `scikit-learn >= 1.3.0`
- `xgboost >= 2.0.0`
- `tensorflow >= 2.15.0`
- `statsmodels >= 0.14.0`
- `matplotlib >= 3.8.0`
- `seaborn >= 0.13.0`

### External Data

- `dataset_build/site_metadata.csv`: EEA site metadata (coordinates, station type/area)
  - Obtained from EEA metadata downloads
  - Required for Step 3 (dataset_build/src/process_data.py)

### Disk Space Requirements

Ensure you have at least **30 GB free disk space** for:
- Raw data downloads: ~5 GB
- Processed datasets: ~3 GB
- Model artifacts: ~1.5 GB
- Results and cache: ~750 MB
- Recommended total: **30+ GB** (with headroom for intermediate files)

---

## Reproducibility

### Quick Start

For a complete guide to reproducing all paper results (data + models + analysis + publication outputs), see the [Quick Start](#quick-start-reproduce-paper-results) section at the top of this README.

### Dataset Creation Only

To reproduce just the dataset from scratch:

### 1. Configure Parameters

Edit `utils/config.yaml` to specify:
- Cities to include
- Date range
- Pollutants to download

### 2. Run Pipeline Sequentially

```bash
# Step 1: Download raw data (30-60 minutes per pollutant)
python dataset_build/src/download_pollutants.py

# Step 2: Filter to PM2.5 sites (1-2 minutes)
python dataset_build/src/filter_pm25_sites.py

# Step 3: Process and merge data (5-10 minutes)
python dataset_build/src/process_data.py

# Step 4: Prepare ML dataset (30-60 minutes first run, 5-10 minutes cached)
python dataset_build/src/prepare_ml_dataset.py

# Step 5.1: Compute station coverage statistics
python dataset_build/src/coverage_only_v6.py

# Step 5.2: Create final filtered dataset (choose one)
# Option 1: Moderate filtering (more stations, broader coverage)
python dataset_build/src/dataset_full_realistic_v6.py

# Option 2: Strict filtering (fewer stations, highest quality)
python dataset_build/src/dataset_best_station.py
```

### 3. Verify Output

Check that the following files exist:

```
data/
├── PM2.5.parquet
├── NO2.parquet
├── PM10.parquet
├── PM2.5_filtered.parquet
├── NO2_filtered.parquet
├── PM10_filtered.parquet
└── pm25_sites.txt

dataset_build/
├── processed_air_quality_data.csv
├── ml_features.txt
├── ml_dataset_summary.txt
└── station_train_test_coverage.csv

ml_ready_dataset_v6.csv

# After Step 5 (choose one):
# If using full_realistic:
ml_ready_dataset_full_realistic.csv
dataset_build/stations_full_realistic.csv
dataset_build/ml_dataset_realistic_summary.txt

# If using best_station:
ml_ready_dataset_best_station.csv
dataset_build/best_station.csv
```

### 4. Validate Dataset

```bash
# Check dataset statistics
cat dataset_build/ml_dataset_summary.txt

# Check feature list
cat dataset_build/ml_features.txt

# Quick inspection with Python
python -c "import pandas as pd; df = pd.read_csv('ml_ready_dataset_v6.csv'); print(df.info()); print(df.describe())"
```

---

## Dataset Characteristics

### Temporal Coverage
- **Start Date**: 2018-01-08 (after 168h lag warmup)
- **End Date**: 2024-12-31
- **Duration**: ~7 years
- **Temporal Resolution**: Hourly

### Spatial Coverage
- **Countries**: Austria (AT), France (FR), Spain (ES), Belgium (BE), Finland (FI)
- **Cities**: Wien, Paris, Madrid, Antwerpen, Helsinki
- **Sites**: Variable (all sites with PM2.5 measurements in these cities)

### Train/Test Split

All models use **temporal split** (not random) to simulate real-world forecasting:

```python
train = df[df['Start'] <= '2022-12-31']  # 2018-2022 (~5 years)
test = df[df['Start'] > '2022-12-31']    # 2023-2024 (~2 years)
```

**Note**: LSTM models use additional validation split within training period (2022-07 to 2022-12) for early stopping.

### Test Sample Sizes

Sample sizes vary between evaluation protocols:

| Protocol | Description | Test Samples (h=1) | Stations |
|----------|-------------|-------------------|----------|
| **Protocol A** | Full coverage (all stations passing 50%/50% completeness) | ~606,635 | Higher |
| **Protocol B** | Sequence-eligible subset (continuous 194h sequences) | ~375,906 | Lower |

Sample counts decrease for longer horizons due to edge effects at end of test period.

### Data Quality Notes

1. **Missing Values**: Rows with missing PM2.5 target are removed. Other features are imputed.
2. **Data Gaps**: Handled via time-based shifting (gaps filled with NaN in complete grid, then dropped if lag features incomplete).
3. **Negative Values**: All negative sentinel values replaced with NaN before imputation.
4. **Temporal Alignment**: All timestamps aligned to UTC in `dt_utc` column. Local time available in `dt_local` for reference.

---


**Data Sources:**
- Air Quality Data: European Environment Agency (EEA) - https://www.eea.europa.eu/
- Weather Data: Open-Meteo Archive API - https://open-meteo.com/
- Site Metadata: EEA AirBase metadata

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note:** The pre-trained models and datasets on Hugging Face are licensed under CC BY 4.0.

---


