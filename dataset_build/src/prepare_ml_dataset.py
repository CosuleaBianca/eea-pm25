"""
Prepare ML-ready dataset with feature engineering.

This script:
1. Loads processed air quality data
2. Replaces sentinel values with NaN
3. Filters rows with missing PM2.5 (target variable)
4. Imputes missing NO2 and PM10 using interpolation
5. Adds timezone-aware datetime columns (dt_utc, dt_local)
6. Fetches and merges weather data from Open-Meteo API (temperature, humidity, wind, pressure, precipitation)
7. Imputes missing weather features using interpolation
8. Creates temporal features from local time (hour, day, month, cyclical encodings, etc.)
9. Creates TIME-BASED lag features (1h, 2h, 3h, 6h, 12h, 24h, 168h)
10. Creates TIME-BASED rolling features (3h, 6h, 12h, 24h mean & std)
11. Removes rows with incomplete features
12. Saves ML-ready dataset

Note: All lag and rolling features use TIME-BASED shifting (not row-based)
to correctly handle data gaps. This ensures 1h lag truly means 1 hour ago in real time.
The target variable (y) should be created in the model training scripts as needed.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'utils'))

import pandas as pd
import numpy as np
from config_utils import PipelineConfig
import pytz


def load_processed_data(config: PipelineConfig = None) -> pd.DataFrame:
    """Load processed air quality data."""
    if config:
        processed_file = config.output_dir / config.processed_file
    else:
        processed_file = Path("processed_air_quality_data.csv")

    df = pd.read_csv(processed_file, parse_dates=['Start'])
    print(f"Loaded {len(df):,} rows from {processed_file}")
    return df


def get_country_timezone_map():
    """
    Map country codes to their IANA timezone names.

    Returns:
        dict: Country code -> timezone name mapping
    """
    return {
        'AT': 'Europe/Vienna',      # Austria (CET/CEST)
        'FR': 'Europe/Paris',        # France (CET/CEST)
        'ES': 'Europe/Madrid',       # Spain (CET/CEST)
        'BE': 'Europe/Brussels',     # Belgium (CET/CEST)
        'FI': 'Europe/Helsinki',     # Finland (EET/EEST)
        'DE': 'Europe/Berlin',       # Germany (CET/CEST)
        'IT': 'Europe/Rome',         # Italy (CET/CEST)
        'NL': 'Europe/Amsterdam',    # Netherlands (CET/CEST)
        'SE': 'Europe/Stockholm',    # Sweden (CET/CEST)
        'NO': 'Europe/Oslo',         # Norway (CET/CEST)
        'DK': 'Europe/Copenhagen',   # Denmark (CET/CEST)
        'PL': 'Europe/Warsaw',       # Poland (CET/CEST)
        'CZ': 'Europe/Prague',       # Czech Republic (CET/CEST)
        'PT': 'Europe/Lisbon',       # Portugal (WET/WEST)
        'GR': 'Europe/Athens',       # Greece (EET/EEST)
        'UK': 'Europe/London',       # United Kingdom (GMT/BST)
    }


def add_timezone_aware_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add timezone-aware datetime columns.

    Assumes 'Start' column is in UTC+1 (CET without DST).
    Creates:
    - dt_utc: Datetime in UTC
    - dt_local: Datetime in each city's local timezone (with DST handling)

    Args:
        df: DataFrame with 'Start', 'Country', and 'SiteNumber' columns

    Returns:
        DataFrame with dt_utc and dt_local columns added
    """
    print("\nAdding timezone-aware datetime columns...")

    # Step 1: Convert Start (UTC+1) to UTC
    # Start is in UTC+1, so subtract 1 hour to get UTC
    utc_tz = pytz.UTC
    df['dt_utc'] = pd.to_datetime(df['Start']) - pd.Timedelta(hours=1)
    df['dt_utc'] = df['dt_utc'].dt.tz_localize(utc_tz)

    print(f"  Created dt_utc column (UTC timezone)")

    # Step 2: Convert UTC to local timezone for each country
    country_tz_map = get_country_timezone_map()

    # Create dt_local column by converting UTC to each country's timezone
    def convert_to_local_tz(row):
        country = row['Country']
        dt_utc = row['dt_utc']

        if country in country_tz_map:
            local_tz = pytz.timezone(country_tz_map[country])
            return dt_utc.astimezone(local_tz)
        else:
            # If country not in map, return UTC time
            print(f"  Warning: No timezone mapping for country '{country}', using UTC")
            return dt_utc

    df['dt_local'] = df.apply(convert_to_local_tz, axis=1)

    print(f"  Created dt_local column (local timezone with DST handling)")
    print(f"  Sample timezones by country:")
    for country in df['Country'].unique():
        country_sample = df[df['Country'] == country].iloc[0]
        tz_name = country_tz_map.get(country, 'UTC')
        sample_utc = country_sample['dt_utc']
        sample_local = country_sample['dt_local']
        print(f"    {country} ({tz_name}): UTC={sample_utc} -> Local={sample_local}")

    return df


def fetch_and_merge_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch weather data from Open-Meteo API and merge with air quality data.

    Fetches hourly weather variables for all unique locations in the dataset:
    - temperature_2m, relative_humidity_2m, dew_point_2m
    - wind_speed_10m, wind_direction_10m (converted to u/v components)
    - precipitation, surface_pressure

    Args:
        df: DataFrame with Latitude, Longitude, and dt_utc columns

    Returns:
        DataFrame with 7 additional weather columns
    """
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry
    import time

    print("\nFetching weather data from Open-Meteo API...")
    print("  Note: Adding 2-second delay between requests to respect API rate limits")

    # Setup API client with caching
    # Cache to SQLite database to persist across runs
    # Use absolute path to ensure cache is always found regardless of working directory
    cache_path = Path(__file__).resolve().parent.parent / 'weather_cache'
    cache_session = requests_cache.CachedSession(
        str(cache_path),  # Cache database name (absolute path)
        backend='sqlite',
        expire_after=-1,  # Never expire
        allowable_methods=['GET', 'POST'],
        stale_if_error=True
    )

    # Check cache status
    print(f"  Cache info: {len(cache_session.cache.responses)} cached responses")

    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Get unique locations
    unique_locations = df[['Country', 'SiteNumber', 'Latitude', 'Longitude']].drop_duplicates()
    print(f"  Found {len(unique_locations)} unique locations")

    # Round coordinates to 4 decimal places for API grid alignment
    unique_locations['Lat_rounded'] = unique_locations['Latitude'].round(4)
    unique_locations['Lon_rounded'] = unique_locations['Longitude'].round(4)

    # Add rounded coordinates to main df for merging
    df = df.merge(
        unique_locations[['Country', 'SiteNumber', 'Lat_rounded', 'Lon_rounded']],
        on=['Country', 'SiteNumber'],
        how='left'
    )

    # Fetch weather data for each unique rounded location
    all_weather_data = []
    api_calls_made = 0
    locations_to_fetch = unique_locations[['Lat_rounded', 'Lon_rounded']].drop_duplicates()
    total_locations = len(locations_to_fetch)

    print(f"  Total unique locations to fetch: {total_locations}")

    url = "https://archive-api.open-meteo.com/v1/archive"

    for counter, (idx, row) in enumerate(locations_to_fetch.iterrows(), 1):
        lat = row['Lat_rounded']
        lon = row['Lon_rounded']

        print(f"  [{counter}/{total_locations}] Fetching weather for lat={lat:.4f}, lon={lon:.4f}")

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": "2018-01-01",
            "end_date": "2024-12-31",
            "hourly": [
                "temperature_2m", "relative_humidity_2m", "dew_point_2m",
                "wind_speed_10m", "wind_direction_10m",
                "precipitation", "surface_pressure"
            ],
            "timezone": "UTC",  # Ensure UTC timestamps for alignment
            # Add your API key here for unlimited requests (paid tier):
            # "apikey": "YOUR_API_KEY_HERE"
        }

        try:
            # Time the request to detect if it's from cache
            start_time = time.time()
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]
            request_duration = time.time() - start_time

            # Extract hourly data
            hourly = response.Hourly()

            # Create datetime range
            hourly_data = {
                "dt_utc": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                )
            }

            # Extract variables (order must match request)
            hourly_data["temperature_2m"] = hourly.Variables(0).ValuesAsNumpy()
            hourly_data["relative_humidity_2m"] = hourly.Variables(1).ValuesAsNumpy()
            hourly_data["dew_point_2m"] = hourly.Variables(2).ValuesAsNumpy()
            hourly_data["wind_speed_10m"] = hourly.Variables(3).ValuesAsNumpy()
            hourly_data["wind_direction_10m"] = hourly.Variables(4).ValuesAsNumpy()
            hourly_data["precipitation"] = hourly.Variables(5).ValuesAsNumpy()
            hourly_data["surface_pressure"] = hourly.Variables(6).ValuesAsNumpy()

            # Add location identifiers
            weather_df = pd.DataFrame(data=hourly_data)
            weather_df['Lat_rounded'] = lat
            weather_df['Lon_rounded'] = lon

            # Convert wind speed/direction to u/v components
            # u = east-west component, v = north-south component
            wind_dir_rad = np.deg2rad(weather_df['wind_direction_10m'])
            weather_df['wind_u'] = weather_df['wind_speed_10m'] * np.cos(wind_dir_rad)
            weather_df['wind_v'] = weather_df['wind_speed_10m'] * np.sin(wind_dir_rad)

            # Drop raw wind columns (we use u/v components instead)
            weather_df = weather_df.drop(['wind_speed_10m', 'wind_direction_10m'], axis=1)

            all_weather_data.append(weather_df)

            # Detect if request was from cache based on duration
            if request_duration < 1.0:
                # Fast response = from cache
                print(f"      → Loaded from cache. Skipping sleep.")
            else:
                # Slow response = fresh API call
                api_calls_made += 1
                print(f"      → API call successful. Sleeping 5s...")
                time.sleep(5)

        except Exception as e:
            error_msg = str(e)
            print(f"    Warning: Failed to fetch weather for lat={lat:.4f}, lon={lon:.4f}: {e}")

            # If rate limited, wait longer before continuing
            if 'rate limit' in error_msg.lower() or 'limit exceeded' in error_msg.lower():
                print(f"    Waiting 60 seconds due to rate limit...")
                time.sleep(60)

            continue

    # Combine all weather data
    if not all_weather_data:
        print("  ERROR: No weather data fetched!")
        return df

    weather_combined = pd.concat(all_weather_data, ignore_index=True)
    print(f"\n  Successfully fetched {len(all_weather_data)} locations")
    print(f"  API calls made: {api_calls_made} (rest from cache)")
    print(f"  Total weather records: {len(weather_combined):,}")

    # Merge with main dataframe
    print(f"  Merging weather data with air quality data...")
    df = df.merge(
        weather_combined,
        on=['Lat_rounded', 'Lon_rounded', 'dt_utc'],
        how='left'
    )

    # Drop temporary rounding columns
    df = df.drop(['Lat_rounded', 'Lon_rounded'], axis=1)

    # Check coverage
    weather_cols = ['temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
                    'wind_u', 'wind_v', 'precipitation', 'surface_pressure']

    print(f"\n  Weather data coverage:")
    for col in weather_cols:
        if col in df.columns:
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            print(f"    {col}: {100 - missing_pct:.1f}% coverage ({missing_pct:.1f}% missing)")

    return df


def impute_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing weather features using time-based interpolation.

    Strategy (same as NO2/PM10 imputation):
    1. Sort by Country, SiteNumber, dt_utc (ensure temporal order)
    2. Interpolate within each site group
    3. For remaining NaNs at edges, use forward/backward fill within site
    4. For any still-remaining NaNs, use site mean

    Args:
        df: DataFrame with weather columns

    Returns:
        DataFrame with imputed weather features
    """
    weather_cols = ['temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
                    'wind_u', 'wind_v', 'precipitation', 'surface_pressure']

    # Check which weather columns exist
    weather_cols_to_impute = [col for col in weather_cols if col in df.columns]

    if not weather_cols_to_impute:
        print("  No weather columns found to impute")
        return df

    print("\nImputing missing weather features...")

    # Sort by site and time for proper interpolation
    df = df.sort_values(['Country', 'SiteNumber', 'dt_utc']).reset_index(drop=True)

    for col in weather_cols_to_impute:
        initial_missing = df[col].isna().sum()

        if initial_missing == 0:
            print(f"  {col}: No missing values")
            continue

        print(f"  {col}: Imputing {initial_missing:,} values ({initial_missing/len(df)*100:.2f}%)")

        # Step 1: Interpolate within each site
        df[col] = df.groupby(['Country', 'SiteNumber'])[col].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )

        # Step 2: For remaining NaNs (at edges), fill with site mean
        df[col] = df.groupby(['Country', 'SiteNumber'])[col].transform(
            lambda x: x.fillna(x.mean())
        )

        final_missing = df[col].isna().sum()
        imputed = initial_missing - final_missing

        print(f"    → Imputed {imputed:,} values")
        if final_missing > 0:
            print(f"    → Remaining missing: {final_missing:,}")

    return df


def replace_sentinel_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace sentinel values with NaN.
    Sentinel values: -999, -9999, -5499, and any negative values.
    Air quality measurements cannot be negative.
    """
    pollutant_cols = ['PM2.5', 'NO2', 'PM10']

    for col in pollutant_cols:
        if col in df.columns:
            # Count sentinels before replacement
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                print(f"{col}: Replacing {negative_count:,} negative/sentinel values with NaN")
                df.loc[df[col] < 0, col] = np.nan

    return df


def filter_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where PM2.5 (target variable) is missing.
    """
    initial_rows = len(df)
    df_clean = df[df['PM2.5'].notna()].copy()
    removed_rows = initial_rows - len(df_clean)

    print(f"\nFiltered out {removed_rows:,} rows with missing PM2.5")
    print(f"Remaining rows: {len(df_clean):,}")

    # Show missing percentages for features
    print("\nMissing values after PM2.5 filtering:")
    for col in ['NO2', 'PM10']:
        if col in df_clean.columns:
            missing_pct = (df_clean[col].isna().sum() / len(df_clean)) * 100
            print(f"  {col}: {missing_pct:.2f}%")

    return df_clean


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing NO2 and PM10 values using time-based interpolation.

    Strategy:
    1. Sort by Country, SiteNumber, Start (ensure temporal order)
    2. Interpolate within each site group
    3. For remaining NaNs at edges, use forward/backward fill within site
    4. For any still-remaining NaNs, use site mean
    """
    df = df.sort_values(['Country', 'SiteNumber', 'Start']).reset_index(drop=True)

    for col in ['NO2', 'PM10']:
        if col not in df.columns:
            continue

        print(f"\nImputing {col}...")
        initial_missing = df[col].isna().sum()

        # Interpolate within each site
        df[col] = df.groupby(['Country', 'SiteNumber'])[col].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )

        # For remaining NaNs (at edges), fill with site mean
        df[col] = df.groupby(['Country', 'SiteNumber'])[col].transform(
            lambda x: x.fillna(x.mean())
        )

        final_missing = df[col].isna().sum()
        imputed = initial_missing - final_missing
        print(f"  Imputed {imputed:,} values")
        print(f"  Remaining missing: {final_missing:,}")

    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from dt_local datetime column.

    Uses local time (with DST handling) instead of UTC to ensure temporal
    patterns like rush hour are captured correctly for each city.

    Creates:
    - hour (0-23) - LOCAL hour
    - day_of_week (0-6, Monday=0) - LOCAL day
    - day_of_month (1-31) - LOCAL day
    - month (1-12) - LOCAL month
    - year (2018-2024) - LOCAL year
    - is_weekend (boolean) - based on LOCAL day
    - season (Winter/Spring/Summer/Fall) - based on LOCAL month
    - hour_sin, hour_cos (cyclical encoding of LOCAL hour)
    - month_sin, month_cos (cyclical encoding of LOCAL month)
    """
    print("\nCreating temporal features from local time (dt_local)...")

    # Basic time features from LOCAL time
    # Since dt_local contains mixed timezones, use apply() instead of .dt accessor
    df['hour'] = df['dt_local'].apply(lambda x: x.hour)
    df['day_of_week'] = df['dt_local'].apply(lambda x: x.dayofweek)
    df['day_of_month'] = df['dt_local'].apply(lambda x: x.day)
    df['month'] = df['dt_local'].apply(lambda x: x.month)
    df['year'] = df['dt_local'].apply(lambda x: x.year)

    # Weekend flag (Saturday=5, Sunday=6)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Season (meteorological seasons based on Northern Hemisphere)
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    df['season'] = df['month'].apply(get_season)

    # Cyclical encoding for hour (24-hour cycle)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Cyclical encoding for month (12-month cycle)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    print(f"  Created 11 temporal features from local time")
    print(f"  Example: Country AT hour range: {df[df['Country']=='AT']['hour'].min()}-{df[df['Country']=='AT']['hour'].max()}")
    print(f"  Example: Country FI hour range: {df[df['Country']=='FI']['hour'].min()}-{df[df['Country']=='FI']['hour'].max()}")

    return df




def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create TIME-BASED lag features for PM2.5, NO2, and PM10.

    Uses actual time-based shifting (not row-based) to handle data gaps correctly.
    For example, lag_1h always refers to 1 hour ago in real time, even if there
    are missing rows in between.

    Strategy:
    1. Drop duplicate timestamps within each site
    2. For each site, reindex to complete hourly grid once (fills gaps with NaN)
    3. Apply row-based shift on complete grid (now equivalent to time-based)
    4. Reindex back to original timestamps only
    5. Assign using index-aligned Series (maintains alignment)

    Lags:
    - 1 hour (previous hour)
    - 2 hours
    - 3 hours
    - 6 hours
    - 12 hours
    - 24 hours (same hour yesterday)
    - 168 hours (same hour last week)

    Important: Lags are created within each site group using dt_utc as the time reference.
    """
    print("\nCreating TIME-BASED lag features...")

    df = df.copy()
    df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True, errors="coerce")

    df = df.sort_values(["Country", "SiteNumber", "dt_utc"]).reset_index(drop=True)

    initial_rows = len(df)
    df = df.drop_duplicates(subset=["Country", "SiteNumber", "dt_utc"], keep="first")
    dropped_dupes = initial_rows - len(df)
    if dropped_dupes > 0:
        print(f"  Dropped {dropped_dupes:,} duplicate timestamps")

    pollutants = ["PM2.5", "NO2", "PM10"]
    lag_periods = {"1h": 1, "2h": 2, "3h": 3, "6h": 6, "12h": 12, "24h": 24, "168h": 168}

    # Initialize lag columns
    for p in pollutants:
        if p not in df.columns:
            continue
        for lag_name in lag_periods:
            df[f"{p}_lag_{lag_name}"] = np.nan

    feature_count = 0

    for (country, site), group in df.groupby(["Country", "SiteNumber"], sort=False):
        group_idx = group.index
        g = group.set_index("dt_utc")

        # Create complete hourly grid (once per site)
        full_range = pd.date_range(g.index.min(), g.index.max(), freq="h", tz="UTC")

        # Build hourly series once per pollutant
        hourly = {}
        for p in pollutants:
            if p in g.columns:
                hourly[p] = g[p].reindex(full_range)

        # Create all lags for all pollutants (efficient: grid built once)
        for p, s in hourly.items():
            for lag_name, k in lag_periods.items():
                col = f"{p}_lag_{lag_name}"
                shifted = s.shift(k)                         # hour-based now
                mapped = shifted.reindex(g.index)            # back to original times

                # Safest assignment: index-aligned to group rows
                df.loc[group_idx, col] = pd.Series(mapped.to_numpy(), index=group_idx)

                feature_count += 1

    print(f"  Created {feature_count} lag features (time-based)")
    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create TIME-BASED rolling mean and std features for PM2.5, NO2, and PM10.

    Uses actual time-based windows (not row-based) to handle data gaps correctly.
    For example, rolling_mean_3h always refers to the past 3 hours in real time,
    even if there are missing rows in between.

    Strategy:
    1. Uses the same complete hourly grid from lag features processing
    2. Apply time-based rolling on complete grid
    3. Reindex back to original timestamps only
    4. Assign using index-aligned Series (maintains alignment)

    Windows:
    - 3 hours
    - 6 hours
    - 12 hours
    - 24 hours

    For each window, creates:
    - Rolling mean (average over past N hours)
    - Rolling std (standard deviation over past N hours - captures volatility)

    Uses right-aligned windows (default) to avoid lookahead bias.
    Created within each site group using dt_utc as the time reference.
    """
    print("\nCreating TIME-BASED rolling mean and std features...")

    df = df.copy()
    df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True, errors="coerce")

    df = df.sort_values(["Country", "SiteNumber", "dt_utc"]).reset_index(drop=True)

    # Note: duplicates already dropped in create_lag_features

    pollutants = ["PM2.5", "NO2", "PM10"]
    windows = {"3h": 3, "6h": 6, "12h": 12, "24h": 24}

    # Initialize rolling columns
    for p in pollutants:
        if p not in df.columns:
            continue
        for window_name in windows:
            df[f"{p}_rolling_mean_{window_name}"] = np.nan
            df[f"{p}_rolling_std_{window_name}"] = np.nan

    feature_count = 0

    for _, group in df.groupby(["Country", "SiteNumber"], sort=False):
        group_idx = group.index
        g = group.set_index("dt_utc")

        # Create complete hourly grid (once per site)
        full_range = pd.date_range(g.index.min(), g.index.max(), freq="h", tz="UTC")

        # Build hourly series once per pollutant
        hourly = {}
        for p in pollutants:
            if p in g.columns:
                hourly[p] = g[p].reindex(full_range)

        # Create all rolling features for all pollutants (efficient: grid built once)
        for p, s in hourly.items():
            for window_name, window_size in windows.items():
                # Rolling mean (time-based)
                col_mean = f"{p}_rolling_mean_{window_name}"
                rolled_mean = s.rolling(window=window_size, min_periods=1).mean()
                mapped_mean = rolled_mean.reindex(g.index)
                df.loc[group_idx, col_mean] = pd.Series(mapped_mean.to_numpy(), index=group_idx)
                feature_count += 1

                # Rolling std (time-based - captures volatility)
                col_std = f"{p}_rolling_std_{window_name}"
                rolled_std = s.rolling(window=window_size, min_periods=1).std()
                mapped_std = rolled_std.reindex(g.index)
                df.loc[group_idx, col_std] = pd.Series(mapped_std.to_numpy(), index=group_idx)
                feature_count += 1

    print(f"  Created {feature_count} rolling features (time-based)")
    return df


def remove_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where lag features are NaN (typically at the start of each site's data).

    This ensures the ML dataset has complete features for training.
    """
    initial_rows = len(df)

    # Find columns that are lag features
    lag_cols = [col for col in df.columns if '_lag_' in col]

    if lag_cols:
        # Remove rows with any NaN in lag columns
        df_clean = df.dropna(subset=lag_cols).copy()

        removed_rows = initial_rows - len(df_clean)
        print(f"\nRemoved {removed_rows:,} rows with incomplete lag features")
        print(f"Final dataset size: {len(df_clean):,} rows")
    else:
        print("\nNo lag features found, keeping all rows")
        df_clean = df.copy()

    return df_clean


def save_ml_dataset(df: pd.DataFrame, output_path: Path, feature_docs_dir: Path = None) -> None:
    """
    Save the ML-ready dataset with all features.

    Also saves a feature list and dataset summary.

    Args:
        df: ML-ready dataframe
        output_path: Path for main dataset CSV
        feature_docs_dir: Directory for feature list and summary (default: same as output_path)
    """
    # Save main dataset
    df.to_csv(output_path, index=False)
    print(f"\nSaved ML dataset to {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Determine where to save documentation files
    if feature_docs_dir is None:
        feature_docs_dir = output_path.parent

    # Save feature list
    feature_list_path = feature_docs_dir / "ml_features.txt"
    with open(feature_list_path, 'w') as f:
        f.write("# ML Dataset Features\n\n")
        f.write("## Metadata\n")
        for col in ['Start', 'Country', 'SiteNumber', 'dt_utc', 'dt_local']:
            if col in df.columns:
                f.write(f"- {col}\n")

        f.write("\n## Site Metadata (Geographic)\n")
        for col in ['Latitude', 'Longitude', 'Altitude']:
            if col in df.columns:
                f.write(f"- {col}\n")

        f.write("\n## Site Metadata (Station Type - One-Hot Encoded)\n")
        station_type_features = [col for col in df.columns if col.startswith('StationType_')]
        for col in sorted(station_type_features):
            f.write(f"- {col}\n")

        f.write("\n## Site Metadata (Station Area - One-Hot Encoded)\n")
        station_area_features = [col for col in df.columns if col.startswith('StationArea_')]
        for col in sorted(station_area_features):
            f.write(f"- {col}\n")

        f.write("\n## Weather Features (Open-Meteo API)\n")
        weather_features = ['temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
                           'wind_u', 'wind_v', 'precipitation', 'surface_pressure']
        for col in weather_features:
            if col in df.columns:
                f.write(f"- {col}\n")

        f.write("\n## Target Variable\n")
        f.write("- PM2.5 (target variable - create y in training scripts as needed)\n")

        f.write("\n## Original Features\n")
        for col in ['NO2', 'PM10']:
            if col in df.columns:
                f.write(f"- {col}\n")

        f.write("\n## Temporal Features\n")
        temporal_features = ['hour', 'day_of_week', 'day_of_month', 'month', 'year',
                            'is_weekend', 'season', 'hour_sin', 'hour_cos',
                            'month_sin', 'month_cos']
        for col in temporal_features:
            if col in df.columns:
                f.write(f"- {col}\n")

        f.write("\n## Lag Features\n")
        lag_features = [col for col in df.columns if '_lag_' in col]
        for col in sorted(lag_features):
            f.write(f"- {col}\n")

        f.write("\n## Rolling Mean Features\n")
        rolling_mean_features = [col for col in df.columns if '_rolling_mean_' in col]
        for col in sorted(rolling_mean_features):
            f.write(f"- {col}\n")

        f.write("\n## Rolling Std Features\n")
        rolling_std_features = [col for col in df.columns if '_rolling_std_' in col]
        for col in sorted(rolling_std_features):
            f.write(f"- {col}\n")

    print(f"Saved feature list to {feature_list_path}")

    # Save summary statistics
    summary_path = feature_docs_dir / "ml_dataset_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("# ML Dataset Summary\n\n")
        f.write(f"Total rows: {len(df):,}\n")
        f.write(f"Total columns: {len(df.columns)}\n")
        f.write(f"Date range: {df['Start'].min()} to {df['Start'].max()}\n")
        f.write(f"Countries: {', '.join(sorted(df['Country'].unique()))}\n")
        f.write(f"Sites: {df['SiteNumber'].nunique()}\n\n")

        f.write("## Missing Values\n")
        missing = df.isnull().sum()
        missing_present = False
        for col in missing[missing > 0].index:
            pct = (missing[col] / len(df)) * 100
            f.write(f"{col}: {missing[col]:,} ({pct:.2f}%)\n")
            missing_present = True

        if not missing_present:
            f.write("No missing values!\n")

    print(f"Saved dataset summary to {summary_path}")


def main():
    """Main ML dataset preparation pipeline."""
    try:
        config = PipelineConfig()
    except (FileNotFoundError, ValueError) as e:
        print(f"Configuration error: {e}")
        print("Using default paths...")
        config = None

    print("=" * 100)
    print("ML DATASET PREPARATION PIPELINE")
    print("=" * 100)

    # Step 1: Load data
    df = load_processed_data(config)

    # Step 2: Replace sentinel values with NaN
    print("\n" + "=" * 100)
    print("STEP 1: REPLACING SENTINEL VALUES")
    print("=" * 100)
    df = replace_sentinel_values(df)

    # Step 3: Filter rows with missing PM2.5
    print("\n" + "=" * 100)
    print("STEP 2: FILTERING MISSING TARGET VARIABLE")
    print("=" * 100)
    df = filter_missing_target(df)

    # Step 4: Impute missing NO2 and PM10
    print("\n" + "=" * 100)
    print("STEP 3: IMPUTING MISSING FEATURE VALUES")
    print("=" * 100)
    df = impute_missing_values(df)

    # Step 5: Add timezone-aware datetime columns
    print("\n" + "=" * 100)
    print("STEP 4: ADDING TIMEZONE-AWARE DATETIME COLUMNS")
    print("=" * 100)
    df = add_timezone_aware_columns(df)

    # Step 6: Fetch and merge weather data
    print("\n" + "=" * 100)
    print("STEP 5: FETCHING AND MERGING WEATHER DATA")
    print("=" * 100)
    df = fetch_and_merge_weather_data(df)

    # Step 7: Impute missing weather features
    print("\n" + "=" * 100)
    print("STEP 6: IMPUTING MISSING WEATHER FEATURES")
    print("=" * 100)
    df = impute_weather_features(df)

    # Step 8: Create temporal features
    print("\n" + "=" * 100)
    print("STEP 7: CREATING TEMPORAL FEATURES")
    print("=" * 100)
    df = create_temporal_features(df)

    # Step 9: Create lag features
    print("\n" + "=" * 100)
    print("STEP 8: CREATING LAG FEATURES")
    print("=" * 100)
    df = create_lag_features(df)

    # Step 10: Create rolling features
    print("\n" + "=" * 100)
    print("STEP 9: CREATING ROLLING MEAN AND STD FEATURES")
    print("=" * 100)
    df = create_rolling_features(df)

    # Step 11: Remove incomplete rows
    print("\n" + "=" * 100)
    print("STEP 10: REMOVING INCOMPLETE ROWS")
    print("=" * 100)
    df = remove_incomplete_rows(df)

    # Step 12: Save ML dataset
    print("\n" + "=" * 100)
    print("STEP 11: SAVING ML DATASET")
    print("=" * 100)
    # Save main dataset to project root (used by training scripts)
    output_path = Path("ml_ready_dataset_v6.csv")
    # Save feature list and summary to dataset_build/
    feature_docs_dir = Path("dataset_build")
    save_ml_dataset(df, output_path, feature_docs_dir)

    print("\n" + "=" * 100)
    print("✓ PIPELINE COMPLETE")
    print("=" * 100)
    print(f"\nML-ready dataset: {output_path}")
    print(f"Feature list: {feature_docs_dir / 'ml_features.txt'}")
    print(f"Dataset summary: {feature_docs_dir / 'ml_dataset_summary.txt'}")


if __name__ == "__main__":
    main()
