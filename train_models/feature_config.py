"""
Shared feature configuration for all models.

Based on Beijing AQI project analysis, these features provide optimal performance:
- PM2.5 rolling_mean_3h contributes 87-90% of predictive power
- Short-interval lags (2h, 3h, 6h, 12h) capture recent trends
- Rolling std features capture volatility/stability
"""

def get_comprehensive_features():
    """
    Get comprehensive feature set for 1-hour ahead PM2.5 prediction.

    Returns:
        list: Feature column names (excluding categorical features like season)
    """
    features = []

    # PM2.5 lag features (CRITICAL - especially short intervals)
    features.extend([
        'PM2.5_lag_1h', 'PM2.5_lag_2h', 'PM2.5_lag_3h',
        'PM2.5_lag_6h', 'PM2.5_lag_12h', 'PM2.5_lag_24h'
    ])

    # PM2.5 rolling mean features (CRITICAL - rolling_mean_3h is most important!)
    features.extend([
        'PM2.5_rolling_mean_3h', 'PM2.5_rolling_mean_6h',
        'PM2.5_rolling_mean_12h', 'PM2.5_rolling_mean_24h'
    ])

    # PM2.5 rolling std features (captures volatility)
    features.extend([
        'PM2.5_rolling_std_3h', 'PM2.5_rolling_std_6h',
        'PM2.5_rolling_std_12h', 'PM2.5_rolling_std_24h'
    ])

    # NO2 features (correlated with PM2.5)
    features.extend([
        'NO2',  # Current value
        'NO2_lag_1h', 'NO2_lag_3h', 'NO2_lag_6h', 'NO2_lag_12h', 'NO2_lag_24h',
        'NO2_rolling_mean_3h', 'NO2_rolling_mean_6h', 'NO2_rolling_mean_12h',
        'NO2_rolling_std_3h', 'NO2_rolling_std_6h'
    ])

    # PM10 features (correlated with PM2.5)
    features.extend([
        'PM10',  # Current value
        'PM10_lag_1h', 'PM10_lag_3h', 'PM10_lag_6h', 'PM10_lag_12h', 'PM10_lag_24h',
        'PM10_rolling_mean_3h', 'PM10_rolling_mean_6h', 'PM10_rolling_mean_12h',
        'PM10_rolling_std_3h', 'PM10_rolling_std_6h'
    ])

    # Temporal features (cyclical encoding prevents discontinuities)
    features.extend([
        'hour_sin', 'hour_cos',
        'month_sin', 'month_cos',
        'day_of_week',
        'is_weekend'
    ])

    # Site metadata - Geographic features
    features.extend([
        'Latitude', 'Longitude', 'Altitude'
    ])

    # Weather features (from Open-Meteo API)
    features.extend([
        'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
        'wind_u', 'wind_v',  # Wind components (u=east-west, v=north-south)
        'precipitation', 'surface_pressure'
    ])

    return features


def get_station_metadata_features(df):
    """
    Get one-hot encoded station metadata features from DataFrame.

    Args:
        df: DataFrame that may contain StationType_* and StationArea_* columns

    Returns:
        list: Column names for one-hot encoded station metadata features
    """
    metadata_features = []

    # Get StationType one-hot encoded columns
    station_type_cols = [col for col in df.columns if col.startswith('StationType_')]
    metadata_features.extend(sorted(station_type_cols))

    # Get StationArea one-hot encoded columns
    station_area_cols = [col for col in df.columns if col.startswith('StationArea_')]
    metadata_features.extend(sorted(station_area_cols))

    return metadata_features


def get_station_encoding_columns(df):
    """
    Create station encoding from Country + SiteNumber.

    Args:
        df: DataFrame with 'Country' and 'SiteNumber' columns

    Returns:
        tuple: (DataFrame with station_encoded column, feature name)
    """
    from sklearn.preprocessing import LabelEncoder

    # Combine Country and SiteNumber for unique station ID
    df = df.copy()
    df['station_id'] = df['Country'].astype(str) + '_' + df['SiteNumber'].astype(str)

    # Encode to numeric
    le = LabelEncoder()
    df['station_encoded'] = le.fit_transform(df['station_id'])

    return df, 'station_encoded'


def print_feature_summary(features):
    """Print summary of feature groups."""
    print("\nFeature Summary:")
    print(f"  Total features: {len(features)}")

    pm25_features = [f for f in features if f.startswith('PM2.5_')]
    no2_features = [f for f in features if f.startswith('NO2')]
    pm10_features = [f for f in features if f.startswith('PM10')]
    temporal_features = [f for f in features if any(x in f for x in ['hour', 'month', 'day', 'weekend'])]
    geographic_features = [f for f in features if f in ['Latitude', 'Longitude', 'Altitude']]
    station_type_features = [f for f in features if f.startswith('StationType_')]
    station_area_features = [f for f in features if f.startswith('StationArea_')]
    weather_features = [f for f in features if f in [
        'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
        'wind_u', 'wind_v', 'precipitation', 'surface_pressure'
    ]]

    print(f"  PM2.5 features: {len(pm25_features)}")
    print(f"  NO2 features: {len(no2_features)}")
    print(f"  PM10 features: {len(pm10_features)}")
    print(f"  Temporal features: {len(temporal_features)}")
    print(f"  Geographic features: {len(geographic_features)}")
    print(f"  Station type features: {len(station_type_features)}")
    print(f"  Station area features: {len(station_area_features)}")
    print(f"  Weather features: {len(weather_features)}")
    if 'station_encoded' in features:
        print(f"  Station encoding: 1")

    print("\nMost Critical Features (based on Beijing analysis):")
    print("  1. PM2.5_rolling_mean_3h (87-90% importance)")
    print("  2. PM2.5_lag_1h, PM2.5_lag_3h")
    print("  3. PM2.5_rolling_std features (volatility)")
