"""
Process air quality data from per-pollutant files to wide format.

This script:
1. Reads config to determine which pollutants to include
2. Loads filtered data for each pollutant from data/
3. Combines into single dataframe
4. Pivots from long to wide format
5. Merges with site metadata (lat/lon/altitude, station type/area)
6. Saves to processed CSV
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'utils'))

import pandas as pd
import re
from config_utils import PipelineConfig

# Country name to country code mapping
COUNTRY_NAME_TO_CODE = {
    'Austria': 'AT',
    'Belgium': 'BE',
    'Spain': 'ES',
    'Finland': 'FI',
    'France': 'FR',
    'Germany': 'DE',
    'Italy': 'IT',
    'Netherlands': 'NL',
    'Poland': 'PL',
    'Sweden': 'SE',
    'Denmark': 'DK',
    'Norway': 'NO',
    'Greece': 'GR',
    'Portugal': 'PT',
    'Czechia': 'CZ',
    'Hungary': 'HU',
    'Romania': 'RO',
    'Bulgaria': 'BG',
    'Croatia': 'HR',
    'Slovakia': 'SK',
    'Slovenia': 'SI',
    'Estonia': 'EE',
    'Latvia': 'LV',
    'Lithuania': 'LT',
    'Luxembourg': 'LU',
    'Malta': 'MT',
    'Cyprus': 'CY',
    'Ireland': 'IE',
    'United Kingdom': 'GB',
    'Switzerland': 'CH',
    'Iceland': 'IS',
    'Albania': 'AL',
    'Andorra': 'AD',
    'Bosnia and Herzegovina': 'BA',
    'Georgia': 'GE',
    'Kosovo under UNSCR 1244/99': 'XK',
    'Montenegro': 'ME',
    'North Macedonia': 'MK',
    'Serbia': 'RS',
    'Türkiye': 'TR',
    'Ukraine': 'UA',
}


def extract_site_number(samplingpoint: str) -> str:
    """
    Extract site number from Samplingpoint string.

    DEPRECATED: Use extract_station_eoi_code() instead for metadata matching.
    This function is kept for backward compatibility with SiteNumber column.

    Args:
        samplingpoint: Sampling point identifier string

    Returns:
        Site number or None if pattern doesn't match
    """
    country = samplingpoint.split('/')[0]

    patterns = {
        'DE': r'DE_([A-Z0-9]+)',
        'ES': r'^ES/SP_(\d+)',
        'FR': r'FR(\d+)',
        'IT': r'IT(\d+[A-Z])',
        'PL': r'PL(\d+[A-Z])',
        'RO': r'RO(\d+[A-Z])',
        'SE': r'SE(\d+)',
        'AT': r'\.09\.([A-Z]+)\.',
        'FI': r'FI(\d{5})',
        'BE': r'BE([A-Z]{2}\d+)'
    }

    if country in patterns:
        match = re.search(patterns[country], samplingpoint)
        if match:
            return match.group(1)

    return None


def extract_station_eoi_code(samplingpoint: str) -> str:
    """
    Extract Air Quality Station EoI Code from Samplingpoint.

    This code matches the 'Air Quality Station EoI Code' in site_metadata.csv
    and is used for merging metadata (lat/lon/altitude, station type/area).

    Examples:
        FI/SPO-FI00208_06001_100 -> FI00208
        FR/SPO-FR04024_6001 -> FR04024
        BE/SPO-BETR817_06001_100 -> BETR817
        ES/SP_28079056_9_47 -> (needs manual mapping)

    Args:
        samplingpoint: Sampling point identifier string

    Returns:
        Station EoI code or None if not extractable
    """
    if '/' not in samplingpoint:
        return None

    country, spo_part = samplingpoint.split('/', 1)

    # Country-specific patterns to extract full EoI code
    patterns = {
        # Finland: SPO-{CountryCode}{5 digits}_{...}
        # Examples: FI/SPO-FI00208_06001_100
        'FI': r'SPO-(FI\d{5})',

        # France: SPO-{CountryCode}{5 digits}_{...}
        # Examples: FR/SPO-FR04024_6001
        'FR': r'SPO-(FR\d{5})',

        # Belgium: SPO-{StationCode}_{...}
        # Examples: BE/SPO-BETR817_06001_100
        # Pattern: BE + 2-4 letters + digits
        'BE': r'SPO-(BE[A-Z0-9]+)',

        # Austria: SPO.09.{Code}.{...}
        # Examples: AT/SPO.09.A23.65516.6001.1
        # Note: This extracts partial code, may need manual mapping
        'AT': r'SPO\.09\.([A-Z0-9]+)\.',

        # Spain: SP_{CityCode}_{...}
        # Examples: ES/SP_28079056_9_47
        # Note: May need manual mapping - EoI codes are different format
        'ES': r'SP_(\d+)',

        # Add more countries as patterns are identified
    }

    if country in patterns:
        match = re.search(patterns[country], spo_part)
        if match:
            return match.group(1)

    return None


def load_pollutant_data(
    pollutant_name: str,
    config: PipelineConfig,
    use_filtered: bool = True
) -> pd.DataFrame:
    """
    Load data for a single pollutant.

    Args:
        pollutant_name: Name of pollutant (e.g., 'PM2.5', 'NO2')
        config: Pipeline configuration
        use_filtered: If True, use _filtered.parquet files

    Returns:
        DataFrame with columns: Samplingpoint, Start, Value, PollutantName
        Returns None if file doesn't exist
    """
    if use_filtered:
        file_path = config.get_filtered_pollutant_file(pollutant_name)
    else:
        file_path = config.get_pollutant_file(pollutant_name)

    if not file_path.exists():
        print(f"⚠ Warning: {pollutant_name} file not found: {file_path}")
        return None

    print(f"Loading {pollutant_name} from {file_path}...")
    df = pd.read_parquet(file_path)

    # Add pollutant name column
    df['PollutantName'] = pollutant_name

    # Keep only necessary columns
    df = df[['Samplingpoint', 'Start', 'Value', 'PollutantName']].copy()

    print(f"  Loaded {len(df):,} rows")

    return df


def combine_pollutants(config: PipelineConfig) -> pd.DataFrame:
    """
    Combine all pollutant files into single long-format dataframe.

    Args:
        config: Pipeline configuration

    Returns:
        Combined dataframe with all pollutants

    Raises:
        ValueError: If no pollutant data loaded
    """
    print("=" * 100)
    print("LOADING POLLUTANT DATA")
    print("=" * 100)

    dataframes = []

    for pollutant in config.pollutants:
        df = load_pollutant_data(pollutant['name'], config, use_filtered=True)
        if df is not None:
            dataframes.append(df)

    if not dataframes:
        raise ValueError("No pollutant data loaded! Run download_pollutants.py and filter_pm25_sites.py first.")

    print(f"\nCombining {len(dataframes)} pollutants...")
    combined = pd.concat(dataframes, ignore_index=True)

    print(f"Total rows: {len(combined):,}")

    return combined


def load_site_metadata(metadata_file: str = 'site_metadata.csv') -> pd.DataFrame:
    """
    Load and prepare site metadata.

    Loads metadata CSV and deduplicates by station (since each station
    has multiple rows for different pollutants).

    Args:
        metadata_file: Path to site metadata CSV file

    Returns:
        DataFrame with columns: Country, Station_EoI, Latitude,
        Longitude, Altitude, Station_Type, Station_Area
        Returns None if file not found.
    """
    print("\n" + "=" * 100)
    print("LOADING SITE METADATA")
    print("=" * 100)

    metadata_path = Path(metadata_file)
    if not metadata_path.exists():
        print(f"⚠ Warning: Metadata file not found: {metadata_path}")
        print("  Continuing without metadata features...")
        return None

    # Load metadata
    print(f"Loading metadata from {metadata_path}...")
    meta = pd.read_csv(metadata_path, low_memory=False)
    print(f"  Loaded {len(meta):,} rows")

    # Map country names to codes
    meta['Country'] = meta['Country'].map(COUNTRY_NAME_TO_CODE)

    # Rename columns for clarity
    meta = meta.rename(columns={
        'Air Quality Station EoI Code': 'Station_EoI',
        'Latitude': 'Latitude',
        'Longitude': 'Longitude',
        'Altitude': 'Altitude',
        'Air Quality Station Type': 'Station_Type',
        'Air Quality Station Area': 'Station_Area'
    })

    # Select relevant columns
    cols = ['Country', 'Station_EoI', 'Latitude', 'Longitude',
            'Altitude', 'Station_Type', 'Station_Area']
    meta = meta[cols].copy()

    # Deduplicate by Country + Station (each station has multiple pollutants)
    meta_dedup = meta.drop_duplicates(subset=['Country', 'Station_EoI'])

    print(f"  Unique stations: {len(meta_dedup):,}")
    print(f"  Countries: {meta_dedup['Country'].nunique()}")

    # Check for missing values
    print("\n  Metadata completeness:")
    for col in ['Latitude', 'Longitude', 'Altitude', 'Station_Type', 'Station_Area']:
        missing_pct = (meta_dedup[col].isna().sum() / len(meta_dedup)) * 100
        print(f"    {col}: {100-missing_pct:.1f}% complete")

    return meta_dedup


def merge_site_metadata(df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Merge site metadata into main dataframe.

    Adds:
    - Numeric features: Latitude, Longitude, Altitude
    - Categorical features (one-hot): Station_Type, Station_Area

    Uses country-specific matching strategies:
    - FI, FR, BE: Direct EoI code extraction
    - AT: Match numeric code in Sampling Point Id
    - ES: Match city code in Nat Code

    Args:
        df: Main dataframe with Country and Samplingpoint columns
        metadata: Metadata dataframe from load_site_metadata()

    Returns:
        DataFrame with metadata columns added
    """
    if metadata is None:
        print("\nSkipping metadata merge (no metadata available)")
        return df

    print("\n" + "=" * 100)
    print("MERGING SITE METADATA")
    print("=" * 100)

    # Load full metadata with all columns for special matching
    print("Loading full metadata for country-specific matching...")
    meta_full = pd.read_csv('site_metadata.csv', low_memory=False)

    # Map country names to codes in full metadata
    meta_full['Country'] = meta_full['Country'].map(COUNTRY_NAME_TO_CODE)

    # Extract Country from Samplingpoint
    print("Extracting country codes from samplingpoints...")
    df['Country'] = df['Samplingpoint'].str.split('/').str[0]

    # Extract Station EoI Code from Samplingpoint (works for FI, FR, BE)
    print("Extracting station EoI codes from samplingpoints...")
    df['Station_EoI'] = df['Samplingpoint'].apply(extract_station_eoi_code)

    # Initialize Station_EoI_Matched column (will be filled for all countries)
    df['Station_EoI_Matched'] = df['Station_EoI']

    # Special handling for Austria (AT)
    print("Special matching for Austria (AT)...")
    at_mask = df['Country'] == 'AT'
    if at_mask.sum() > 0:
        # Extract numeric code before pollutant code (works for all pollutants: 5, 8, 6001, etc.)
        df.loc[at_mask, 'AT_numeric'] = df.loc[at_mask, 'Samplingpoint'].str.extract(r'\.(\d+)\.(\d+)\.1')[0]

        # Create lookup: numeric code -> EoI code
        at_meta = meta_full[meta_full['Country'] == 'AT']
        at_lookup = {}
        for _, row in at_meta.iterrows():
            spo_id = str(row['Sampling Point Id'])
            numeric_match = re.search(r'\.(\d+)\.\d+\.', spo_id)
            if numeric_match:
                numeric_code = numeric_match.group(1)
                at_lookup[numeric_code] = row['Air Quality Station EoI Code']

        # Map AT stations
        df.loc[at_mask, 'Station_EoI_Matched'] = df.loc[at_mask, 'AT_numeric'].map(at_lookup)
        matched_at = df.loc[at_mask, 'Station_EoI_Matched'].notna().sum()
        print(f"  Matched {matched_at}/{at_mask.sum()} Austria stations")

    # Special handling for Spain (ES)
    print("Special matching for Spain (ES)...")
    es_mask = df['Country'] == 'ES'
    if es_mask.sum() > 0:
        # Extract city code
        df.loc[es_mask, 'ES_city_code'] = df.loc[es_mask, 'Samplingpoint'].str.extract(r'SP_(\d+)')[0]

        # Create lookup: nat code -> EoI code
        es_meta = meta_full[meta_full['Country'] == 'ES']
        es_lookup = {}
        for _, row in es_meta.iterrows():
            nat_code = str(row['Air Quality Station Nat Code'])
            if nat_code and nat_code != 'nan':
                es_lookup[nat_code] = row['Air Quality Station EoI Code']

        # Map ES stations
        df.loc[es_mask, 'Station_EoI_Matched'] = df.loc[es_mask, 'ES_city_code'].map(es_lookup)
        matched_es = df.loc[es_mask, 'Station_EoI_Matched'].notna().sum()
        print(f"  Matched {matched_es}/{es_mask.sum()} Spain stations")

    # Count total extraction success
    unique_sps = df['Samplingpoint'].unique()
    extracted_sps = df[df['Station_EoI_Matched'].notna()]['Samplingpoint'].unique()
    print(f"\n  Total extracted EoI codes: {len(extracted_sps)}/{len(unique_sps)} ({len(extracted_sps)/len(unique_sps)*100:.1f}%)")

    # Merge with metadata using matched EoI codes
    print("Merging metadata...")
    df_merged = df.merge(
        metadata.rename(columns={'Station_EoI': 'Station_EoI_Matched'}),
        on=['Country', 'Station_EoI_Matched'],
        how='left',
        suffixes=('', '_meta')
    )

    # Check merge success rate
    match_rate = (df_merged['Latitude'].notna().sum() / len(df_merged)) * 100
    print(f"  Metadata match rate: {match_rate:.1f}%")

    # One-hot encode categorical features
    print("\nOne-hot encoding categorical features...")

    # Station Type
    if 'Station_Type' in df_merged.columns:
        station_types = df_merged['Station_Type'].dropna().unique()
        print(f"  Station Types found: {sorted(station_types)}")

        type_dummies = pd.get_dummies(df_merged['Station_Type'], prefix='StationType', dummy_na=False)
        df_merged = pd.concat([df_merged, type_dummies], axis=1)
        print(f"  Created {len(type_dummies.columns)} Station Type features")

    # Station Area
    if 'Station_Area' in df_merged.columns:
        station_areas = df_merged['Station_Area'].dropna().unique()
        print(f"  Station Areas found: {sorted(station_areas)}")

        area_dummies = pd.get_dummies(df_merged['Station_Area'], prefix='StationArea', dummy_na=False)
        df_merged = pd.concat([df_merged, area_dummies], axis=1)
        print(f"  Created {len(area_dummies.columns)} Station Area features")

    # Drop intermediate columns
    cols_to_drop = ['Station_EoI', 'Station_EoI_Matched', 'Station_Type', 'Station_Area',
                    'AT_numeric', 'ES_city_code']
    df_merged = df_merged.drop(columns=cols_to_drop, errors='ignore')

    print(f"\nFinal shape after metadata merge: {df_merged.shape}")

    return df_merged


def pivot_to_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform from long format (one row per measurement) to wide format
    (one row per time/location, pollutants as columns).

    Preserves metadata columns (Latitude, Longitude, Altitude, one-hot features).

    Args:
        df: Long format dataframe

    Returns:
        Wide format dataframe
    """
    print("\n" + "=" * 100)
    print("TRANSFORMING TO WIDE FORMAT")
    print("=" * 100)

    # Extract SiteNumber (Country already extracted in merge_site_metadata)
    print("Extracting site information...")
    if 'Country' not in df.columns:
        df['Country'] = df['Samplingpoint'].str.split('/').str[0]
    df['SiteNumber'] = df['Samplingpoint'].apply(extract_site_number)

    # Ensure Start is datetime
    df['Start'] = pd.to_datetime(df['Start'])

    # Identify metadata columns (numeric and one-hot encoded)
    metadata_cols = []
    for col in df.columns:
        if col in ['Latitude', 'Longitude', 'Altitude']:
            metadata_cols.append(col)
        elif col.startswith('StationType_') or col.startswith('StationArea_'):
            metadata_cols.append(col)

    # Select columns for pivot
    pivot_cols = ['Start', 'Country', 'SiteNumber', 'PollutantName', 'Value']
    df_pivot = df[pivot_cols].copy()

    # Pivot
    print("Pivoting data...")
    df_wide = df_pivot.pivot_table(
        index=['Start', 'Country', 'SiteNumber'],
        columns='PollutantName',
        values='Value',
        aggfunc='mean'  # Handle duplicates
    ).reset_index()

    # Preserve datetime
    df_wide['Start'] = pd.to_datetime(df_wide['Start'])

    # Merge back metadata columns (they're the same for all rows with same Country+SiteNumber)
    if metadata_cols:
        print(f"Merging back {len(metadata_cols)} metadata columns...")
        # Get unique metadata per site
        meta_per_site = df[['Country', 'SiteNumber'] + metadata_cols].drop_duplicates(
            subset=['Country', 'SiteNumber']
        )

        # Merge back
        df_wide = df_wide.merge(
            meta_per_site,
            on=['Country', 'SiteNumber'],
            how='left'
        )

    print(f"Pivoted shape: {df_wide.shape}")
    print(f"Columns: {df_wide.columns.tolist()[:10]}... ({len(df_wide.columns)} total)")

    return df_wide


def reorder_columns(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    Reorder columns: Start, Country, SiteNumber, metadata, then pollutants.
    Add missing pollutant columns with NaN.

    Column order:
    1. Base: Start, Country, SiteNumber
    2. Metadata numeric: Latitude, Longitude, Altitude
    3. Metadata categorical (one-hot): StationType_*, StationArea_*
    4. Pollutants: PM2.5, NO2, PM10, etc.

    Args:
        df: Wide format dataframe
        config: Pipeline configuration

    Returns:
        Dataframe with reordered columns
    """
    # Standard first columns
    base_cols = ['Start', 'Country', 'SiteNumber']

    # Metadata numeric columns (in specific order)
    metadata_numeric = ['Latitude', 'Longitude', 'Altitude']
    metadata_numeric = [col for col in metadata_numeric if col in df.columns]

    # Metadata categorical columns (one-hot encoded)
    station_type_cols = sorted([col for col in df.columns if col.startswith('StationType_')])
    station_area_cols = sorted([col for col in df.columns if col.startswith('StationArea_')])

    # Pollutant columns (from config, in order)
    pollutant_cols = [p['name'] for p in config.pollutants]

    # Add missing pollutant columns
    for col in pollutant_cols:
        if col not in df.columns:
            df[col] = pd.NA

    # Reorder: base + metadata numeric + station type + station area + pollutants
    final_cols = base_cols + metadata_numeric + station_type_cols + station_area_cols + pollutant_cols

    # Keep only columns that exist (in case some metadata is missing)
    final_cols = [col for col in final_cols if col in df.columns]

    df = df[final_cols]

    return df


def main():
    """Main processing orchestrator."""
    try:
        config = PipelineConfig()
    except (FileNotFoundError, ValueError) as e:
        print(f"Configuration error: {e}")
        return

    print("=" * 100)
    print("AIR QUALITY DATA PROCESSING")
    print("=" * 100)
    print(f"Pollutants to include: {', '.join(p['name'] for p in config.pollutants)}")
    print("=" * 100)

    try:
        # Step 1: Load and combine all pollutants
        df_combined = combine_pollutants(config)

        # Step 2: Load site metadata
        metadata = load_site_metadata('site_metadata.csv')

        # Step 3: Merge site metadata (before pivoting, while we have Samplingpoint)
        df_with_meta = merge_site_metadata(df_combined, metadata)

        # Step 4: Pivot to wide format
        df_wide = pivot_to_wide_format(df_with_meta)

        # Step 5: Reorder columns
        df_final = reorder_columns(df_wide, config)

        # Step 6: Save
        output_file = config.output_dir / config.processed_file

        print("\n" + "=" * 100)
        print("SAVING PROCESSED DATA")
        print("=" * 100)

        df_final.to_csv(output_file, index=False)

        print(f"✓ Saved to {output_file}")
        print(f"  Shape: {df_final.shape}")
        print(f"  Columns: {df_final.columns.tolist()}")

        # Show sample
        print(f"\nFirst 5 rows:")
        print(df_final.head())

        # Show missing values
        print("\nMissing values:")
        print(df_final.isnull().sum())

    except ValueError as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()
