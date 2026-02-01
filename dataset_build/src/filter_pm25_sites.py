"""
Filter sites to those that measure PM2.5.

This script:
1. Reads PM2.5 data from data/PM25.parquet
2. Identifies all sites that measure PM2.5
3. Saves site list to data/pm25_sites.txt
4. Creates filtered parquet for each pollutant (sites with PM2.5 only)
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'utils'))

import pandas as pd
import re
from config_utils import PipelineConfig


def extract_site_number(samplingpoint: str) -> str:
    """
    Extract site number from Samplingpoint string.

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


def get_pm25_sites(config: PipelineConfig) -> set:
    """
    Identify all sites that measure PM2.5.

    Args:
        config: Pipeline configuration

    Returns:
        Set of 'Country/SiteNumber' strings

    Raises:
        FileNotFoundError: If PM2.5 data file doesn't exist
    """
    pm25_file = config.get_pollutant_file('PM2.5')

    if not pm25_file.exists():
        raise FileNotFoundError(
            f"PM2.5 data not found: {pm25_file}\n"
            f"Run download_pollutants.py first."
        )

    print(f"Loading PM2.5 data from {pm25_file}...")
    df = pd.read_parquet(pm25_file)
    print(f"Loaded {len(df):,} rows")

    # Extract Country and SiteNumber
    df['Country'] = df['Samplingpoint'].str.split('/').str[0]
    df['SiteNumber'] = df['Samplingpoint'].apply(extract_site_number)
    df['Site'] = df['Country'] + '/' + df['SiteNumber'].astype(str)

    # Get unique sites
    sites_with_pm25 = set(df['Site'].unique())

    print(f"\nFound {len(sites_with_pm25)} unique sites with PM2.5 measurements")

    return sites_with_pm25


def save_site_list(sites: set, config: PipelineConfig):
    """
    Save list of PM2.5 sites to text file.

    Args:
        sites: Set of site identifiers
        config: Pipeline configuration
    """
    output_file = config.data_dir / "pm25_sites.txt"

    with open(output_file, 'w') as f:
        for site in sorted(sites):
            f.write(f"{site}\n")

    print(f"\n✓ Saved site list to {output_file}")


def filter_pollutant_data(
    pollutant_name: str,
    sites: set,
    config: PipelineConfig
) -> int:
    """
    Filter a pollutant file to only PM2.5 sites.

    Args:
        pollutant_name: Name of pollutant (e.g., 'PM2.5', 'NO2')
        sites: Set of PM2.5 site identifiers
        config: Pipeline configuration

    Returns:
        Number of rows in filtered data
    """
    input_file = config.get_pollutant_file(pollutant_name)

    if not input_file.exists():
        print(f"⚠ Skipping {pollutant_name} - file not found: {input_file}")
        return 0

    print(f"\nFiltering {pollutant_name}...")
    df = pd.read_parquet(input_file)

    # Extract site information
    df['Country'] = df['Samplingpoint'].str.split('/').str[0]
    df['SiteNumber'] = df['Samplingpoint'].apply(extract_site_number)
    df['Site'] = df['Country'] + '/' + df['SiteNumber'].astype(str)

    # Filter to PM2.5 sites
    df_filtered = df[df['Site'].isin(sites)].copy()

    # Drop temporary columns
    df_filtered = df_filtered.drop(columns=['Country', 'SiteNumber', 'Site'])

    # Save filtered version
    output_file = config.get_filtered_pollutant_file(pollutant_name)
    df_filtered.to_parquet(output_file, index=False)

    print(f"  Before: {len(df):,} rows")
    print(f"  After: {len(df_filtered):,} rows")
    print(f"  Saved to: {output_file}")

    return len(df_filtered)


def main():
    """Main filter orchestrator."""
    try:
        config = PipelineConfig()
    except (FileNotFoundError, ValueError) as e:
        print(f"Configuration error: {e}")
        return

    print("=" * 100)
    print("FILTERING SITES TO THOSE WITH PM2.5 MEASUREMENTS")
    print("=" * 100)

    # Step 1: Get PM2.5 sites
    try:
        sites_with_pm25 = get_pm25_sites(config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Step 2: Save site list
    save_site_list(sites_with_pm25, config)

    # Step 3: Filter all pollutant files
    print("\n" + "=" * 100)
    print("FILTERING POLLUTANT FILES")
    print("=" * 100)

    for pollutant in config.pollutants:
        filter_pollutant_data(pollutant['name'], sites_with_pm25, config)

    print("\n" + "=" * 100)
    print("✓ FILTERING COMPLETE")
    print("=" * 100)
    print(f"Sites with PM2.5: {len(sites_with_pm25)}")
    print(f"Site list: {config.data_dir / 'pm25_sites.txt'}")


if __name__ == "__main__":
    main()
