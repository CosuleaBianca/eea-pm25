"""
Download air quality data per-pollutant from EEA API.

This script:
1. Reads configuration from config.yaml
2. Downloads each pollutant separately in 6-month chunks
3. Saves to data/POLLUTANT_NAME.parquet
4. Skips re-downloading if file exists (configurable)
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'utils'))

import requests
import zipfile
import io
import pandas as pd
from config_utils import PipelineConfig


def download_pollutant(
    config: PipelineConfig,
    pollutant: dict,
    force_download: bool = False
) -> bool:
    """
    Download data for a single pollutant.

    Args:
        config: Pipeline configuration
        pollutant: Dict with 'name', 'code', 'uri' keys
        force_download: If True, download even if file exists

    Returns:
        True if successful, False otherwise
    """
    output_file = config.get_pollutant_file(pollutant['name'])

    # Skip if file exists and skip_existing is enabled
    if output_file.exists() and config.skip_existing and not force_download:
        print(f"✓ Skipping {pollutant['name']} - file already exists: {output_file}")
        return True

    print("=" * 100)
    print(f"DOWNLOADING {pollutant['name']} (code: {pollutant['code']})")
    print("=" * 100)

    all_dataframes = []

    for date_start, date_end, period_name in config.date_ranges:
        print(f"\nDownloading {period_name} ({date_start} to {date_end})...")

        payload = {
            "countries": config.countries,
            "cities": config.city_names,
            "pollutants": [pollutant['uri']],  # Single pollutant
            "dataset": config.dataset,
            "source": "Website",
            "dateTimeStart": date_start,
            "dateTimeEnd": date_end,
            "aggregationType": config.aggregation_type,
            "email": None
        }

        try:
            response = requests.post(
                config.api_url + "ParquetFile",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()

            zip_data = response.content
            print(f"Successfully downloaded {len(zip_data)} bytes for {period_name}.")

            if len(zip_data) < 100:
                print(f"WARNING: Response too small ({len(zip_data)} bytes)")
                continue

            # Extract parquet from ZIP
            with zipfile.ZipFile(io.BytesIO(zip_data)) as zip_file:
                parquet_filename = [f for f in zip_file.namelist()
                                   if f.endswith('.parquet')][0]
                parquet_data = zip_file.read(parquet_filename)

                df_period = pd.read_parquet(io.BytesIO(parquet_data))
                all_dataframes.append(df_period)
                print(f"Loaded {len(df_period)} rows from {period_name}.")

        except requests.RequestException as e:
            print(f"Failed to retrieve data for {period_name}: {e}")
            return False
        except (zipfile.BadZipFile, IndexError) as e:
            print(f"Error processing ZIP file for {period_name}: {e}")
            return False

    # Combine and save
    if all_dataframes:
        print(f"\nCombining {len(all_dataframes)} periods...")
        combined_df = pd.concat(all_dataframes, ignore_index=True)

        # Ensure data directory exists
        config.data_dir.mkdir(exist_ok=True)

        combined_df.to_parquet(output_file, index=False)
        print(f"✓ Saved {pollutant['name']} data to {output_file}")
        print(f"  Total rows: {len(combined_df):,}")
        return True
    else:
        print(f"✗ No data downloaded for {pollutant['name']}")
        return False


def main():
    """Main download orchestrator."""
    try:
        config = PipelineConfig()
    except (FileNotFoundError, ValueError) as e:
        print(f"Configuration error: {e}")
        return

    print("=" * 100)
    print("EEA AIR QUALITY DATA DOWNLOADER")
    print("=" * 100)
    print(f"Countries: {', '.join(config.countries)}")
    print(f"Cities: {', '.join(config.city_names)}")
    print(f"Date range: {config.config['date_range']['start']} to "
          f"{config.config['date_range']['end']}")
    print(f"Pollutants: {', '.join(p['name'] for p in config.pollutants)}")
    print(f"Output directory: {config.data_dir}")
    print("=" * 100)

    results = {}
    for pollutant in config.pollutants:
        success = download_pollutant(config, pollutant)
        results[pollutant['name']] = success

    # Summary
    print("\n" + "=" * 100)
    print("DOWNLOAD SUMMARY")
    print("=" * 100)
    for name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {name}")

    successful = sum(results.values())
    print(f"\nCompleted: {successful}/{len(results)} pollutants")


if __name__ == "__main__":
    main()
