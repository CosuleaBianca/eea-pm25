"""
Configuration utilities for EEA Air Quality Pipeline.

This module provides the PipelineConfig class for loading and validating
configuration from config.yaml.
"""
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple


class PipelineConfig:
    """Manages pipeline configuration from YAML file."""

    def __init__(self, config_path: str = None):
        """
        Load and validate configuration.

        Args:
            config_path: Path to YAML configuration file
                        If None, looks for config.yaml in utils/ directory

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        if config_path is None:
            # Default to utils/config.yaml relative to this file
            config_path = Path(__file__).parent / "config.yaml"

        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self._validate()

    def _validate(self):
        """
        Validate required configuration fields.

        Raises:
            ValueError: If required fields are missing or invalid
        """
        required_keys = ['cities', 'date_range', 'pollutants', 'api', 'paths']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        # Validate at least one pollutant is PM2.5
        pm25_found = any(p['code'] == 6001 for p in self.config['pollutants'])
        if not pm25_found:
            raise ValueError(
                "PM2.5 (code 6001) must be included in pollutants list. "
                "PM2.5 is required for site filtering."
            )

        # Validate date range
        try:
            datetime.strptime(self.config['date_range']['start'], '%Y-%m-%d')
            datetime.strptime(self.config['date_range']['end'], '%Y-%m-%d')
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid date_range format: {e}")

    @property
    def cities(self) -> List[Dict[str, str]]:
        """Get list of cities to download."""
        return self.config['cities']

    @property
    def countries(self) -> List[str]:
        """Get unique list of country codes."""
        return list(set(c['country'] for c in self.cities))

    @property
    def city_names(self) -> List[str]:
        """Get list of city names."""
        return [c['city'] for c in self.cities]

    @property
    def pollutants(self) -> List[Dict]:
        """Get list of pollutants to download."""
        return self.config['pollutants']

    @property
    def date_ranges(self) -> List[Tuple[str, str, str]]:
        """
        Generate 6-month date ranges for downloading.

        Returns:
            List of (start_date, end_date, period_name) tuples in ISO format
            Example: [("2018-01-01T00:00:00.000Z", "2018-06-30T23:59:00.000Z", "2018-H1"), ...]
        """
        start = datetime.strptime(self.config['date_range']['start'], '%Y-%m-%d')
        end = datetime.strptime(self.config['date_range']['end'], '%Y-%m-%d')

        ranges = []
        current_year = start.year

        while current_year <= end.year:
            # First half (Jan-Jun)
            h1_start = f"{current_year}-01-01T00:00:00.000Z"
            h1_end = f"{current_year}-06-30T23:59:00.000Z"
            ranges.append((h1_start, h1_end, f"{current_year}-H1"))

            # Second half (Jul-Dec)
            h2_start = f"{current_year}-07-01T00:00:00.000Z"
            h2_end = f"{current_year}-12-31T23:59:00.000Z"
            ranges.append((h2_start, h2_end, f"{current_year}-H2"))

            current_year += 1

        return ranges

    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        return Path(self.config['paths']['data_dir'])

    @property
    def output_dir(self) -> Path:
        """Get output directory path."""
        return Path(self.config['paths']['output_dir'])

    @property
    def processed_file(self) -> str:
        """Get processed output filename."""
        return self.config['paths']['processed_file']

    @property
    def api_url(self) -> str:
        """Get API base URL."""
        return self.config['api']['base_url']

    @property
    def dataset(self) -> int:
        """Get dataset ID."""
        return self.config['api']['dataset']

    @property
    def aggregation_type(self) -> str:
        """Get aggregation type."""
        return self.config['api']['aggregation_type']

    @property
    def skip_existing(self) -> bool:
        """Whether to skip existing downloads."""
        return self.config['processing'].get('skip_existing_downloads', True)

    def get_pollutant_file(self, pollutant_name: str) -> Path:
        """
        Get file path for a specific pollutant.

        Args:
            pollutant_name: Name of pollutant (e.g., 'PM2.5', 'NO2')

        Returns:
            Path to pollutant file (e.g., data/PM2.5.parquet)
        """
        return self.data_dir / f"{pollutant_name}.parquet"

    def get_filtered_pollutant_file(self, pollutant_name: str) -> Path:
        """
        Get file path for a filtered pollutant file.

        Args:
            pollutant_name: Name of pollutant (e.g., 'PM2.5', 'NO2')

        Returns:
            Path to filtered pollutant file (e.g., data/PM2.5_filtered.parquet)
        """
        return self.data_dir / f"{pollutant_name}_filtered.parquet"
