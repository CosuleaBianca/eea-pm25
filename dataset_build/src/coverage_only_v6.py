import pandas as pd
import numpy as np

import sys
from pathlib import Path


# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'utils'))

from config_utils import PipelineConfig


CSV_PATH = "ml_ready_dataset_v6.csv"

# Split temporal (UTC)
TRAIN_END = pd.Timestamp("2022-12-31 23:00:00", tz="UTC")
TEST_START = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
TEST_END = pd.Timestamp("2024-12-31 23:00:00", tz="UTC")

OUT_COVERAGE = "station_train_test_coverage.csv"


def station_window_stats(g: pd.DataFrame) -> dict:
    """Compute observed hours, expected hours, completeness, and max gap for a window."""
    if len(g) == 0:
        return {
            "start": pd.NaT,
            "end": pd.NaT,
            "observed_hours": 0,
            "expected_hours": 0,
            "completeness": np.nan,
            "max_gap_hours": np.nan,
        }

    g = g.sort_values("dt_utc")
    start = g["dt_utc"].min()
    end = g["dt_utc"].max()

    expected_hours = int((end - start).total_seconds() // 3600) + 1
    observed_hours = g["dt_utc"].nunique()

    completeness = observed_hours / expected_hours if expected_hours > 0 else np.nan

    gaps = g["dt_utc"].diff().dt.total_seconds().div(3600)
    max_gap = gaps.max()

    return {
        "start": start,
        "end": end,
        "observed_hours": int(observed_hours),
        "expected_hours": int(expected_hours),
        "completeness": float(completeness),
        "max_gap_hours": float(max_gap) if pd.notna(max_gap) else np.nan,
    }


def main():
    try:
        config = PipelineConfig()
    except (FileNotFoundError, ValueError) as e:
        print(f"Configuration error: {e}")
        return
    
    df = pd.read_csv(CSV_PATH)

    df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True, errors="coerce")

    df = df.sort_values(["Country", "SiteNumber", "dt_utc"])
    df = df.drop_duplicates(subset=["Country", "SiteNumber", "dt_utc"], keep="first")

    rows = []

    for (country, site), g in df.groupby(["Country", "SiteNumber"], sort=False):
        g_train = g[g["dt_utc"] <= TRAIN_END]
        train_stats = station_window_stats(g_train)

        g_test = g[(g["dt_utc"] >= TEST_START) & (g["dt_utc"] <= TEST_END)]
        test_stats = station_window_stats(g_test)

        rows.append(
            {
                "Country": country,
                "SiteNumber": site,
                "train_start": train_stats["start"],
                "train_end": train_stats["end"],
                "train_observed_hours": train_stats["observed_hours"],
                "train_expected_hours": train_stats["expected_hours"],
                "train_completeness": train_stats["completeness"],
                "train_max_gap_hours": train_stats["max_gap_hours"],
                "test_start": test_stats["start"],
                "test_end": test_stats["end"],
                "test_observed_hours": test_stats["observed_hours"],
                "test_expected_hours": test_stats["expected_hours"],
                "test_completeness": test_stats["completeness"],
                "test_max_gap_hours": test_stats["max_gap_hours"],
            }
        )

    cov = pd.DataFrame(rows)
    cov.to_csv(config.output_dir / OUT_COVERAGE, index=False)
    print(f"Saved: {config.output_dir / OUT_COVERAGE}")

    print("\n=== Summary ===")
    print(f"Total stations: {len(cov)}")
    print("Stations per country:")
    print(cov.groupby("Country")["SiteNumber"].nunique())


if __name__ == "__main__":
    main()
