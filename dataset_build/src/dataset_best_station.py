import pandas as pd

import sys
from pathlib import Path


# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'utils'))

from config_utils import PipelineConfig

COVERAGE_CSV = "station_train_test_coverage.csv"
DATASET_CSV = "ml_ready_dataset_v6.csv"

OUT_STATIONS = "best_station.csv"
OUT_DATASET = "ml_ready_dataset_best_station.csv"
MIN_TRAIN = 0.80
MIN_TEST = 0.80
MAX_GAP = 72  # set to None to skip max_gap filtering


def main():
    try:
        config = PipelineConfig()
    except (FileNotFoundError, ValueError) as e:
        print(f"Configuration error: {e}")
        return

    cov = pd.read_csv(config.output_dir / COVERAGE_CSV)

    filt = (
        (cov["train_completeness"] >= MIN_TRAIN)
        & (cov["test_completeness"] >= MIN_TEST)
        & (cov["test_observed_hours"] > 0)
    )
    if MAX_GAP is not None:
        filt = filt & (cov["test_max_gap_hours"] <= MAX_GAP)

    sel = cov.loc[filt].copy()

    print("\nStations per country:")
    print(sel.groupby("Country")["SiteNumber"].nunique())

    sel[
        [
            "Country",
            "SiteNumber",
            "train_completeness",
            "test_completeness",
            "test_max_gap_hours",
        ]
    ].to_csv(OUT_STATIONS, index=False)

    df = pd.read_csv(DATASET_CSV)
    df["SiteNumber"] = df["SiteNumber"].astype(str)

    df_f = df.merge(sel[["Country", "SiteNumber"]], on=["Country", "SiteNumber"], how="inner")
    df_f.to_csv(OUT_DATASET, index=False)


    print("Selected stations:", sel[["Country", "SiteNumber"]].drop_duplicates().shape[0])
    print("Final dataset shape:", df_f.shape)


if __name__ == "__main__":
    main()
