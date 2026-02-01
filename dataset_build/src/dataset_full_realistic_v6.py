import pandas as pd

import sys
from pathlib import Path


# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'utils'))

from config_utils import PipelineConfig

COVERAGE_CSV = "station_train_test_coverage.csv"
DATASET_CSV = "ml_ready_dataset_v6.csv"

OUT_STATIONS = "stations_full_realistic.csv"
OUT_DATASET = "ml_ready_dataset_full_realistic.csv"
MIN_TRAIN = 0.50
MIN_TEST = 0.50
MAX_GAP = 168  # set to None to skip max_gap filtering


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
    ].to_csv(config.output_dir / OUT_STATIONS, index=False)

    df = pd.read_csv(DATASET_CSV)
    df["SiteNumber"] = df["SiteNumber"].astype(str)

    df_f = df.merge(sel[["Country", "SiteNumber"]], on=["Country", "SiteNumber"], how="inner")
    df_f.to_csv(OUT_DATASET, index=False)

    summary_path = config.output_dir / "ml_dataset_realistic_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("# ML Dataset Summary\n\n")
        f.write(f"Total rows: {len(df_f):,}\n")
        f.write(f"Total columns: {len(df_f.columns)}\n")
        f.write(f"Date range: {df_f['Start'].min()} to {df_f['Start'].max()}\n")
        f.write(f"Countries: {', '.join(sorted(df_f['Country'].unique()))}\n")
        f.write(f"Sites: {df_f['SiteNumber'].nunique()}\n\n")

        f.write("## Missing Values\n")
        missing = df_f.isnull().sum()
        missing_present = False
        for col in missing[missing > 0].index:
            pct = (missing[col] / len(df_f)) * 100
            f.write(f"{col}: {missing[col]:,} ({pct:.2f}%)\n")
            missing_present = True

        if not missing_present:
            f.write("No missing values!\n")

    print(f"Saved dataset summary to {summary_path}")

    print("Selected stations:", sel[["Country", "SiteNumber"]].drop_duplicates().shape[0])
    print("Final dataset shape:", df_f.shape)


if __name__ == "__main__":
    main()
