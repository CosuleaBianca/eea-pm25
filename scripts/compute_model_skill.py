from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("../results")
RESULTS_WITH_SKILL_DIR = Path("../results_with_skill")
PERSISTENCE_FILE = RESULTS_DIR / "persistence_metrics.csv"
PERSISTENCE_DL_FILE = RESULTS_DIR / "persistence_dl_metrics.csv"

KEY_COLS = ["Scope", "Group", "horizon"]
METRIC_COLS = ["MAE", "RMSE"]


def load_metrics(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in KEY_COLS + METRIC_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")
    return df


def is_dl_model(path: Path) -> bool:
    # iTransformer is a Protocol B (sequence) model like the LSTMs, so it must be
    # scored against the persistence_dl baseline, not the full-coverage persistence.
    name = path.name.lower()
    return "_dl_" in name or "lstm" in name or "itransformer" in name


def iter_model_files() -> list:
    files = sorted(RESULTS_DIR.glob("*metrics*.csv"))
    return [
        path for path in files
        if "persistence" not in path.name.lower()
        # skip per-horizon shards (e.g. itransformer_metrics_h1.csv); use the merged file
        and "_metrics_h" not in path.name.lower()
    ]


def add_skill_columns(model_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    baseline_subset = baseline_df[KEY_COLS + METRIC_COLS].copy()
    merged = model_df.merge(
        baseline_subset, on=KEY_COLS, how="left", suffixes=("", "_baseline")
    )
    for metric in METRIC_COLS:
        merged[f"{metric}_skill"] = merged[metric] - merged[f"{metric}_baseline"]
    return merged


def build_summary_df(
    merged: pd.DataFrame, model_path: Path, baseline_label: str
) -> pd.DataFrame:
    summary = merged.copy()
    summary.insert(0, "baseline", baseline_label)
    summary.insert(0, "model", model_path.stem)
    summary.insert(0, "model_file", model_path.name)
    keep_cols = [
        "model_file",
        "model",
        "baseline",
        *KEY_COLS,
        *METRIC_COLS,
        *[f"{metric}_skill" for metric in METRIC_COLS],
    ]
    return summary[keep_cols]


def main() -> None:
    persistence = load_metrics(PERSISTENCE_FILE, PERSISTENCE_FILE.name)
    persistence_dl = load_metrics(PERSISTENCE_DL_FILE, PERSISTENCE_DL_FILE.name)

    RESULTS_WITH_SKILL_DIR.mkdir(exist_ok=True)

    outputs = []
    for path in iter_model_files():
        use_dl = is_dl_model(path)
        baseline_df = persistence_dl if use_dl else persistence
        baseline_label = "persistence_dl" if use_dl else "persistence"
        model_df = load_metrics(path, path.name)
        merged = add_skill_columns(model_df, baseline_df)
        skill_df = build_summary_df(merged, path, baseline_label)

        missing_baseline = skill_df[
            [f"{metric}_skill" for metric in METRIC_COLS]
        ].isna().any(axis=1)
        if missing_baseline.any():
            print(
                f"Warning: {path.name} has {missing_baseline.sum()} rows without "
                f"{baseline_label} match."
            )

        per_file = merged.drop(
            columns=[f"{metric}_baseline" for metric in METRIC_COLS], errors="ignore"
        )
        per_file.to_csv(RESULTS_WITH_SKILL_DIR / path.name, index=False)

        outputs.append(skill_df)

    if not outputs:
        print("No model metrics files found in results/.")
        return

    print(
        f"Wrote {len(outputs)} files to {RESULTS_WITH_SKILL_DIR}/."
    )


if __name__ == "__main__":
    main()
