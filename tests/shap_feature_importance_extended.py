import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path handling: work whether launched from repo root or tests/
# ---------------------------------------------------------------------------
THIS = Path(__file__).resolve()
REPO = THIS.parent.parent  # .../eea-pm25
sys.path.insert(0, str(REPO / "train_models"))

CSV_PATH = str(REPO / "ml_ready_dataset_full_realistic.csv")
OUTPUT_DIR = str(REPO / "results")
FIGURES_DIR = str(REPO / "figures")
MODELS_INDOMAIN = REPO / "models" / "models_lgb"
MODELS_LEAVE_FR = REPO / "models" / "models_lgb_exclude_FR"

HORIZONS = [1, 3, 6, 12, 24]
SAMPLE_SIZE = 3000           # global SHAP sample per horizon (group-level shares stable at 3k)
FR_SAMPLE_SIZE = 3000        # France-only sample for the leave-FR comparison
SEED = 42
EXCLUDE_COUNTRY = "FR"

TEST_START = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
TEST_END = pd.Timestamp("2024-12-31 23:00:00", tz="UTC")
SEASON_NAMES = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

from dl_sampling_utils import (
    build_multi_horizon_targets_like_lstm,
    normalize_season_column,
)


# ---------------------------------------------------------------------------
# Feature groups (identical to shap_feature_importance.py)
# ---------------------------------------------------------------------------
def define_feature_groups() -> Dict[str, List[str]]:
    return {
        "Meteorological": [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "wind_u", "wind_v", "precipitation", "surface_pressure",
        ],
        "PM2.5 History": [
            "PM2.5_lag_1h", "PM2.5_lag_2h", "PM2.5_lag_3h", "PM2.5_lag_6h",
            "PM2.5_lag_12h", "PM2.5_lag_24h", "PM2.5_lag_168h",
            "PM2.5_rolling_mean_3h", "PM2.5_rolling_mean_6h",
            "PM2.5_rolling_mean_12h", "PM2.5_rolling_mean_24h",
            "PM2.5_rolling_std_3h", "PM2.5_rolling_std_6h",
            "PM2.5_rolling_std_12h", "PM2.5_rolling_std_24h", "PM2.5",
        ],
        "Other Pollutants": [
            "NO2", "NO2_lag_1h", "NO2_lag_2h", "NO2_lag_3h", "NO2_lag_6h",
            "NO2_lag_12h", "NO2_lag_24h", "NO2_lag_168h",
            "NO2_rolling_mean_3h", "NO2_rolling_mean_6h",
            "NO2_rolling_mean_12h", "NO2_rolling_mean_24h",
            "NO2_rolling_std_3h", "NO2_rolling_std_6h",
            "NO2_rolling_std_12h", "NO2_rolling_std_24h",
            "PM10", "PM10_lag_1h", "PM10_lag_2h", "PM10_lag_3h", "PM10_lag_6h",
            "PM10_lag_12h", "PM10_lag_24h", "PM10_lag_168h",
            "PM10_rolling_mean_3h", "PM10_rolling_mean_6h",
            "PM10_rolling_mean_12h", "PM10_rolling_mean_24h",
            "PM10_rolling_std_3h", "PM10_rolling_std_6h",
            "PM10_rolling_std_12h", "PM10_rolling_std_24h",
        ],
        "Temporal": [
            "hour", "hour_sin", "hour_cos", "day_of_week", "day_of_month",
            "month", "month_sin", "month_cos", "year", "is_weekend", "season",
        ],
        "Metadata": [
            "Latitude", "Longitude", "Altitude",
            "StationType_background", "StationType_industrial", "StationType_traffic",
            "StationArea_rural", "StationArea_rural-nearcity",
            "StationArea_suburban", "StationArea_urban", "station_encoded",
        ],
    }


def assign_features_to_groups(feature_cols, groups) -> Dict[str, str]:
    f2g = {}
    for feature in feature_cols:
        placed = False
        for gname, gfeats in groups.items():
            if feature in gfeats:
                f2g[feature] = gname
                placed = True
                break
        if not placed:
            if feature.startswith("StationType_") or feature.startswith("StationArea_"):
                f2g[feature] = "Metadata"
            elif "PM2.5" in feature:
                f2g[feature] = "PM2.5 History"
            elif "NO2" in feature or "PM10" in feature:
                f2g[feature] = "Other Pollutants"
            else:
                f2g[feature] = "Other"
    return f2g


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """Identical selection logic to shap_feature_importance.py / the ML training scripts."""
    exclude = {
        "Start", "dt_utc", "dt_local", "Country", "SiteNumber",
        "PM2.5", "NO2", "PM10", "season", "PM2.5_next",
    }
    candidates = []
    for c in df.columns:
        if c in exclude or c.startswith("y_h"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            candidates.append(c)
    for c in ["NO2", "PM10"]:
        if c in df.columns:
            candidates.append(c)
    return list(dict.fromkeys(candidates))


# ---------------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------------
GROUP_ORDER = ["PM2.5 History", "Other Pollutants", "Meteorological", "Temporal", "Metadata"]


def _is_generic(names) -> bool:
    """LightGBM trained on a numpy array names features 'Column_0'..'Column_N' — useless here."""
    import re
    return len(names) > 0 and all(re.fullmatch(r"Column_\d+", str(n)) for n in names)


def model_feature_order(model, fallback: List[str]) -> List[str]:
    """Return the feature order the model was trained with, if recoverable AND meaningful.
    The LightGBM models here were fit on numpy arrays in `fallback` order, so their stored
    names are the generic 'Column_N' placeholders; in that case we keep `fallback`."""
    for attr in ("feature_name_", "feature_names_in_"):
        names = getattr(model, attr, None)
        if names is not None and len(names) == len(fallback) and not _is_generic(names):
            return list(names)
    booster = getattr(model, "booster_", None)
    if booster is not None:
        try:
            names = booster.feature_name()
            if names and len(names) == len(fallback) and not _is_generic(names):
                return list(names)
        except Exception:
            pass
    return fallback


def tree_shap(model, X: np.ndarray) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[0]
    return np.asarray(sv)


def grouped_importance(abs_shap: np.ndarray, feature_cols: List[str],
                       f2g: Dict[str, str], row_mask: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Mean (over rows) of summed |SHAP| per group, plus normalized share (%)."""
    if row_mask is not None:
        abs_shap = abs_shap[row_mask]
    if abs_shap.shape[0] == 0:
        return pd.DataFrame(columns=["group", "importance_mean", "share_pct", "n"])
    rows = []
    total = 0.0
    for gname in set(f2g.values()):
        idx = [i for i, f in enumerate(feature_cols) if f2g.get(f) == gname]
        if not idx:
            continue
        val = float(abs_shap[:, idx].sum(axis=1).mean())
        rows.append({"group": gname, "importance_mean": val})
        total += val
    out = pd.DataFrame(rows)
    out["share_pct"] = 100.0 * out["importance_mean"] / (total if total > 0 else 1.0)
    out["n"] = abs_shap.shape[0]
    return out.sort_values("importance_mean", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_test_df() -> pd.DataFrame:
    print(f"Loading dataset: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True)
    df = normalize_season_column(df)
    df = build_multi_horizon_targets_like_lstm(df, HORIZONS)
    test_df = df[(df["dt_utc"] >= TEST_START) & (df["dt_utc"] <= TEST_END)].copy()
    print(f"Test rows: {len(test_df):,}")
    return test_df


def load_lgb(model_dir: Path, horizon: int):
    path = model_dir / f"lgb_h{horizon}.pkl"
    if not path.exists():
        # tolerate the alternate flat layout used by the base script
        alt = REPO / "models" / "lgb" / f"lgb_h{horizon}.pkl"
        path = alt if alt.exists() else path
    if not path.exists():
        raise FileNotFoundError(f"LightGBM model not found: {path}")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------
def main():
    test_df = load_test_df()
    feature_cols = select_feature_columns(test_df)
    f2g = assign_features_to_groups(feature_cols, define_feature_groups())
    rng = np.random.RandomState(SEED)

    # season readable label
    if "season" in test_df.columns:
        test_df["season_label"] = test_df["season"].round().map(SEASON_NAMES).fillna("Unknown")
    else:
        test_df["season_label"] = "Unknown"

    by_horizon, by_country, by_season = [], [], []

    print("\n=== Comment 6: horizon / country / season (in-domain LightGBM) ===")
    for h in HORIZONS:
        sub = test_df.dropna(subset=[f"y_h{h}"]).copy()
        if len(sub) > SAMPLE_SIZE:
            sub = sub.sample(n=SAMPLE_SIZE, random_state=SEED)
        model = load_lgb(MODELS_INDOMAIN, h)
        order = model_feature_order(model, feature_cols)
        X = sub[order].values
        print(f"  h={h}: SHAP on {len(sub):,} samples ...")
        abs_shap = np.abs(tree_shap(model, X))

        # (i) global by horizon
        g = grouped_importance(abs_shap, order, f2g)
        g.insert(0, "horizon", h)
        by_horizon.append(g)

        # (ii) by country
        for country, m in sub.groupby("Country"):
            mask = (sub["Country"] == country).values
            gc = grouped_importance(abs_shap, order, f2g, row_mask=mask)
            if gc.empty:
                continue
            gc.insert(0, "country", str(country))
            gc.insert(0, "horizon", h)
            by_country.append(gc)

        # (iii) by season
        for season, m in sub.groupby("season_label"):
            mask = (sub["season_label"] == season).values
            gs = grouped_importance(abs_shap, order, f2g, row_mask=mask)
            if gs.empty:
                continue
            gs.insert(0, "season", str(season))
            gs.insert(0, "horizon", h)
            by_season.append(gs)

    df_h = pd.concat(by_horizon, ignore_index=True)
    df_c = pd.concat(by_country, ignore_index=True)
    df_s = pd.concat(by_season, ignore_index=True)
    df_h.to_csv(f"{OUTPUT_DIR}/shap_grouped_by_horizon.csv", index=False)
    df_c.to_csv(f"{OUTPUT_DIR}/shap_grouped_by_country.csv", index=False)
    df_s.to_csv(f"{OUTPUT_DIR}/shap_grouped_by_season.csv", index=False)
    print("  saved shap_grouped_by_{horizon,country,season}.csv")

    # --- Leave-FR vs in-domain on France only ------------------------------
    print("\n=== Comment 5: meteorology under spatial transfer (France held out) ===")
    leavefr_rows = []
    have_leavefr = MODELS_LEAVE_FR.exists() or (MODELS_LEAVE_FR / "lgb_h1.pkl").exists()
    if not have_leavefr:
        print(f"  WARNING: leave-FR models not found at {MODELS_LEAVE_FR}; skipping comment-5 block.")
    else:
        fr_all = test_df[test_df["Country"] == EXCLUDE_COUNTRY].copy()
        for h in HORIZONS:
            sub = fr_all.dropna(subset=[f"y_h{h}"]).copy()
            if len(sub) == 0:
                continue
            if len(sub) > FR_SAMPLE_SIZE:
                sub = sub.sample(n=FR_SAMPLE_SIZE, random_state=SEED)
            m_in = load_lgb(MODELS_INDOMAIN, h)
            m_fr = load_lgb(MODELS_LEAVE_FR, h)
            o_in = model_feature_order(m_in, feature_cols)
            o_fr = model_feature_order(m_fr, feature_cols)
            print(f"  h={h}: FR SHAP on {len(sub):,} samples (in-domain + leave-FR) ...")
            g_in = grouped_importance(np.abs(tree_shap(m_in, sub[o_in].values)), o_in, f2g)
            g_fr = grouped_importance(np.abs(tree_shap(m_fr, sub[o_fr].values)), o_fr, f2g)
            merged = g_in[["group", "share_pct"]].merge(
                g_fr[["group", "share_pct"]], on="group",
                suffixes=("_indomain", "_leaveFR"))
            merged["delta_share_pct"] = merged["share_pct_leaveFR"] - merged["share_pct_indomain"]
            merged.insert(0, "horizon", h)
            leavefr_rows.append(merged)
        if leavefr_rows:
            df_fr = pd.concat(leavefr_rows, ignore_index=True)
            df_fr.to_csv(f"{OUTPUT_DIR}/shap_leaveFR_vs_indomain_FR.csv", index=False)
            print("  saved shap_leaveFR_vs_indomain_FR.csv")

    # --- Figures ----------------------------------------------------------
    _plot_horizon(df_h)
    _plot_heatmap(df_c, "country", f"{FIGURES_DIR}/shap_by_country_heatmap.png",
                  "Meteorological share (%) by country and horizon")
    _plot_heatmap(df_s, "season", f"{FIGURES_DIR}/shap_by_season_heatmap.png",
                  "Meteorological share (%) by season and horizon")
    if leavefr_rows:
        _plot_leavefr(df_fr)

    _print_placeholder_summary(df_h, df_c, df_s, leavefr_rows and df_fr)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _plot_horizon(df_h: pd.DataFrame):
    piv = df_h.pivot(index="horizon", columns="group", values="share_pct").reindex(HORIZONS)
    cols = [g for g in GROUP_ORDER if g in piv.columns]
    ax = piv[cols].plot(kind="bar", stacked=True, figsize=(9, 5), edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("Share of total |SHAP| (%)")
    ax.set_title("Grouped feature importance by horizon (LightGBM, Protocol A test)",
                 fontweight="bold")
    ax.legend(title="Feature group", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/shap_grouped_by_horizon.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  saved figures/shap_grouped_by_horizon.png")


def _plot_heatmap(df: pd.DataFrame, dim: str, path: str, title: str):
    met = df[df["group"] == "Meteorological"]
    if met.empty:
        return
    piv = met.pivot_table(index=dim, columns="horizon", values="share_pct")
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(piv) + 2))
    im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_title(title, fontweight="bold")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Meteorological share (%)")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


def _plot_leavefr(df_fr: pd.DataFrame):
    met = df_fr[df_fr["group"].isin(["Meteorological", "Temporal", "Metadata"])]
    fig, ax = plt.subplots(figsize=(9, 5))
    for grp, sub in met.groupby("group"):
        sub = sub.sort_values("horizon")
        ax.plot(sub["horizon"], sub["delta_share_pct"], marker="o", label=grp)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("Δ share (leave-FR − in-domain, %)")
    ax.set_title("France held out: change in feature-group reliance (LightGBM, FR test)",
                 fontweight="bold")
    ax.set_xticks(HORIZONS)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/shap_leaveFR_meteorology.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  saved figures/shap_leaveFR_meteorology.png")


# ---------------------------------------------------------------------------
# Console summary of key numbers
# ---------------------------------------------------------------------------
def _share(df, horizon, group):
    m = (df["horizon"] == horizon) & (df["group"] == group)
    return float(df.loc[m, "share_pct"].iloc[0]) if m.any() else float("nan")


def _mean_imp(df, horizon, group):
    m = (df["horizon"] == horizon) & (df["group"] == group)
    return float(df.loc[m, "importance_mean"].iloc[0]) if m.any() else float("nan")


def _print_placeholder_summary(df_h, df_c, df_s, df_fr):
    print("\n" + "=" * 78)
    print("PLACEHOLDER FILL  ->  paste into revision/REVISION_TEXT.md")
    print("=" * 78)
    print("[§6.1.4]")
    print(f"  PM2.5 History mean|SHAP|  h=1 : {_mean_imp(df_h,1,'PM2.5 History'):.3f}"
          f"   (share {_share(df_h,1,'PM2.5 History'):.0f}%)")
    print(f"  PM2.5 History mean|SHAP|  h=24: {_mean_imp(df_h,24,'PM2.5 History'):.3f}"
          f"   (share {_share(df_h,24,'PM2.5 History'):.0f}%)")
    print(f"  Meteorological share      h=1 -> h=24: "
          f"{_share(df_h,1,'Meteorological'):.0f}% -> {_share(df_h,24,'Meteorological'):.0f}%")
    print(f"  Temporal share            h=1 -> h=24: "
          f"{_share(df_h,1,'Temporal'):.0f}% -> {_share(df_h,24,'Temporal'):.0f}%")
    # country: highest meteorology reliance at h=24
    met_c = df_c[(df_c["group"] == "Meteorological")]
    if not met_c.empty:
        top = met_c[met_c["horizon"] == 24].sort_values("share_pct", ascending=False)
        if not top.empty:
            print("  Country meteorology share @h=24 (high->low): "
                  + ", ".join(f"{r.country} {r.share_pct:.0f}%" for r in top.itertuples()))
    # season
    met_s = df_s[(df_s["group"] == "Meteorological") & (df_s["horizon"] == 24)]
    if not met_s.empty:
        print("  Season meteorology share @h=24: "
              + ", ".join(f"{r.season} {r.share_pct:.0f}%" for r in met_s.itertuples()))
    if df_fr is not None and len(df_fr):
        print("\n[§6.4.2 leave-FR meteorology]")
        for grp in ["Meteorological", "Temporal", "Metadata"]:
            sub = df_fr[df_fr["group"] == grp].sort_values("horizon")
            if sub.empty:
                continue
            deltas = ", ".join(f"h{int(r.horizon)} {r.delta_share_pct:+.1f}pp" for r in sub.itertuples())
            print(f"  {grp:14s} Δshare (leave-FR − in-domain on FR): {deltas}")
    print("=" * 78)


if __name__ == "__main__":
    main()
