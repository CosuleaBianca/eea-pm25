import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

LOOKBACK_H = 168
GAP_BREAK_HOURS = 2


def make_station_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SiteNumber"] = df["SiteNumber"].astype(str)
    df["station_key"] = df["Country"].astype(str) + "_" + df["SiteNumber"].astype(str)
    return df


def recover_station_area_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    area_cols = [c for c in df.columns if c.startswith("StationArea_")]
    if not area_cols:
        df["StationArea_Label"] = "Unknown"
        return df
    df["StationArea_Label"] = df[area_cols].idxmax(axis=1).apply(lambda x: x.replace("StationArea_", ""))
    return df


def normalize_season_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Map season strings to ints whenever the column is non-numeric. (pandas>=3 reads text
    # columns as the 'str'/arrow dtype rather than 'object', so a `== object` check misses them
    # and the later float cast on e.g. 'Winter' fails.)
    if "season" in df.columns and not pd.api.types.is_numeric_dtype(df["season"]):
        season_map = {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3, "Fall": 3}
        df["season"] = df["season"].map(season_map)
    if "season" in df.columns:
        df["season"] = df["season"].fillna(-1).astype("float32")
    return df


def build_multi_horizon_targets_like_lstm(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    df = df.copy()
    df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True, errors="coerce")
    df = df.sort_values(["Country", "SiteNumber", "dt_utc"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["Country", "SiteNumber", "dt_utc"], keep="first")

    df["_row_id"] = np.arange(len(df), dtype=np.int64)
    out_rows = []

    for (country, site), g in df.groupby(["Country", "SiteNumber"], sort=False):
        g = g.sort_values("dt_utc")
        idx_orig = g["dt_utc"]
        row_ids = g["_row_id"].to_numpy()

        g_idx = g.set_index("dt_utc")
        g_hourly = g_idx.asfreq("h")

        targets = {}
        for h in horizons:
            targets[f"y_h{h}"] = g_hourly["PM2.5"].shift(-h)

        targets_df = pd.DataFrame(targets, index=g_hourly.index)
        mapped = targets_df.loc[idx_orig].copy()
        mapped["_row_id"] = row_ids
        out_rows.append(mapped)

    targets_all = pd.concat(out_rows, axis=0, ignore_index=True)
    df = df.merge(targets_all, on="_row_id", how="left").drop(columns=["_row_id"])
    return df


def select_feature_columns_like_lstm_script(df: pd.DataFrame) -> List[str]:
    features_from_file = [
        "Latitude", "Longitude", "Altitude",
        "StationType_background", "StationType_industrial", "StationType_traffic",
        "StationArea_rural", "StationArea_rural-nearcity", "StationArea_suburban", "StationArea_urban",
        "temperature_2m", "relative_humidity_2m", "dew_point_2m", "wind_u", "wind_v",
        "precipitation", "surface_pressure",
        "PM2.5", "NO2", "PM10",
        "hour", "day_of_week", "day_of_month", "month", "year", "is_weekend",
        "season", "hour_sin", "hour_cos", "month_sin", "month_cos",
        "NO2_lag_12h", "NO2_lag_168h", "NO2_lag_1h", "NO2_lag_24h", "NO2_lag_2h", "NO2_lag_3h", "NO2_lag_6h",
        "PM10_lag_12h", "PM10_lag_168h", "PM10_lag_1h", "PM10_lag_24h", "PM10_lag_2h", "PM10_lag_3h", "PM10_lag_6h",
        "PM2.5_lag_12h", "PM2.5_lag_168h", "PM2.5_lag_1h", "PM2.5_lag_24h", "PM2.5_lag_2h", "PM2.5_lag_3h", "PM2.5_lag_6h",
        "NO2_rolling_mean_12h", "NO2_rolling_mean_24h", "NO2_rolling_mean_3h", "NO2_rolling_mean_6h",
        "PM10_rolling_mean_12h", "PM10_rolling_mean_24h", "PM10_rolling_mean_3h", "PM10_rolling_mean_6h",
        "PM2.5_rolling_mean_12h", "PM2.5_rolling_mean_24h", "PM2.5_rolling_mean_3h", "PM2.5_rolling_mean_6h",
        "NO2_rolling_std_12h", "NO2_rolling_std_24h", "NO2_rolling_std_3h", "NO2_rolling_std_6h",
        "PM10_rolling_std_12h", "PM10_rolling_std_24h", "PM10_rolling_std_3h", "PM10_rolling_std_6h",
        "PM2.5_rolling_std_12h", "PM2.5_rolling_std_24h", "PM2.5_rolling_std_3h", "PM2.5_rolling_std_6h",
    ]
    non_feature_cols = {"Country", "SiteNumber", "dt_utc", "dt_local", "Start", "PM2.5_next", "station_key", "StationArea_Label"}
    feature_cols = [c for c in features_from_file if c in df.columns and c not in non_feature_cols]
    if "PM2.5" not in feature_cols:
        raise ValueError("PM2.5 must be present in LSTM feature columns.")
    return feature_cols


def compute_segments(g: pd.DataFrame, gap_break_hours: int = 2) -> List[pd.DataFrame]:
    g = g.sort_values("dt_utc")
    gaps = g["dt_utc"].diff().dt.total_seconds().div(3600)
    breakpoints = np.where(gaps > gap_break_hours)[0]
    if len(breakpoints) == 0:
        return [g]
    segs = []
    start = 0
    for bp in breakpoints:
        segs.append(g.iloc[start:bp].copy())
        start = bp
    segs.append(g.iloc[start:].copy())
    return [s for s in segs if len(s) > 0]


def filter_df_by_anchors(df: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    if anchors is None or anchors.empty:
        return df.iloc[0:0].copy()
    return df.merge(anchors, on=["station_key", "dt_utc"], how="inner")


def build_lstm_aligned_anchors(
    df: pd.DataFrame,
    feature_cols: List[str],
    horizons: List[int],
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    lookback_h: int = LOOKBACK_H,
    gap_break_hours: int = GAP_BREAK_HOURS,
) -> Tuple[Dict[int, pd.DataFrame], Dict[int, pd.DataFrame], List[str], List[str]]:
    df = df.copy()
    df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True, errors="coerce")
    if "station_key" not in df.columns:
        df = make_station_key(df)
    df = df.sort_values(["station_key", "dt_utc"]).drop_duplicates(["station_key", "dt_utc"], keep="first")

    train_st = set(df.loc[df["dt_utc"] <= train_end, "station_key"].unique())
    test_st = set(df.loc[(df["dt_utc"] >= test_start) & (df["dt_utc"] <= test_end), "station_key"].unique())
    common_st = sorted(list(train_st & test_st))
    df = df[df["station_key"].isin(common_st)].copy()

    train_df = df[df["dt_utc"] <= train_end].copy()
    valid_stations = sorted(train_df.dropna(subset=feature_cols)["station_key"].unique())
    valid_station_set = set(valid_stations)

    min_seg_len = lookback_h + max(horizons) + 1

    def collect_anchors(
        df_window: pd.DataFrame,
        anchor_start: Optional[pd.Timestamp],
        anchor_end: Optional[pd.Timestamp],
    ) -> Dict[int, List[Tuple[str, np.datetime64]]]:
        anchors: Dict[int, List[Tuple[str, np.datetime64]]] = {h: [] for h in horizons}
        anchor_start64 = np.datetime64(anchor_start.to_datetime64()) if anchor_start is not None else None
        anchor_end64 = np.datetime64(anchor_end.to_datetime64()) if anchor_end is not None else None

        for station_key, g in df_window.groupby("station_key", sort=False):
            if station_key not in valid_station_set:
                continue
            segs = compute_segments(g, gap_break_hours=gap_break_hours)
            for seg in segs:
                if len(seg) < min_seg_len:
                    continue
                seg = seg.sort_values("dt_utc").reset_index(drop=True)
                x = seg[feature_cols].to_numpy(dtype=np.float32)
                tstamp = seg["dt_utc"].to_numpy(dtype="datetime64[ns]")
                y_by_h = {h: seg[f"y_h{h}"].to_numpy() for h in horizons if f"y_h{h}" in seg.columns}

                n = len(seg)
                start_t = lookback_h - 1

                for h in horizons:
                    y_vals = y_by_h.get(h)
                    if y_vals is None:
                        continue
                    end_t = n - h
                    for t in range(start_t, end_t):
                        ts = tstamp[t]
                        if anchor_start64 is not None and ts < anchor_start64:
                            continue
                        if anchor_end64 is not None and ts > anchor_end64:
                            continue

                        x_win = x[t - lookback_h + 1:t + 1, :]
                        if np.isnan(x_win).any():
                            continue
                        if pd.isna(y_vals[t]):
                            continue

                        anchors[h].append((station_key, ts))

        return anchors

    train_anchors = collect_anchors(train_df, anchor_start=None, anchor_end=train_end)

    warmup = pd.Timedelta(hours=lookback_h - 1)
    test_df = df[(df["dt_utc"] >= (test_start - warmup)) & (df["dt_utc"] <= test_end)].copy()
    test_anchors = collect_anchors(test_df, anchor_start=test_start, anchor_end=test_end)

    train_frames: Dict[int, pd.DataFrame] = {}
    test_frames: Dict[int, pd.DataFrame] = {}

    for h in horizons:
        train_df_h = pd.DataFrame(train_anchors[h], columns=["station_key", "dt_utc"]).drop_duplicates()
        test_df_h = pd.DataFrame(test_anchors[h], columns=["station_key", "dt_utc"]).drop_duplicates()
        if not train_df_h.empty:
            train_df_h["dt_utc"] = pd.to_datetime(train_df_h["dt_utc"], utc=True)
        if not test_df_h.empty:
            test_df_h["dt_utc"] = pd.to_datetime(test_df_h["dt_utc"], utc=True)
        train_frames[h] = train_df_h
        test_frames[h] = test_df_h

    return train_frames, test_frames, common_st, valid_stations
