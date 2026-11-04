"""Feature engineering + strict chronological splitting for VAYU.

Builds the modelling table from 100% real CPCB CAAQMS telemetry:
- Standardises raw CPCB columns to clean names.
- Aggregates all Delhi stations to a city-wide hourly median.
- Uses CPCB's OWN onboard meteorological sensors (AT, RH, WS, WD, BP) as
  baseline met features; enriches with Open-Meteo ERA5 (BLH, u/v) when the
  network allows. No met value is ever fabricated.
- Adds diagnostic chemical ratios, lag features, rolling means and diurnal
  encodings known at forecast time t (strictly no future information).
- Splits chronologically (default 75/25) - the test set is always strictly
  later in time than the training set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CPCB_PARQUET = BASE_DIR / "data" / "processed" / "delhi_cpcb_real.parquet"

# Features that describe the ATMOSPHERE's ability to disperse pollution.
MET_FEATURES = ["temp", "humidity", "wind_speed", "wind_u", "wind_v", "pressure", "blh"]
# Features that describe actual emitted pollutants / their chemical signatures.
EMISSION_FEATURES = ["pm10", "no2", "so2", "co", "ozone", "pm_ratio", "no2_so2_ratio", "co_no2_ratio"]

CORE_POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co", "ozone"]


def clean_cpcb_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise raw CPCB column names to ASCII snake_case."""
    mapping: dict[str, str] = {}
    for col in df.columns:
        c = str(col).lower()
        if "pm2.5" in c:
            mapping[col] = "pm25"
        elif "pm10" in c:
            mapping[col] = "pm10"
        elif c.startswith("no2") or "no2" in c:
            mapping[col] = "no2"
        elif "nox" in c:
            mapping[col] = "nox"
        elif "so2" in c:
            mapping[col] = "so2"
        elif c.startswith("co ") or c == "co":
            mapping[col] = "co"
        elif "ozone" in c or "o3" in c:
            mapping[col] = "ozone"
        elif "station name" in c:
            mapping[col] = "station"
        elif "timestamp" in c:
            mapping[col] = "timestamp"
        elif c.startswith("at ") or "temp" in c:
            mapping[col] = "temp"
        elif c.startswith("rh ") or "humidity" in c:
            mapping[col] = "humidity"
        elif c.startswith("ws ") or c == "wind_speed":
            mapping[col] = "wind_speed"
        elif c.startswith("wd ") or c == "wind_dir":
            mapping[col] = "wind_dir"
        elif c.startswith("bp ") or "pressure" in c:
            mapping[col] = "pressure"
        elif "benzene" in c and "eth" not in c:
            mapping[col] = "benzene"
    return df.rename(columns=mapping)


def hourly_city_aggregate(df: pd.DataFrame, min_stations: int = 3) -> pd.DataFrame:
    """Aggregate all stations to city-wide hourly medians (robust to one bad sensor).

    Records how many stations contributed per hour; hours with fewer than
    ``min_stations`` reporting PM2.5 are dropped rather than interpolated.
    """
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"])
    out["hour"] = out["timestamp"].dt.floor("h")

    agg: dict[str, pd.Series | str] = {}
    for col in CORE_POLLUTANTS + ["temp", "humidity", "wind_speed", "wind_dir", "pressure", "nox", "benzene"]:
        if col in out.columns:
            agg[col] = (col, "median")
    hourly = out.groupby("hour").agg(**agg).reset_index().rename(columns={"hour": "timestamp"})

    if "pm25" in out.columns:
        n_stations = out.dropna(subset=["pm25"]).groupby("hour")["station"].nunique()
        hourly["n_stations"] = hourly["timestamp"].map(n_stations).fillna(0).astype(int)
        hourly = hourly[hourly["n_stations"] >= min_stations]
    return hourly.reset_index(drop=True)


def wind_dir_to_uv(speed: pd.Series, direction_deg: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Meteorological wind direction (from-North) -> u/v components (m/s)."""
    rad = np.radians(pd.to_numeric(direction_deg, errors="coerce"))
    speed = pd.to_numeric(speed, errors="coerce")
    u = -speed * np.sin(rad)
    v = -speed * np.cos(rad)
    return u, v


def fill_short_gaps(hourly: pd.DataFrame, max_gap_hours: int = 2) -> pd.DataFrame:
    """Time-interpolate internal gaps up to ``max_gap_hours``; leave longer gaps NaN."""
    filled = hourly.set_index("timestamp").sort_index()
    numeric = filled.select_dtypes(include=[np.number])
    # Interpolate with a limit; gaps longer than the limit remain NaN.
    interp = numeric.interpolate(method="time", limit=max_gap_hours, limit_area="inside")
    for col in interp.columns:
        filled[col] = interp[col]
    return filled.reset_index()


def build_features(cpcb_df: pd.DataFrame, weather_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Create the full hourly modelling table. Returns (features, meta)."""
    meta: dict = {"weather_provenance": "not used (CPCB onboard sensors only)"}
    cleaned = clean_cpcb_columns(cpcb_df)
    hourly = fill_short_gaps(hourly_city_aggregate(cleaned))

    # CPCB onboard meteorology -> u/v ground wind
    if "wind_speed" in hourly.columns and "wind_dir" in hourly.columns:
        hourly["wind_u"], hourly["wind_v"] = wind_dir_to_uv(hourly["wind_speed"], hourly["wind_dir"])

    # Optional ERA5 enrichment (BLH + reanalysis winds). Merge on hourly UTC time.
    if weather_df is not None and not weather_df.empty:
        w = weather_df.copy()
        w["timestamp"] = pd.to_datetime(w["timestamp"], utc=True, errors="coerce")
        w = w.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        w = w[~w.index.duplicated(keep="first")].reset_index()
        w = w.rename(columns={"temperature_2m": "era5_temp", "boundary_layer_height": "blh"})
        keep = [c for c in ["timestamp", "blh", "era5_temp"] if c in w.columns]
        hourly = hourly.merge(w[keep], on="timestamp", how="left")
        if "blh" in hourly.columns and hourly["blh"].notna().any():
            meta["weather_provenance"] = "CPCB onboard + Open-Meteo ERA5 (BLH)"
        else:
            meta["weather_provenance"] = "CPCB onboard only (ERA5 fetch returned no BLH)"
    else:
        hourly["blh"] = np.nan

    # ---- Diagnostic chemical ratios (receptor-model side features) ----
    if "pm25" in hourly.columns and "pm10" in hourly.columns:
        hourly["pm_ratio"] = hourly["pm25"] / hourly["pm10"].replace(0, np.nan)
    if "no2" in hourly.columns and "so2" in hourly.columns:
        hourly["no2_so2_ratio"] = hourly["no2"] / hourly["so2"].replace(0, np.nan)
    if "co" in hourly.columns and "no2" in hourly.columns:
        hourly["co_no2_ratio"] = (hourly["co"] * 1000.0) / hourly["no2"].replace(0, np.nan)

    # ---- Lag / rolling features (information available at time t) ----
    lag_specs = {"pm25": [1, 3, 6, 24], "no2": [1, 6], "so2": [1, 24]}
    for col, lags in lag_specs.items():
        if col in hourly.columns:
            for lag in lags:
                hourly[f"{col}_lag{lag}"] = hourly[col].shift(lag)
    if "pm25" in hourly.columns:
        hourly["pm25_roll3"] = hourly["pm25"].shift(1).rolling(3, min_periods=1).mean()
        hourly["pm25_roll24"] = hourly["pm25"].shift(1).rolling(24, min_periods=6).mean()

    # ---- Diurnal encodings ----
    hr = hourly["timestamp"].dt.hour
    hourly["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    hourly["hour_cos"] = np.cos(2 * np.pi * hr / 24)

    meta["n_hourly_rows"] = int(len(hourly))
    meta["time_start"] = str(hourly["timestamp"].min())
    meta["time_end"] = str(hourly["timestamp"].max())
    return hourly, meta


MODEL_FEATURES = [
    "pm25", "pm10", "no2", "so2", "co", "ozone",
    "pm_ratio", "no2_so2_ratio", "co_no2_ratio",
    "temp", "humidity", "wind_speed", "wind_u", "wind_v", "pressure", "blh",
    "pm25_lag1", "pm25_lag3", "pm25_lag6", "pm25_lag24",
    "no2_lag1", "no2_lag6", "so2_lag1", "so2_lag24",
    "pm25_roll3", "pm25_roll24",
    "hour_sin", "hour_cos",
]


def make_supervised(features: pd.DataFrame, horizon: int, feature_list: list[str] | None = None) -> pd.DataFrame:
    """Create a supervised frame: X = state at t, y = PM2.5 at t + horizon."""
    feature_list = feature_list or MODEL_FEATURES
    sup = features[["timestamp"] + [f for f in feature_list if f in features.columns]].copy()
    if "pm25" not in features.columns:
        raise ValueError("pm25 column required to build targets")
    sup[f"target_h{horizon}"] = features["pm25"].shift(-horizon)
    return sup.dropna(subset=[f"target_h{horizon}"]).reset_index(drop=True)


def chronological_split(sup: pd.DataFrame, target: str, train_frac: float = 0.75) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split strictly by time: earlier fraction trains, later fraction tests."""
    sup = sup.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(sup) * train_frac)
    train = sup.iloc[:split_idx]
    test = sup.iloc[split_idx:]
    split_time = sup["timestamp"].iloc[split_idx]
    return train, test, split_time


def load_and_build(weather_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Convenience: load CPCB telemetry (local file or auto-bootstrapped), build features."""
    from src.ingestion.cpcb_data import ensure_cpcb_dataframe

    raw, _ = ensure_cpcb_dataframe()
    return build_features(raw, weather_df)


if __name__ == "__main__":
    feats, meta = load_and_build()
    print("Meta:", meta)
    print("Shape:", feats.shape)
    print("Columns:", feats.columns.tolist())
    print(feats[["timestamp", "pm25", "pm_ratio", "no2_so2_ratio", "wind_u", "blh"]].tail(8).round(2).to_string())
