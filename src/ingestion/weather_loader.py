"""Open-Meteo ERA5 archive weather ingestion for VAYU.

Fetches hourly reanalysis weather for Delhi-NCR from the free Open-Meteo
archive API (no API key required), computes u/v wind-vector components and
caches results locally so the app also works offline.

Honesty rules:
- If the network fetch fails and no cache exists, this module returns None
  and reports the failure. It NEVER fabricates weather values.
- Whether data came from the live API or the local cache is always reported.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent.parent
WEATHER_CACHE = BASE_DIR / "data" / "raw" / "delhi_weather_openmeteo.parquet"

DELHI_LAT, DELHI_LON = 28.6139, 77.2090
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "boundary_layer_height",
]


def wind_to_uv(speed: np.ndarray | pd.Series, direction_deg: np.ndarray | pd.Series) -> tuple[pd.Series, pd.Series]:
    """Convert (wind speed, meteorological direction) to u/v vector components.

    Meteorological convention: direction is where wind COMES FROM (0 deg = North).
    u = -speed * sin(dir * pi / 180)  (positive u -> wind blowing toward East)
    v = -speed * cos(dir * pi / 180)  (positive v -> wind blowing toward North)
    """
    rad = np.radians(np.asarray(direction_deg, dtype=float))
    u = -np.asarray(speed, dtype=float) * np.sin(rad)
    v = -np.asarray(speed, dtype=float) * np.cos(rad)
    return pd.Series(u), pd.Series(v)


def fetch_open_meteo_archive(
    start_date: str,
    end_date: str,
    lat: float = DELHI_LAT,
    lon: float = DELHI_LON,
    max_retries: int = 3,
    timeout_s: int = 30,
) -> pd.DataFrame | None:
    """Fetch hourly ERA5 reanalysis from Open-Meteo archive. Returns None on failure."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
    }
    for attempt in range(max_retries):
        try:
            resp = requests.get(ARCHIVE_URL, params=params, timeout=timeout_s)
            if resp.status_code == 200:
                hourly = resp.json().get("hourly", {})
                if not hourly.get("time"):
                    return None
                wdf = pd.DataFrame(hourly)
                wdf["timestamp"] = pd.to_datetime(wdf["time"], utc=True, errors="coerce")
                wdf = wdf.drop(columns=["time"]).dropna(subset=["timestamp"])
                for col in HOURLY_VARS:
                    wdf[col] = pd.to_numeric(wdf[col], errors="coerce")
                u, v = wind_to_uv(wdf["wind_speed_10m"], wdf["wind_direction_10m"])
                wdf["wind_u"] = u.values
                wdf["wind_v"] = v.values
                return wdf.reset_index(drop=True)
            last_status = resp.status_code
        except requests.RequestException:
            last_status = "network-error"
        time.sleep(1.5 * (attempt + 1))
    print(f"[weather_loader] Open-Meteo archive failed after {max_retries} attempts (last status: {last_status}).")
    return None


def load_weather(
    start_date: str,
    end_date: str,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> tuple[pd.DataFrame | None, str]:
    """Load weather with live fetch + local cache. Returns (dataframe | None, provenance).

    Provenance is one of: "open-meteo-archive (live)", "local-cache (offline)",
    or "unavailable (<reason>)". A cache is only used when it actually covers
    the requested date range; otherwise a live fetch is attempted.
    """
    cache_path = WEATHER_CACHE

    def cache_covers() -> bool:
        if not cache_path.exists():
            return False
        c = pd.read_parquet(cache_path)
        if c.empty or "timestamp" not in c.columns:
            return False
        ts = pd.to_datetime(c["timestamp"], utc=True, errors="coerce").dropna()
        req_start = pd.Timestamp(start_date, tz="UTC")
        req_end = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
        return ts.min() <= req_start and ts.max() >= req_end

    if use_cache and not refresh_cache and cache_covers():
        cached = pd.read_parquet(cache_path)
        return cached, "local-cache (offline)"

    wdf = fetch_open_meteo_archive(start_date, end_date)
    if wdf is not None and not wdf.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        wdf.to_parquet(cache_path, index=False)
        return wdf, "open-meteo-archive (live)"

    if use_cache and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        return cached, "local-cache (offline, live fetch failed / partial coverage)"

    return None, "unavailable (Open-Meteo unreachable and no cache present)"


if __name__ == "__main__":
    wdf, provenance = load_weather("2025-12-24", "2026-01-01", refresh_cache=True)
    if wdf is None:
        print(f"Weather: {provenance}")
        print("Pipeline will continue using CPCB onboard met sensors only (no fabricated BLH).")
    else:
        print(f"Weather rows: {len(wdf)} | provenance: {provenance}")
        print(wdf.head(3).to_string())
        print("BLH range:", wdf["boundary_layer_height"].min(), "-", wdf["boundary_layer_height"].max())
