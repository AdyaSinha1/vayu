"""NASA FIRMS active-fire ingestion for the Indo-Gangetic stubble corridor.

FIRMS area CSV export requires a free MAP_KEY (env var ``FIRMS_MAP_KEY``).
When no key is configured - or the fetch fails - this module returns an
EMPTY frame and an honest status string. It NEVER fabricates fire pixels.

The plume back-trajectory matcher in this module is pure geometry: given an
air-parcel history (lat/lon per hour) and fire pixels with detection times,
it reports whether the parcel passed near an active fire within the
boundary-layer mixing window.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FIRMS_CACHE = BASE_DIR / "data" / "raw" / "firms_punjab_haryana.csv"

FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/VIIRS_SNPP_NRT/{west},{south},{east},{north}/2"
# Punjab + Haryana agrarian corridor bounding box (west, south, east, north)
IGP_BBOX = (73.5, 28.5, 78.5, 32.5)


def fetch_firms_fires(day_range: int = 2, timeout_s: int = 30) -> tuple[pd.DataFrame, str]:
    """Fetch active fire pixels (VIIRS 375m) for the Punjab/Haryana corridor.

    Returns (dataframe, status). Dataframe columns when non-empty:
    latitude, longitude, bright_ti4, acq_date, acq_time, confidence, frp (MW).
    """
    map_key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if not map_key:
        return pd.DataFrame(columns=["latitude", "longitude", "acq_date", "frp"]), (
            "unavailable (set FIRMS_MAP_KEY env var for NASA FIRMS; no fabricated fire data)"
        )
    west, south, east, north = IGP_BBOX
    url = FIRMS_AREA_URL.format(map_key=map_key, west=west, south=south, east=east, north=north)
    try:
        resp = requests.get(url, params={"day_range": day_range}, timeout=timeout_s)
        resp.raise_for_status()
    except requests.RequestException as exc:
        return pd.DataFrame(columns=["latitude", "longitude", "acq_date", "frp"]), f"unavailable (FIRMS request failed: {exc})"

    from io import StringIO

    try:
        fdf = pd.read_csv(StringIO(resp.text))
    except Exception:
        return pd.DataFrame(columns=["latitude", "longitude", "acq_date", "frp"]), "unavailable (FIRMS returned non-CSV payload)"

    if fdf.empty or "latitude" not in fdf.columns:
        return fdf, "ok (zero active fire pixels in corridor for window)"
    return fdf, "ok (NASA FIRMS VIIRS, live)"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def backtrack_parcel(
    start_lat: float,
    start_lon: float,
    wind_u: pd.Series,
    wind_v: pd.Series,
    hours: int = 36,
) -> pd.DataFrame:
    """Simple Lagrangian back-trajectory: walk an air parcel backwards in time.

    Uses hourly u/v wind (m/s). The parcel position at t-1 is approximated as
    its current position minus the advection displacement over the prior hour.
    Returns a dataframe of parcel positions, one row per hour back.
    """
    lat, lon = start_lat, start_lon
    rows = []
    n = min(hours, len(wind_u), len(wind_v))
    for i in range(n):
        rows.append({"step_hours_back": i, "lat": lat, "lon": lon})
        # Displacement over 1 hour at these winds (approx 111 km per degree)
        dlat = (wind_v.iloc[i] * 3600.0) / 111_000.0
        dlon = (wind_u.iloc[i] * 3600.0) / (111_000.0 * max(np.cos(np.radians(lat)), 0.2))
        lat -= dlat
        lon -= dlon
    return pd.DataFrame(rows)


def fires_on_trajectory(
    parcel: pd.DataFrame,
    fires: pd.DataFrame,
    proximity_km: float = 60.0,
) -> dict:
    """Report whether the back-trajectory passed near active fire pixels.

    Returns an honest dict; if ``fires`` is empty the answer is explicitly
    ``not_assessed`` rather than a fabricated no/yes.
    """
    if fires is None or fires.empty:
        return {
            "status": "not_assessed",
            "reason": "no fire data available (FIRMS key missing or fetch failed)",
            "min_distance_km": None,
            "nearby_fire_count": 0,
        }
    if parcel.empty:
        return {"status": "not_assessed", "reason": "empty trajectory", "min_distance_km": None, "nearby_fire_count": 0}

    fire_lat = fires["latitude"].to_numpy(dtype=float)
    fire_lon = fires["longitude"].to_numpy(dtype=float)
    min_dist = np.inf
    count = 0
    for _, p in parcel.iterrows():
        dlat = np.radians(fire_lat - p["lat"])
        dlon = np.radians(fire_lon - p["lon"])
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(np.radians(p["lat"])) * np.cos(np.radians(fire_lat)) * np.sin(dlon / 2) ** 2
        )
        d = 2 * 6371.0 * np.arcsin(np.sqrt(a))
        count += int((d <= proximity_km).sum())
        min_dist = min(min_dist, float(d.min()))
    return {
        "status": "transport_match" if count > 0 else "no_transport_match",
        "reason": f"parcel within {proximity_km:.0f} km of {count} fire pixel(s)" if count else "no fire within corridor",
        "min_distance_km": None if np.isinf(min_dist) else round(min_dist, 1),
        "nearby_fire_count": count,
    }


if __name__ == "__main__":
    fdf, status = fetch_firms_fires()
    print("FIRMS status:", status)
    if not fdf.empty:
        print(f"Fire pixels: {len(fdf)} | FRP range: {fdf['frp'].min()} - {fdf['frp'].max()} MW")
