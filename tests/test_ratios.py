"""Tests for diagnostic ratios and wind-vector math (edge cases included)."""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_pipeline import wind_dir_to_uv
from src.ingestion.fire_loader import backtrack_parcel, haversine_km
from src.ingestion.weather_loader import wind_to_uv


def test_pm_ratio_zero_denominator_is_nan_not_crash():
    df = pd.DataFrame({"pm25": [100.0, 50.0], "pm10": [0.0, 200.0]})
    df["pm_ratio"] = df["pm25"] / df["pm10"].replace(0, np.nan)
    assert np.isnan(df["pm_ratio"].iloc[0])
    assert df["pm_ratio"].iloc[1] == pytest.approx(0.25)


def test_no2_so2_ratio_zero_denominator_is_nan():
    df = pd.DataFrame({"no2": [40.0], "so2": [0.0]})
    ratio = df["no2"] / df["so2"].replace(0, np.nan)
    assert np.isnan(ratio.iloc[0])


def test_pm_ratio_thresholds_documented_values():
    # >0.65 combustion, <0.40 crustal - sanity check on typical real values
    df = pd.DataFrame({"pm25": [130.0, 40.0], "pm10": [200.0, 200.0]})
    r = df["pm25"] / df["pm10"]
    assert r.iloc[0] == pytest.approx(0.65)
    assert r.iloc[1] == pytest.approx(0.20)


def test_wind_dir_to_uv_cardinal_directions():
    speed = pd.Series([10.0, 10.0, 10.0, 10.0])
    direction = pd.Series([0.0, 90.0, 180.0, 270.0])  # from N, E, S, W
    u, v = wind_dir_to_uv(speed, direction)
    # From North -> blows toward South: v negative, u ~ 0
    assert u.iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert v.iloc[0] == pytest.approx(-10.0)
    # From East -> blows toward West: u negative, v ~ 0
    assert u.iloc[1] == pytest.approx(-10.0)
    assert v.iloc[1] == pytest.approx(0.0, abs=1e-9)
    # From South -> blows toward North: v positive
    assert v.iloc[2] == pytest.approx(10.0)
    # From West -> blows toward East: u positive
    assert u.iloc[3] == pytest.approx(10.0)


def test_wind_to_uv_loader_matches_pipeline_convention():
    u, v = wind_to_uv(pd.Series([5.0]), pd.Series([0.0]))
    assert float(u.iloc[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(v.iloc[0]) == pytest.approx(-5.0)


def test_haversine_known_distance():
    # Delhi ~ 25 km roughly between two well known points; use exact reference:
    # distance between (28.61, 77.20) and (28.61, 77.21) at this latitude
    d = haversine_km(28.61, 77.20, 28.61, 77.21)
    assert 0.5 < d < 1.5  # ~0.98 km


def test_backtrack_parcel_moves_upwind():
    # Wind blowing toward East (u=+10) at parcel origin: stepping BACK in time
    # must move the parcel origin WEST (smaller longitude).
    u = pd.Series([10.0] * 5)
    v = pd.Series([0.0] * 5)
    parcel = backtrack_parcel(28.6, 77.2, u, v, hours=3)
    assert len(parcel) == 3
    assert parcel["lon"].iloc[0] == pytest.approx(77.2)
    assert parcel["lon"].iloc[2] < parcel["lon"].iloc[0]
