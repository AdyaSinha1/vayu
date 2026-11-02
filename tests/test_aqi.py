"""Tests for CPCB AQI sub-index computation."""

import pytest

from src.features.aqi import aqi_category, overall_aqi, sub_index


def test_pm25_breakpoints_known_values():
    assert sub_index("pm25", 0) == 0
    assert sub_index("pm25", 30) == 50
    assert sub_index("pm25", 60) == 100
    assert sub_index("pm25", 167.0) == pytest.approx(336.8, abs=0.1)  # between 120-250
    # segment 250-500 maps to 401-500: 400 -> 401 + 99/250*150 = 460.4
    assert sub_index("pm25", 400) == pytest.approx(460.4, abs=0.1)


def test_interpolation_is_linear_within_segment():
    # segment 30-60 maps to 51-100: midpoint 45 -> 75.5
    assert sub_index("pm25", 45) == pytest.approx(75.5, abs=0.1)


def test_above_highest_breakpoint_caps_at_500():
    assert sub_index("pm25", 9999) == 500


def test_unknown_pollutant_and_invalid_values_return_none():
    assert sub_index("no_x", 10) is None
    assert sub_index("pm25", -5) is None


def test_overall_aqi_takes_max_and_reports_driver():
    aqi, driver, subs = overall_aqi({"pm25": 167.0, "no2": 40.0, "so2": 10.0})
    assert aqi == pytest.approx(336.8, abs=0.1)
    assert driver == "PM2.5"
    assert subs["no2"] == 50.0


def test_overall_aqi_all_none_returns_none():
    aqi, driver, subs = overall_aqi({"pm25": None, "no2": None})
    assert aqi is None and driver is None


def test_categories_cover_full_scale():
    assert aqi_category(0)[0] == "Good"
    assert aqi_category(180)[0] == "Moderate"
    assert aqi_category(350)[0] == "Very Poor"
    assert aqi_category(999)[0] == "Severe"
    assert aqi_category(None) == (None, None)


def test_co_units_are_mg_m3():
    assert sub_index("co", 1.0) == 50
    # segment 1-2 maps to 51-100: 1.6 -> 51 + 49*0.6 = 80.4
    assert sub_index("co", 1.6) == pytest.approx(80.4, abs=0.2)
