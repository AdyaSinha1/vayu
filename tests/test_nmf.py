"""Tests for the NMF receptor model: constraints, dimensions, naming, bootstrap CIs."""

import numpy as np
import pandas as pd
import pytest

from src.attribution.receptor_nmf import (
    SOURCE_CATALOG,
    _design_matrix,
    _score_profiles,
    bootstrap_source_shares,
    clean_column_names,
    compute_diagnostic_ratios,
    fit_receptor,
)


def _synthetic_cpcb(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Synthetic non-negative multi-pollutant frame with 3 latent sources."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    veh = 40 + 20 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 3, n)  # rush-hour NO2-ish
    ind = 10 + 2 * rng.random(n)
    dust = 30 + 15 * np.sin(2 * np.pi * t / (24 * 3)) + rng.normal(0, 3, n)
    df = pd.DataFrame(
        {
            "pm25": np.clip(veh * 0.8 + ind * 0.5 + dust * 0.4 + rng.normal(0, 2, n), 1, None),
            "pm10": np.clip(veh * 1.1 + ind * 0.6 + dust * 1.8 + rng.normal(0, 3, n), 1, None),
            "no2": np.clip(veh * 1.2 + ind * 0.5 + rng.normal(0, 2, n), 0.1, None),
            "so2": np.clip(ind * 1.5 + rng.normal(0, 0.5, n), 0.1, None),
            "co": np.clip(veh * 0.02 + rng.normal(0, 0.002, n), 0.01, None),
            "ozone": np.clip(50 + 10 * np.cos(2 * np.pi * t / 24) + rng.normal(0, 3, n), 1, None),
        }
    )
    return df


def test_nmf_shapes_and_nonnegativity():
    df = _synthetic_cpcb()
    X, feats = _design_matrix(df)
    assert X.shape[1] == len(feats) == 6
    assert (X >= 0).all()
    res = fit_receptor(df, n_sources=4, random_state=0)
    assert res["profiles"].shape == (4, 6)
    assert (res["profiles"].to_numpy() >= 0).all()
    assert res["n_samples"] == len(X)


def test_nmf_shares_are_percentages_summing_to_100():
    res = fit_receptor(_synthetic_cpcb(), random_state=1)
    total = sum(res["shares"].values())
    assert total == pytest.approx(100.0, abs=1e-6)


def test_source_names_are_unique_and_from_catalog():
    res = fit_receptor(_synthetic_cpcb(), random_state=2)
    names = res["names"]
    assert len(names) == 4
    assert len(set(names)) == 4
    for n in names:
        assert n in SOURCE_CATALOG


def test_heuristic_scores_assign_all_four_sources():
    H_norm = np.array(
        [
            [0.2, 0.3, 0.45, 0.02, 0.01, 0.02],  # NO2 dominant -> vehicular-ish
            [0.3, 0.45, 0.1, 0.05, 0.02, 0.08],  # PM10 dominant -> dust-ish
            [0.25, 0.2, 0.1, 0.05, 0.38, 0.02],  # CO dominant -> stubble-ish
            [0.2, 0.15, 0.15, 0.42, 0.03, 0.05],  # SO2 dominant -> industrial-ish
        ]
    )
    names = _score_profiles(H_norm, ["pm25", "pm10", "no2", "so2", "co", "ozone"])
    assert len(set(names)) == 4


def test_clean_column_names_mapping():
    raw = pd.DataFrame(
        {
            "PM2.5 (\u00b5g/m\u00b3)": [1.0],
            "PM10 (\u00b5g/m\u00b3)": [2.0],
            "NO2 (\u00b5g/m\u00b3)": [3.0],
            "SO2 (\u00b5g/m\u00b3)": [4.0],
            "CO (mg/m\u00b3)": [5.0],
            "Ozone (\u00b5g/m\u00b3)": [6.0],
            "Station Name": ["X"],
            "Timestamp": ["2025-01-01"],
        }
    )
    out = clean_column_names(raw)
    for col in ["pm25", "pm10", "no2", "so2", "co", "ozone", "station", "timestamp"]:
        assert col in out.columns


def test_diagnostic_ratios_handle_zeros():
    df = pd.DataFrame({"pm25": [1.0], "pm10": [0.0], "no2": [5.0], "so2": [0.0], "co": [1.0]})
    out = compute_diagnostic_ratios(df)
    assert np.isnan(out["pm_ratio"].iloc[0])
    assert np.isnan(out["no2_so2_ratio"].iloc[0])
    assert out["co_no2_ratio"].iloc[0] == pytest.approx(200.0)  # co*1000/no2


def test_bootstrap_cis_bracket_mean_with_real_structure():
    res = bootstrap_source_shares(_synthetic_cpcb(n=400), n_sources=3, n_boot=40, random_state=7)
    for name, s in res.items():
        if np.isnan(s["ci90_low"]):
            continue
        assert s["ci90_low"] <= s["mean"] + 1e-6
        assert s["mean"] <= s["ci90_high"] + 1e-6


def test_insufficient_rows_raise():
    tiny = _synthetic_cpcb(n=10)
    with pytest.raises(ValueError):
        fit_receptor(tiny)
