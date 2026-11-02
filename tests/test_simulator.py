"""Tests for the GRAP policy simulator: bounds, monotonicity, renormalisation."""

import pytest

from src.attribution.simulator import (
    SOURCE_TO_LEVER,
    shares_to_intervention_keys,
    simulate_policy,
)

SHARES = {"vehicular": 36.2, "stubble": 31.4, "industrial": 18.1, "dust": 14.3}


def test_zero_interventions_leave_pm25_unchanged():
    out = simulate_policy(250.0, SHARES, {"vehicular": 0, "stubble": 0, "industrial": 0, "dust": 0})
    assert out["new_pm25"] == pytest.approx(250.0)
    assert out["avoided_ug"] == pytest.approx(0.0)


def test_more_reduction_gives_less_pm25_monotonic():
    low = simulate_policy(250.0, SHARES, {"vehicular": 10, "stubble": 10, "industrial": 10, "dust": 10})
    high = simulate_policy(250.0, SHARES, {"vehicular": 50, "stubble": 50, "industrial": 50, "dust": 50})
    assert high["new_pm25"] < low["new_pm25"]
    assert high["avoided_ug"] > low["avoided_ug"]


def test_full_intervention_on_all_sources_gives_zero_pm25():
    out = simulate_policy(250.0, SHARES, {"vehicular": 100, "stubble": 100, "industrial": 100, "dust": 100})
    assert out["new_pm25"] == pytest.approx(0.0, abs=1e-9)
    assert out["avoided_ug"] == pytest.approx(250.0, abs=1e-9)


def test_out_of_bounds_reductions_raise():
    with pytest.raises(ValueError):
        simulate_policy(250.0, SHARES, {"vehicular": 120, "stubble": 0, "industrial": 0, "dust": 0})
    with pytest.raises(ValueError):
        simulate_policy(250.0, SHARES, {"vehicular": -5, "stubble": 0, "industrial": 0, "dust": 0})


def test_unknown_intervention_key_raises():
    with pytest.raises(ValueError):
        simulate_policy(250.0, SHARES, {"magic": 50})


def test_negative_current_pm25_raises():
    with pytest.raises(ValueError):
        simulate_policy(-1.0, SHARES, {})


def test_post_shares_renormalise_to_100():
    out = simulate_policy(250.0, SHARES, {"vehicular": 50, "stubble": 0, "industrial": 0, "dust": 0})
    total = sum(out["post_shares"].values())
    assert total == pytest.approx(100.0, abs=1e-6)
    assert out["post_shares"]["vehicular"] < out["post_shares"]["stubble"]


def test_linear_first_order_arithmetic():
    # 30% cut on a 36.2% source removes 0.3 * 36.2% = 10.86% of total mass
    out = simulate_policy(100.0, SHARES, {"vehicular": 30, "stubble": 0, "industrial": 0, "dust": 0})
    assert out["avoided_pct"] == pytest.approx(0.30 * 36.2, abs=1e-6)
    assert out["new_pm25"] == pytest.approx(100.0 * (1 - 0.30 * 0.362), abs=1e-6)


def test_lever_mapping_covers_all_four_sources():
    shares = {name: 25.0 for name in SOURCE_TO_LEVER}
    keys = shares_to_intervention_keys(shares, SOURCE_TO_LEVER)
    assert set(keys) == {"vehicular", "stubble", "industrial", "dust"}
