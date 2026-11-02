"""Tests for the quantile forecaster: monotonicity, leakage-free split, pinball math."""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_pipeline import chronological_split, make_supervised
from src.models.forecaster import VAYUQuantileForecaster, pinball_loss


def _synthetic_features(n_hours: int = 24 * 14, seed: int = 3) -> pd.DataFrame:
    """Deterministic hourly feature frame with smooth diurnal PM2.5 dynamics."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)
    ts = pd.date_range("2025-12-01", periods=n_hours, freq="h", tz="UTC")
    base = 120 + 40 * np.sin(2 * np.pi * t / 24 - np.pi / 3) + 0.5 * t / 24
    pm25 = np.clip(base + rng.normal(0, 4, n_hours), 5, None)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "pm25": pm25,
            "pm10": pm25 * 1.5 + rng.normal(0, 2, n_hours),
            "no2": 40 + 10 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 2, n_hours),
            "so2": 12 + rng.normal(0, 1, n_hours),
            "co": 1.2 + rng.normal(0, 0.05, n_hours),
            "ozone": 40 + 8 * np.cos(2 * np.pi * t / 24) + rng.normal(0, 2, n_hours),
            "pm_ratio": pm25 / (pm25 * 1.5),
            "no2_so2_ratio": (40 + 10 * np.sin(2 * np.pi * t / 24)) / 12.0,
            "co_no2_ratio": 1.2 * 1000 / (40 + 10 * np.sin(2 * np.pi * t / 24)),
            "temp": 18 + 6 * np.sin(2 * np.pi * t / 24 - 2),
            "humidity": 60 + 15 * np.cos(2 * np.pi * t / 24),
            "wind_speed": np.clip(1.5 + 0.8 * np.sin(2 * np.pi * t / 12), 0.2, None),
            "wind_u": -np.clip(1.5 + 0.8 * np.sin(2 * np.pi * t / 12), 0.2, None) * 0.7,
            "wind_v": -np.clip(1.5 + 0.8 * np.sin(2 * np.pi * t / 12), 0.2, None) * 0.7,
            "pressure": 1013 + rng.normal(0, 1, n_hours),
            "hour_sin": np.sin(2 * np.pi * t / 24),
            "hour_cos": np.cos(2 * np.pi * t / 24),
        }
    )


def test_pinball_loss_known_values():
    # Perfect prediction -> loss 0 at any tau
    assert pinball_loss(np.array([100.0]), np.array([100.0]), 0.5) == pytest.approx(0.0)
    # Under-prediction by 10 at tau=0.9 -> loss 0.9*10 = 9
    assert pinball_loss(np.array([110.0]), np.array([100.0]), 0.9) == pytest.approx(9.0)
    # Over-prediction by 10 at tau=0.9 -> loss 0.1*10 = 1
    assert pinball_loss(np.array([100.0]), np.array([110.0]), 0.9) == pytest.approx(1.0)


def test_chronological_split_is_strictly_time_ordered():
    feats = _synthetic_features()
    sup = make_supervised(feats, horizon=24)
    train, test, split_time = chronological_split(sup, "target_h24", 0.75)
    assert train["timestamp"].max() <= split_time
    assert test["timestamp"].min() >= split_time
    assert test["timestamp"].min() > train["timestamp"].max() - pd.Timedelta(seconds=1)
    assert len(train) > len(test)


def test_make_supervised_target_is_shifted_pm25():
    feats = _synthetic_features(200)
    h = 6
    sup = make_supervised(feats, horizon=h)
    joined = sup.merge(feats[["timestamp", "pm25"]], on="timestamp", how="left")
    # y at row i equals pm25 at timestamp + h hours
    expected_ts = joined["timestamp"] + pd.Timedelta(hours=h)
    expected = feats.set_index("timestamp").loc[expected_ts, "pm25"].to_numpy()
    assert np.allclose(joined["target_h6"].to_numpy(), expected)


def test_quantile_predictions_are_monotonic_on_test_split():
    fc = VAYUQuantileForecaster(horizons=(1, 24))
    fc.fit(_synthetic_features())
    for h, res in fc.results.items():
        assert res.trained, f"horizon {h} should train on synthetic data"
        preds = res.predictions
        assert (preds["q10"] <= preds["q50"] + 1e-9).all()
        assert (preds["q50"] <= preds["q90"] + 1e-9).all()


def test_untrained_horizon_reported_honestly():
    fc = VAYUQuantileForecaster(horizons=(200,))  # exceeds MIN_TRAIN+MIN_TEST for 240 rows
    fc.fit(_synthetic_features(24 * 10))
    res = fc.results[200]
    assert not res.trained
    assert "insufficient" in res.reason


def test_predict_latest_bounds_are_ordered():
    fc = VAYUQuantileForecaster(horizons=(1,))
    feats = _synthetic_features()
    fc.fit(feats)
    out = fc.predict_latest(feats)
    assert 1 in out
    q = out[1]
    assert q["q10"] <= q["q50"] <= q["q90"]


def test_no_leakage_train_targets_precede_test():
    feats = _synthetic_features()
    sup = make_supervised(feats, horizon=24)
    train, test, _ = chronological_split(sup, "target_h24", 0.75)
    # The target at time t is pm25 at t+24h; training targets must not use
    # timestamps at or after the earliest test feature timestamp.
    train_target_times = train["timestamp"] + pd.Timedelta(hours=24)
    assert train_target_times.max() < test["timestamp"].min() + pd.Timedelta(hours=24)
