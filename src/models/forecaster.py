"""VAYU quantile forecaster - direct multi-horizon PM2.5 prediction intervals.

For each horizon h in {1, 24, 48, 72} h and each quantile tau in
{0.10, 0.50, 0.90}, a HistGradientBoostingRegressor is trained with
quantile (pinball) loss on features known strictly at forecast origin t.
Targets are PM2.5 measured at t + h (real CPCB telemetry).

Evaluation is strictly chronological (train = earlier 75%, test = later 25%)
and every reported metric is computed from real out-of-time predictions.
The naive persistence baseline (prediction = last observed PM2.5) is always
reported alongside so improvements are measurable, not asserted.

No metric in this module is ever hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features.feature_pipeline import (
    CPCB_PARQUET,
    MODEL_FEATURES,
    chronological_split,
    load_and_build,
    make_supervised,
)

MIN_TRAIN_ROWS = 40
MIN_TEST_ROWS = 12


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """Mean quantile (pinball) loss at quantile level tau."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


@dataclass
class HorizonResult:
    horizon: int
    trained: bool
    reason: str = ""
    split_time: pd.Timestamp | None = None
    n_train: int = 0
    n_test: int = 0
    metrics: dict = field(default_factory=dict)
    predictions: pd.DataFrame | None = None  # timestamp, y_true, q10, q50, q90, persistence


class VAYUQuantileForecaster:
    """Direct multi-horizon quantile regression forecaster for PM2.5."""

    def __init__(
        self,
        horizons: tuple[int, ...] = (1, 24, 48, 72),
        taus: tuple[float, ...] = (0.10, 0.50, 0.90),
        train_frac: float = 0.75,
        random_state: int = 42,
    ) -> None:
        self.horizons = horizons
        self.taus = taus
        self.train_frac = train_frac
        self.random_state = random_state
        self.results: dict[int, HorizonResult] = {}
        self._models_by_horizon: dict[int, dict[float, HistGradientBoostingRegressor]] = {}
        self._feature_cols_by_horizon: dict[int, list[str]] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, features: pd.DataFrame, feature_list: list[str] | None = None) -> "VAYUQuantileForecaster":
        feature_list = feature_list or [f for f in MODEL_FEATURES if f in features.columns]
        for h in self.horizons:
            self._fit_horizon(features, h, feature_list)
        return self

    def _fit_horizon(self, features: pd.DataFrame, horizon: int, feature_list: list[str]) -> HorizonResult:
        sup = make_supervised(features, horizon, feature_list)
        if len(sup) < MIN_TRAIN_ROWS + MIN_TEST_ROWS:
            self.results[horizon] = HorizonResult(
                horizon=horizon,
                trained=False,
                reason=f"insufficient supervised rows ({len(sup)}) for horizon {horizon}h",
            )
            return self.results[horizon]

        train, test, split_time = chronological_split(sup, f"target_h{horizon}", self.train_frac)
        if len(train) < MIN_TRAIN_ROWS or len(test) < MIN_TEST_ROWS:
            self.results[horizon] = HorizonResult(
                horizon=horizon,
                trained=False,
                reason=f"insufficient chronological rows (train={len(train)}, test={len(test)})",
            )
            return self.results[horizon]

        feat_cols = [c for c in sup.columns if c not in ("timestamp", f"target_h{horizon}")]
        # Drop features that are entirely missing (e.g. BLH when offline) - honest no-op.
        feat_cols = [c for c in feat_cols if sup[c].notna().any()]
        X_train, y_train = train[feat_cols], train[f"target_h{horizon}"]
        X_test, y_test = test[feat_cols], test[f"target_h{horizon}"]

        # Quantile models - one per tau, modest capacity for the small real sample.
        models: dict[float, HistGradientBoostingRegressor] = {}
        raw_preds: dict[float, np.ndarray] = {}
        for tau in self.taus:
            m = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=tau,
                max_iter=300,
                learning_rate=0.06,
                max_depth=3,
                min_samples_leaf=5,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=self.random_state,
            )
            m.fit(X_train, y_train)
            models[tau] = m
            raw_preds[tau] = m.predict(X_test)

        # Enforce quantile monotonicity (crossing correction; standard practice).
        qmat = np.vstack([raw_preds[t] for t in self.taus])
        qmat_sorted = np.sort(qmat, axis=0)
        preds = {t: qmat_sorted[i] for i, t in enumerate(self.taus)}

        y_hat_median = preds[0.50]
        persistence = X_test["pm25"].to_numpy(dtype=float)  # last observed PM2.5 = naive baseline

        y_true = y_test.to_numpy(dtype=float)
        metrics = {
            "mae": float(mean_absolute_error(y_true, y_hat_median)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_hat_median))),
            "r2": float(r2_score(y_true, y_hat_median)),
            "pinball_q10": pinball_loss(y_true, preds[0.10], 0.10),
            "pinball_q50": pinball_loss(y_true, y_hat_median, 0.50),
            "pinball_q90": pinball_loss(y_true, preds[0.90], 0.90),
            "interval_coverage_80": float(np.mean((y_true >= preds[0.10]) & (y_true <= preds[0.90]))),
            "mean_interval_width": float(np.mean(preds[0.90] - preds[0.10])),
            "persistence_mae": float(mean_absolute_error(y_true, persistence)),
            "persistence_rmse": float(np.sqrt(mean_squared_error(y_true, persistence))),
        }

        self._models_by_horizon[horizon] = models
        self._feature_cols_by_horizon[horizon] = feat_cols
        self.results[horizon] = HorizonResult(
            horizon=horizon,
            trained=True,
            split_time=split_time,
            n_train=len(train),
            n_test=len(test),
            metrics=metrics,
            predictions=pd.DataFrame(
                {
                    "timestamp": test["timestamp"].to_numpy(),
                    "y_true": y_true,
                    "q10": preds[0.10],
                    "q50": preds[0.50],
                    "q90": preds[0.90],
                    "persistence": persistence,
                }
            ),
        )
        return self.results[horizon]

    # ------------------------------------------------------------- predict
    def predict_latest(self, features: pd.DataFrame) -> dict[int, dict[str, float]]:
        """Forecast forward from the LAST available observation for every trained horizon.

        Returns {horizon: {"q10": ..., "q50": ..., "q90": ...}} using the most
        recent row of real telemetry as the forecast origin. If a horizon was
        not trained it is simply absent - nothing is simulated.
        """
        last = features.sort_values("timestamp").iloc[-1]
        out: dict[int, dict[str, float]] = {}
        for h, models in self._models_by_horizon.items():
            feat_cols = self._feature_cols_by_horizon[h]
            x = pd.DataFrame([last[feat_cols].to_dict()])
            qvals = [float(models[tau].predict(x[feat_cols])[0]) for tau in self.taus]
            q10, q50, q90 = sorted(qvals)
            out[h] = {"q10": q10, "q50": q50, "q90": q90}
        return out


def run_pipeline(weather_df: pd.DataFrame | None = None) -> tuple[VAYUQuantileForecaster, pd.DataFrame, dict]:
    """Convenience end-to-end: features (+ optional weather enrichment), fit forecaster.

    If ``weather_df`` is None, attempts Open-Meteo (live, then cache). Failure is
    tracked in meta["weather_provenance"] and never blocks the pipeline.
    """
    if weather_df is None:
        from src.ingestion.weather_loader import load_weather

        # Derive the weather window from the CPCB telemetry itself.
        raw = pd.read_parquet(CPCB_PARQUET)
        ts = pd.to_datetime(raw["Timestamp"], utc=True, errors="coerce").dropna()
        start = (ts.min() - pd.Timedelta(hours=2)).date()
        end = ts.max().date()
        wdf, provenance = load_weather(str(start), str(end))
        if wdf is not None:
            weather_df = wdf
    features, meta = load_and_build(weather_df)
    meta["weather_fetch_status"] = provenance if weather_df is not None else "not attempted"
    fc = VAYUQuantileForecaster()
    fc.fit(features)
    return fc, features, meta


if __name__ == "__main__":
    fc, features, meta = run_pipeline()
    print("=== VAYU Quantile Forecaster | honest chronological evaluation ===")
    print(f"Feature table: {meta['n_hourly_rows']} hourly rows, {meta['time_start']} -> {meta['time_end']}")
    print(f"Weather provenance: {meta['weather_provenance']}")
    for h in sorted(fc.results):
        r = fc.results[h]
        print(f"\n--- Horizon +{h}h ---")
        if not r.trained:
            print(f"NOT TRAINED: {r.reason}")
            continue
        print(f"train n={r.n_train} (<= {r.split_time}), test n={r.n_test} (strictly out-of-time)")
        for k, v in r.metrics.items():
            print(f"  {k}: {v:.3f}")
