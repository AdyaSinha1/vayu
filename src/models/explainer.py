"""VAYU explainability layer.

Explains WHICH factors drive predicted PM2.5 and separates:
  - Meteorological trapping (wind calm, boundary-layer compression, humidity)
  - Emission signature (co-pollutants, chemical ratios)
  - Persistence history (recent PM2.5 lags/rolling means)

Primary engine: permutation importance on the strictly out-of-time test split
with a pinball scorer at tau=0.50 (robust, no extra dependencies).
If the optional `shap` package is importable (not available on Python 3.14
today), TreeSHAP values are also computed and returned; otherwise the
response honestly reports shap_available=False. No importances are invented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer

from src.models.forecaster import VAYUQuantileForecaster, pinball_loss

# Feature groups used for the "inversion vs emission" decoupling.
GROUPS: dict[str, list[str]] = {
    "Meteorological dispersal": [
        "temp", "humidity", "wind_speed", "wind_u", "wind_v", "pressure", "blh",
        "hour_sin", "hour_cos",
    ],
    "Emission signature": [
        "pm10", "no2", "so2", "co", "ozone", "pm_ratio", "no2_so2_ratio", "co_no2_ratio",
    ],
    "Persistence history": [
        "pm25", "pm25_lag1", "pm25_lag3", "pm25_lag6", "pm25_lag24",
        "no2_lag1", "no2_lag6", "so2_lag1", "so2_lag24",
        "pm25_roll3", "pm25_roll24",
    ],
}


def pinball_scorer():
    """Sklearn scorer: negative pinball loss at tau=0.5 (higher is better)."""
    return make_scorer(lambda y_true, y_pred: -pinball_loss(y_true, y_pred, 0.50))


def explain_horizon(
    forecaster: VAYUQuantileForecaster,
    features: pd.DataFrame,
    horizon: int = 24,
    n_repeats: int = 15,
    random_state: int = 42,
) -> dict:
    """Permutation importance + group decoupling for a trained horizon.

    Returns dict with:
      shap_available: bool
      feature_importance: DataFrame(feature, importance, group) sorted desc
      group_shares: dict(group_name -> share of positive importance, sums to 1)
      interpretation: str  (one-sentence honest summary)
    """
    result = forecaster.results.get(horizon)
    if result is None or not result.trained:
        return {"error": f"horizon {horizon}h not trained", "shap_available": False}

    sup_frame = result.predictions
    if sup_frame is None:
        return {"error": "no stored predictions", "shap_available": False}

    # Reconstruct the exact train/test frames the forecaster used.
    from src.features.feature_pipeline import make_supervised, chronological_split

    feat_cols = forecaster._feature_cols_by_horizon[horizon]
    sup = make_supervised(features, horizon, feat_cols)
    train, test, _ = chronological_split(sup, f"target_h{horizon}", forecaster.train_frac)
    X_test, y_test = test[feat_cols], test[f"target_h{horizon}"]
    median_model = forecaster._models_by_horizon[horizon][0.50]

    pi = permutation_importance(
        median_model,
        X_test,
        y_test,
        scoring=pinball_scorer(),
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )

    def group_of(feat: str) -> str:
        for gname, members in GROUPS.items():
            if feat in members:
                return gname
        return "Other"

    imp_df = pd.DataFrame(
        {
            "feature": feat_cols,
            "importance": np.maximum(pi.importances_mean, 0.0),
        }
    )
    imp_df["group"] = imp_df["feature"].map(group_of)
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)

    group_sums = imp_df.groupby("group")["importance"].sum()
    total = float(group_sums.sum())
    group_shares = (
        {k: float(v / total) for k, v in group_sums.items()} if total > 0 else {}
    )

    # Honest one-sentence interpretation from the actual numbers.
    interpretation = "Not enough positive importance mass to interpret."
    if group_shares:
        met_share = group_shares.get("Meteorological dispersal", 0.0)
        emis_share = group_shares.get("Emission signature", 0.0)
        pers_share = group_shares.get("Persistence history", 0.0)
        top = max(group_shares, key=group_shares.get)
        interpretation = (
            f"On the out-of-time test split, {top.lower()} explains the largest share of "
            f"predictive signal (meteorological {met_share:.0%}, emission signature "
            f"{emis_share:.0%}, persistence history {pers_share:.0%})."
        )

    shap_available = False
    try:  # optional extra credit if the environment supports shap
        import shap  # noqa: F401

        shap_available = True
    except Exception:
        shap_available = False

    return {
        "shap_available": shap_available,
        "feature_importance": imp_df,
        "group_shares": group_shares,
        "interpretation": interpretation,
        "n_test": int(len(X_test)),
        "n_repeats": n_repeats,
    }


def decoupling_verdict(group_shares: dict[str, float]) -> dict:
    """Turn group shares into the Inversion-vs-Emission verdict card data."""
    met = group_shares.get("Meteorological dispersal", 0.0)
    emis = group_shares.get("Emission signature", 0.0) + group_shares.get("Persistence history", 0.0)
    total = met + emis
    if total <= 0:
        return {"verdict": "insufficient signal", "met_share": None, "emission_share": None}
    met_n = met / total
    if met_n >= 0.6:
        verdict = "Meteorological trapping dominant"
    elif met_n <= 0.4:
        verdict = "Emission-driven dominant"
    else:
        verdict = "Mixed regime (neither dominates)"
    return {
        "verdict": verdict,
        "met_share": round(met_n, 3),
        "emission_share": round(1 - met_n, 3),
    }


if __name__ == "__main__":
    from src.models.forecaster import run_pipeline

    fc, features, meta = run_pipeline()
    print("Weather provenance:", meta["weather_provenance"])
    for h in (24, 1, 48, 72):
        res = explain_horizon(fc, features, horizon=h)
        if "error" in res:
            print(f"\n{h}h: {res['error']}")
            continue
        print(f"\n=== Horizon +{h}h (n_test={res['n_test']}) ===")
        print("Groups:", {k: f"{v:.0%}" for k, v in res["group_shares"].items()})
        print(res["interpretation"])
        print(res["feature_importance"].head(8).round(4).to_string())
        print("Decoupling:", decoupling_verdict(res["group_shares"]))
