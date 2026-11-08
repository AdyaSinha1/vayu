"""Precompute all engine artifacts so the Streamlit app boots instantly.

Run once (locally or in CI):  python scripts/precompute.py

Writes:
  data/processed/engine_cache.json     - metrics, forecasts, explainability, attribution, station stats, meta
  data/processed/features_cache.parquet - hourly modelling table (for tables and fan charts)
  data/processed/stations_cleaned.parquet - cleaned per-station telemetry (station view + map)

The app checks these files first and only recomputes when they are missing.
"""

import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.attribution.receptor_nmf import get_attribution_summary  # noqa: E402
from src.features.aqi import POLLUTANT_LABELS, overall_aqi  # noqa: E402
from src.features.feature_pipeline import build_features, clean_cpcb_columns  # noqa: E402
from src.ingestion.weather_loader import load_weather  # noqa: E402
from src.models.explainer import explain_horizon  # noqa: E402
from src.models.forecaster import VAYUQuantileForecaster  # noqa: E402

OUT_JSON = BASE / "data" / "processed" / "engine_cache.json"
OUT_FEATURES = BASE / "data" / "processed" / "features_cache.parquet"
OUT_STATIONS = BASE / "data" / "processed" / "stations_cleaned.parquet"

STATION_COORDS_PATH = BASE / "app" / "streamlit_app.py"  # coords live in the app module


def _df_records(df: pd.DataFrame, max_rows: int | None = None) -> list[dict]:
    out = df if max_rows is None else df.tail(max_rows)
    return json.loads(out.to_json(orient="records", date_format="iso"))


def main() -> None:
    from src.ingestion.cpcb_data import ensure_cpcb_dataframe

    raw, cpcb_provenance = ensure_cpcb_dataframe()
    ts = pd.to_datetime(raw["Timestamp"], utc=True, errors="coerce").dropna()
    wdf, weather_status = load_weather(str((ts.min() - pd.Timedelta(hours=2)).date()), str(ts.max().date()))

    features, meta = build_features(raw, wdf if wdf is not None else None)
    meta["weather_fetch_status"] = weather_status
    meta["cpcb_provenance"] = cpcb_provenance

    forecaster = VAYUQuantileForecaster()
    forecaster.fit(features)

    explainer_24 = explain_horizon(forecaster, features, horizon=24)
    explainer_1 = explain_horizon(forecaster, features, horizon=1)
    attribution = get_attribution_summary(n_boot=150)

    # Station aggregates (same logic as the app)
    from src.features.aqi import aqi_category
    from src.features.stations import STATION_COORDS

    cleaned = clean_cpcb_columns(raw)
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], utc=True, errors="coerce")
    station_stats = []
    for s_name, grp in cleaned.groupby("station"):
        if s_name not in STATION_COORDS:
            continue
        means = {}
        for pol in ("pm25", "pm10", "no2", "so2", "co", "ozone"):
            if pol in grp.columns:
                m = grp[pol].dropna().mean()
                means[pol] = float(m) if pd.notna(m) else None
        if means.get("pm25") is None:
            continue
        aqi_v, driver, _ = overall_aqi(means)
        cat, _colour = aqi_category(aqi_v)
        lat, lon = STATION_COORDS[s_name]
        station_stats.append(
            {
                "station": s_name.split(",")[0],
                "full_name": s_name,
                "lat": lat,
                "lon": lon,
                **{POLLUTANT_LABELS[p]: (round(v, 1) if v is not None else None) for p, v in means.items()},
                "AQI": round(aqi_v) if aqi_v is not None else None,
                "category": cat,
                "driver": driver,
            }
        )

    results = {}
    for h, r in forecaster.results.items():
        results[str(h)] = {
            "horizon": h,
            "trained": r.trained,
            "reason": r.reason,
            "split_time": str(r.split_time) if r.split_time is not None else None,
            "n_train": r.n_train,
            "n_test": r.n_test,
            "metrics": r.metrics,
            "predictions": _df_records(r.predictions) if r.predictions is not None else None,
        }

    def _expl(ex: dict) -> dict:
        if "error" in ex:
            return {"error": ex["error"]}
        return {
            "shap_available": ex["shap_available"],
            "group_shares": ex["group_shares"],
            "interpretation": ex["interpretation"],
            "n_test": ex["n_test"],
            "n_repeats": ex["n_repeats"],
            "feature_importance": _df_records(ex["feature_importance"]),
        }

    cache = {
        "meta": meta,
        "latest_forecast": {str(h): q for h, q in forecaster.predict_latest(features).items()},
        "results": results,
        "explainer_24": _expl(explainer_24),
        "explainer_1": _expl(explainer_1),
        "attribution": attribution,
        "station_stats": station_stats,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(cache, fh)
    features.to_parquet(OUT_FEATURES, index=False)
    cleaned.drop(columns=[]) .to_parquet(OUT_STATIONS, index=False)

    print(f"engine_cache.json written ({OUT_JSON.stat().st_size / 1024:.0f} KB)")
    print(f"features_cache.parquet written ({OUT_FEATURES.stat().st_size / 1024:.0f} KB)")
    print(f"stations_cleaned.parquet written ({OUT_STATIONS.stat().st_size / 1024:.0f} KB)")
    print("Precompute complete - the app will now boot from cache.")


if __name__ == "__main__":
    main()
