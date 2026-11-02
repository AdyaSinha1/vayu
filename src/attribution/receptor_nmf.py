"""Positive Matrix Factorization (via NMF) receptor model for VAYU.

Decomposes the multi-pollutant concentration matrix X (n x m) into
non-negative factors W (n x k) * H (k x m) - the same mathematical form as
the US EPA PMF5 receptor model - and reports mean source shares with
BOOTSTRAP credible intervals computed by refitting on resampled data.

Factor-to-source naming uses a documented heuristic over dominant chemical
markers (NO2/benzene -> vehicular, CO -> biomass burning, SO2 -> industrial
combustion, coarse PM10 -> crustal dust) and is always presented as a
heuristic assignment, never as a measured truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_PARQUET = BASE_DIR / "data" / "processed" / "delhi_cpcb_real.parquet"

NMF_FEATURES = ["pm25", "pm10", "no2", "so2", "co", "ozone"]

SOURCE_CATALOG = [
    "Vehicular Fleet Exhaust",
    "Industrial & Thermal Combustion",
    "Agricultural Biomass / Stubble Burning",
    "Crustal & Resuspended Dust",
]


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize CPCB column names to simple ASCII snake_case."""
    mapping = {}
    for col in df.columns:
        c = str(col).lower()
        if "pm2.5" in c:
            mapping[col] = "pm25"
        elif "pm10" in c:
            mapping[col] = "pm10"
        elif "no2" in c:
            mapping[col] = "no2"
        elif "nox" in c:
            mapping[col] = "nox"
        elif "so2" in c:
            mapping[col] = "so2"
        elif c.startswith("co ") or c == "co":
            mapping[col] = "co"
        elif "ozone" in c or "o3" in c:
            mapping[col] = "ozone"
        elif "nh3" in c:
            mapping[col] = "nh3"
        elif "station name" in c:
            mapping[col] = "station"
        elif "timestamp" in c:
            mapping[col] = "timestamp"
        elif "benzene" in c:
            mapping[col] = "benzene"
        elif "toluene" in c:
            mapping[col] = "toluene"
        elif c.startswith("at ") or "ambient" in c:
            mapping[col] = "temp"
        elif c.startswith("rh "):
            mapping[col] = "humidity"
        elif c.startswith("ws "):
            mapping[col] = "wind_speed"
        elif c.startswith("wd "):
            mapping[col] = "wind_dir"
        elif c.startswith("bp "):
            mapping[col] = "pressure"
    return df.rename(columns=mapping)


def compute_diagnostic_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate atmospheric chemistry diagnostic ratios (documented thresholds)."""
    df = df.copy()
    if "pm25" in df.columns and "pm10" in df.columns:
        # > 0.65 secondary combustion / fine aerosols; < 0.40 crustal dust
        df["pm_ratio"] = df["pm25"] / df["pm10"].replace(0, np.nan)
    if "no2" in df.columns and "so2" in df.columns:
        # > 2.5 vehicular NOx dominance; < 1.0 coal/industrial boilers
        df["no2_so2_ratio"] = df["no2"] / df["so2"].replace(0, np.nan)
    if "co" in df.columns and "no2" in df.columns:
        # high values indicate low-temperature smoldering (biomass)
        df["co_no2_ratio"] = (df["co"] * 1000.0) / df["no2"].replace(0, np.nan)
    return df


def _design_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feats = [f for f in NMF_FEATURES if f in df.columns]
    sub = df[feats].dropna()
    sub = sub[(sub >= 0).all(axis=1)]
    return sub.to_numpy(dtype=float), feats


def _score_profiles(H_norm: np.ndarray, feats: list[str]) -> list[str]:
    """Assign each factor a physical source name via documented marker heuristics.

    Score vector per source (weights reflect which species characterise it):
      vehicular:   NO2 (1.0) + benzene (0.8) + PM2.5 (0.4) - SO2 (0.5)
      industrial:  SO2 (1.0) + NO2 (0.4) - PM10 (0.3)
      stubble:     CO (1.0) + PM2.5 (0.4)
      dust:        PM10 (1.0) - PM2.5 (0.5)
    Negative terms express anti-markers. Assignment is greedy by score.
    """
    idx = {f: i for i, f in enumerate(feats)}

    def val(profile: np.ndarray, f: str) -> float:
        return float(profile[idx[f]]) if f in idx else 0.0

    scores = {s: [] for s in SOURCE_CATALOG}
    for profile in H_norm:
        scores["Vehicular Fleet Exhaust"].append(
            1.0 * val(profile, "no2") + 0.8 * val(profile, "benzene") + 0.4 * val(profile, "pm25") - 0.5 * val(profile, "so2")
        )
        scores["Industrial & Thermal Combustion"].append(
            1.0 * val(profile, "so2") + 0.4 * val(profile, "no2") - 0.3 * val(profile, "pm10")
        )
        scores["Agricultural Biomass / Stubble Burning"].append(
            1.0 * val(profile, "co") + 0.4 * val(profile, "pm25")
        )
        scores["Crustal & Resuspended Dust"].append(
            1.0 * val(profile, "pm10") - 0.5 * val(profile, "pm25")
        )

    # Greedy unique assignment by descending score.
    pairs = sorted(
        ((scores[s][i], i, s) for s in SOURCE_CATALOG for i in range(len(H_norm))),
        key=lambda t: -t[0],
    )
    assigned: dict[int, str] = {}
    used: set[str] = set()
    for _, i, s in pairs:
        if i in assigned or s in used:
            continue
        assigned[i] = s
        used.add(s)
    for i in range(len(H_norm)):  # fallback: any unused name
        if i not in assigned:
            leftover = [s for s in SOURCE_CATALOG if s not in used]
            assigned[i] = leftover.pop() if leftover else f"Factor {i + 1}"
    return [assigned[i] for i in range(len(H_norm))]


def fit_receptor(df: pd.DataFrame, n_sources: int = 4, random_state: int = 42):
    """Fit NMF once. Returns dict with profiles, shares, names and diagnostics."""
    X, feats = _design_matrix(df)
    if len(X) < 50:
        raise ValueError(f"insufficient valid multi-pollutant rows for NMF ({len(X)})")
    model = NMF(
        n_components=n_sources,
        init="nndsvda",
        max_iter=1000,
        random_state=random_state,
        alpha_W=0.01,
        alpha_H=0.01,
    )
    W = model.fit_transform(X)
    H = model.components_
    H_norm = H / H.sum(axis=1, keepdims=True)
    names = _score_profiles(H_norm, feats)
    shares = W.mean(axis=0)
    shares_pct = 100.0 * shares / shares.sum()
    residual = float(np.linalg.norm(X - W @ H) / np.linalg.norm(X))
    return {
        "n_samples": int(len(X)),
        "features": feats,
        "profiles": pd.DataFrame(H_norm, index=names, columns=feats),
        "names": names,
        "shares": {n: float(p) for n, p in zip(names, shares_pct)},
        "reconstruction_residual": residual,
    }


def bootstrap_source_shares(
    df: pd.DataFrame,
    n_sources: int = 4,
    n_boot: int = 200,
    random_state: int = 42,
) -> dict:
    """Fit NMF on bootstrap resamples -> mean shares + empirical 90% credible intervals.

    This replaces hand-written error bars: every interval comes from the real
    variability of the fit under row resampling.
    """
    X, feats = _design_matrix(df)
    if len(X) < 50:
        raise ValueError(f"insufficient valid multi-pollutant rows for bootstrap ({len(X)})")

    rng = np.random.default_rng(random_state)
    share_samples: dict[str, list[float]] = {s: [] for s in SOURCE_CATALOG}
    base = fit_receptor(df, n_sources=n_sources, random_state=random_state)
    name_order = base["names"]

    for b in range(n_boot):
        idx = rng.integers(0, len(X), size=len(X))
        boot_df = pd.DataFrame(X[idx], columns=feats)
        try:
            res = fit_receptor(boot_df, n_sources=n_sources, random_state=random_state + b)
        except Exception:
            continue
        for name, pct in res["shares"].items():
            share_samples[name].append(pct)

    summary = {}
    for name in name_order:
        vals = np.asarray(share_samples.get(name, []), dtype=float)
        if len(vals) < 30:
            summary[name] = {
                "mean": base["shares"][name],
                "ci90_low": np.nan,
                "ci90_high": np.nan,
                "sigma": np.nan,
                "n_boot_ok": int(len(vals)),
            }
            continue
        summary[name] = {
            "mean": float(np.mean(vals)),
            "ci90_low": float(np.percentile(vals, 5)),
            "ci90_high": float(np.percentile(vals, 95)),
            "sigma": float(np.std(vals, ddof=1)),
            "n_boot_ok": int(len(vals)),
        }
    return summary


def run_full_attribution(df_raw: pd.DataFrame, n_boot: int = 200) -> dict:
    """End-to-end attribution on raw CPCB frame: clean -> ratios -> fit -> bootstrap."""
    cleaned = clean_column_names(df_raw)
    enriched = compute_diagnostic_ratios(cleaned)
    result = fit_receptor(enriched)
    result["bootstrap"] = bootstrap_source_shares(enriched, n_sources=4, n_boot=n_boot)

    med = enriched[["pm_ratio", "no2_so2_ratio", "co_no2_ratio"]].median(numeric_only=True)
    result["diagnostic_medians"] = {k: float(v) for k, v in med.items()}
    return result


ATTRIBUTION_CACHE = BASE_DIR / "data" / "processed" / "attribution_summary.json"


def _profiles_to_dict(profiles: pd.DataFrame) -> dict:
    return {str(idx): {c: round(float(v), 4) for c, v in row.items()} for idx, row in profiles.iterrows()}


def get_attribution_summary(n_boot: int = 150, use_cache: bool = True) -> dict:
    """Attribution summary with a JSON disk cache (invalidated when CPCB parquet changes).

    Returns a JSON-serialisable dict for UI consumption. When the cache is
    stale or missing, the full NMF + bootstrap is recomputed from real data.
    """
    raw_path = Path(PROCESSED_PARQUET)
    if use_cache and ATTRIBUTION_CACHE.exists() and raw_path.exists():
        if ATTRIBUTION_CACHE.stat().st_mtime > raw_path.stat().st_mtime:
            import json

            with open(ATTRIBUTION_CACHE, "r", encoding="utf-8") as fh:
                return json.load(fh)

    raw = pd.read_parquet(raw_path)
    res = run_full_attribution(raw, n_boot=n_boot)
    summary = {
        "n_samples": res["n_samples"],
        "reconstruction_residual": round(res["reconstruction_residual"], 4),
        "diagnostic_medians": {k: round(v, 3) for k, v in res["diagnostic_medians"].items()},
        "shares": {k: round(v, 2) for k, v in res["shares"].items()},
        "bootstrap": {
            k: {kk: (None if vv is None or (isinstance(vv, float) and np.isnan(vv)) else round(vv, 2)) for kk, vv in v.items()}
            for k, v in res["bootstrap"].items()
        },
        "profiles": _profiles_to_dict(res["profiles"]),
        "features_used": res["features"],
    }
    if use_cache:
        import json

        ATTRIBUTION_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(ATTRIBUTION_CACHE, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    return summary


if __name__ == "__main__":
    raw = pd.read_parquet(PROCESSED_PARQUET)
    print("Fitting receptor model on 100% real CPCB Delhi telemetry...")
    res = run_full_attribution(raw, n_boot=200)
    print(f"\nValid multi-pollutant samples: {res['n_samples']}")
    print(f"NMF reconstruction residual (rel. Frobenius): {res['reconstruction_residual']:.3f}")
    print("\nDiagnostic ratio medians:", {k: round(v, 2) for k, v in res["diagnostic_medians"].items()})
    print("\n--- Heuristic-assigned source shares with bootstrap 90% CI ---")
    for name, s in res["bootstrap"].items():
        print(f"  {name}: {s['mean']:.1f}%  [90% CI: {s['ci90_low']:.1f} - {s['ci90_high']:.1f}]  (n_boot_ok={s['n_boot_ok']})")
    print("\nProfiles (normalised):")
    print(res["profiles"].round(3).to_string())
