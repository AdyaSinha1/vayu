"""Download the latest CPCB air-quality parquet and derive the Delhi-NCR slice.

Uses the shared resolver in src/ingestion/cpcb_data.py, so paths are
repo-relative (never machine-specific): if the repo's data/ directory is not
writable, the cache falls back to a temp directory automatically.

This script is optional: the Streamlit app bootstraps the same dataset from
the open mirror on first load (see cpcb_data.ensure_cpcb_dataframe).
"""

from __future__ import annotations

from src.ingestion.cpcb_data import (
    PROCESSED_PARQUET,
    ensure_cpcb_dataframe,
)


def main() -> None:
    print("1. Resolving CPCB Delhi telemetry (local cache or open mirror)...")
    delhi, provenance = ensure_cpcb_dataframe()
    print(f"   Provenance: {provenance}")

    print(f"2. Delhi records: {len(delhi)}")
    station_col = "Station Name" if "Station Name" in delhi.columns else "Station ID"
    if station_col in delhi.columns:
        print("   Unique stations (sample):", delhi[station_col].unique()[:10])

    out_csv = PROCESSED_PARQUET.with_suffix(".csv")
    try:
        delhi.to_parquet(PROCESSED_PARQUET, index=False)
        print(f"   Saved parquet: {PROCESSED_PARQUET}")
        delhi.head(1000).to_csv(out_csv, index=False)
        print(f"   Preview CSV:   {out_csv}")
    except Exception as exc:
        print(f"   Cache write skipped ({exc}); data remains available in memory this session.")


if __name__ == "__main__":
    main()
