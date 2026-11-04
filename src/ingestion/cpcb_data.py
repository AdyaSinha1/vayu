"""Locate or bootstrap the real CPCB Delhi telemetry for VAYU.

Resolution order (honest provenance, never fabricates):
1. ``data/processed/delhi_cpcb_real.parquet`` - as produced by the download script.
2. ``data/raw/latest-air-quality.parquet`` - derive the Delhi slice locally.
3. Download the public CPCB mirror parquet (same open dataset as the download
   script) and derive the Delhi slice - cached best-effort for faster restarts.

Step 3 is what makes the Streamlit Community Cloud deployment work: no data
files are committed (they are git-ignored), so the live app bootstraps the
telemetry from the open mirror on first load. If every step fails, callers
receive ``CPCBDataUnavailable`` and must display "unavailable" - no value is
ever estimated or invented.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Public open mirror of the CPCB CAAQMS feed (same source as
# src/ingestion/download_cpcb_delhi.py).
MIRROR_PARQUET_URL = (
    "https://raw.githubusercontent.com/Vonter/india-cpcb-aqi/main/data/latest-air-quality.parquet"
)


class CPCBDataUnavailable(RuntimeError):
    """No local copy exists and the open mirror could not be reached."""


def _resolve_cache_dirs() -> tuple[Path, Path]:
    """(processed_dir, raw_dir) - prefer the repo, fall back to a temp dir on read-only filesystems."""
    base = REPO_ROOT / "data"
    try:
        (base / "processed").mkdir(parents=True, exist_ok=True)
        probe = base / "raw" / ".write_probe"
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.touch()
        probe.unlink()
        return base / "processed", base / "raw"
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "vayu-data"
        (fallback / "processed").mkdir(parents=True, exist_ok=True)
        (fallback / "raw").mkdir(parents=True, exist_ok=True)
        return fallback / "processed", fallback / "raw"


PROCESSED_DIR, RAW_DIR = _resolve_cache_dirs()
PROCESSED_PARQUET = PROCESSED_DIR / "delhi_cpcb_real.parquet"
RAW_PARQUET = RAW_DIR / "latest-air-quality.parquet"


def _delhi_slice(df: pd.DataFrame) -> pd.DataFrame:
    if "City" in df.columns:
        out = df[df["City"].astype(str).str.contains("Delhi", case=False, na=False)].copy()
    elif "State" in df.columns:
        out = df[df["State"].astype(str).str.contains("Delhi", case=False, na=False)].copy()
    else:
        out = df.copy()
    return out.reset_index(drop=True)


def _download_raw_mirror() -> tuple[pd.DataFrame, str]:
    resp = requests.get(MIRROR_PARQUET_URL, timeout=120)
    resp.raise_for_status()
    try:
        RAW_PARQUET.write_bytes(resp.content)
    except OSError:
        pass  # read-only cache: keep working from memory
    return pd.read_parquet(io.BytesIO(resp.content)), "downloaded live from open CPCB mirror"


def ensure_cpcb_dataframe() -> tuple[pd.DataFrame, str]:
    """Return (Delhi telemetry frame, provenance string), bootstrapping if needed."""
    if PROCESSED_PARQUET.exists():
        return pd.read_parquet(PROCESSED_PARQUET), "local file (data/processed)"

    if RAW_PARQUET.exists():
        try:
            df = pd.read_parquet(RAW_PARQUET)
            delhi = _delhi_slice(df)
            if not delhi.empty:
                _cache_processed(delhi)
                return pd.read_parquet(PROCESSED_PARQUET), "derived from local raw mirror parquet"
        except Exception:
            pass  # fall through to the live download

    df, provenance = _download_raw_mirror()
    delhi = _delhi_slice(df)
    if delhi.empty:
        raise CPCBDataUnavailable("CPCB mirror reachable but no Delhi rows found in the parquet")
    _cache_processed(delhi)
    if PROCESSED_PARQUET.exists():
        return pd.read_parquet(PROCESSED_PARQUET), provenance
    return delhi, provenance + " (read-only cache: served from memory)"


def _cache_processed(delhi: pd.DataFrame) -> None:
    try:
        delhi.to_parquet(PROCESSED_PARQUET, index=False)
    except Exception:
        pass  # read-only filesystem: provenance string reflects this
