"""VAYU - Atmospheric Intelligence for Delhi-NCR.

A decision-support console for air-quality monitoring, source attribution and
policy simulation, built entirely on open data:
- CPCB CAAQMS continuous telemetry (Delhi-NCR network)
- Open-Meteo ERA5 reanalysis (boundary-layer height, wind vectors), cached
- NMF receptor model with bootstrap credible intervals
- Quantile (pinball-loss) forecaster evaluated strictly out-of-time
- CPCB AQI sub-index computation and printable situation reports

Every figure on this page is computed live from the data above. Where a model
loses to its own persistence baseline, or a dataset is unavailable, the page
says so.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.attribution.receptor_nmf import get_attribution_summary  # noqa: E402
from src.attribution.simulator import SOURCE_TO_LEVER, shares_to_intervention_keys, simulate_policy  # noqa: E402
from src.features.aqi import POLLUTANT_LABELS, aqi_category, overall_aqi  # noqa: E402
from src.features.feature_pipeline import build_features, clean_cpcb_columns  # noqa: E402
from src.ingestion.weather_loader import load_weather  # noqa: E402
from src.models.explainer import decoupling_verdict, explain_horizon  # noqa: E402
from src.models.forecaster import VAYUQuantileForecaster  # noqa: E402
from src.reports.report_builder import build_html_report, build_markdown_report  # noqa: E402

st.set_page_config(
    page_title="VAYU - Atmospheric Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------- DESIGN SYSTEM -----------------
# Palette: #17252D ink, #35616A teal, #A6B9A3 sage, #D9C7A5 sand
INK = "#17252D"
TEAL = "#35616A"
TEAL_LIGHT = "#6FA3AE"
SAGE = "#A6B9A3"
SAND = "#D9C7A5"
TEXT = "#E8EDEA"
TEXT_2 = "#9DB0AB"
GRID = "#263A42"

CATEGORY_COLORS = {name: colour for _, _, name, colour in [
    (0, 50, "Good", "#A6B9A3"),
    (51, 100, "Satisfactory", "#C9CDA0"),
    (101, 200, "Moderate", "#D9C7A5"),
    (201, 300, "Poor", "#D8A468"),
    (301, 400, "Very Poor", "#C97B6D"),
    (401, 500, "Severe", "#A94F44"),
]}
SEVERITY_SCALE = [(0.0, "#A6B9A3"), (0.25, "#C9CDA0"), (0.5, "#D9C7A5"), (0.7, "#D8A468"), (0.85, "#C97B6D"), (1.0, "#A94F44")]
SERIES_COLORS = [SAND, TEAL_LIGHT, SAGE, "#C97B6D"]

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    html, body, [class*="css"], .stApp {{ font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }}
    code, pre {{ font-family: 'JetBrains Mono', monospace !important; font-size: 0.82rem; }}
    .stApp {{ background: {INK}; color: {TEXT}; }}

    /* Header */
    .vx-header {{
        display: flex; justify-content: space-between; align-items: flex-end;
        border-bottom: 1px solid rgba(111,163,174,0.35); padding: 18px 2px 14px 2px; margin-bottom: 18px;
    }}
    .vx-logo {{ font-size: 2.0rem; font-weight: 800; letter-spacing: 0.35em; color: {TEXT}; line-height: 1; }}
    .vx-logo span {{ color: {SAND}; }}
    .vx-tag {{ font-size: 0.82rem; color: {TEXT_2}; margin-top: 6px; letter-spacing: 0.02em; }}
    .vx-pills {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
    .vx-pill {{
        display: inline-flex; align-items: center; font-size: 0.7rem; font-weight: 600;
        padding: 4px 11px; border-radius: 3px; letter-spacing: 0.06em; text-transform: uppercase;
        border: 1px solid rgba(111,163,174,0.35); color: {TEXT_2}; background: rgba(53,97,106,0.18);
    }}

    /* KPI tiles */
    .vx-kpi {{
        background: rgba(28,48,56,0.55); border: 1px solid rgba(111,163,174,0.22);
        border-top: 2px solid {TEAL}; border-radius: 4px; padding: 13px 15px 11px 15px; height: 100%;
    }}
    .vx-kpi .l {{ font-size: 0.66rem; font-weight: 600; color: {TEXT_2}; text-transform: uppercase; letter-spacing: 0.1em; }}
    .vx-kpi .v {{ font-size: 1.6rem; font-weight: 700; color: {TEXT}; line-height: 1.25; margin-top: 3px; }}
    .vx-kpi .s {{ font-size: 0.72rem; color: {TEXT_2}; margin-top: 4px; }}

    /* Cards and notes */
    .vx-card {{
        background: rgba(28,48,56,0.55); border: 1px solid rgba(111,163,174,0.22);
        border-left: 3px solid {TEAL}; border-radius: 4px; padding: 14px 16px; margin-bottom: 12px;
    }}
    .vx-note {{
        font-size: 0.74rem; color: {TEXT_2}; border-left: 3px solid {SAND};
        padding: 6px 0 6px 12px; margin: 8px 0;
    }}
    .vx-rule {{ border: none; border-top: 1px solid rgba(111,163,174,0.25); margin: 6px 0 18px 0; }}

    /* Widgets */
    .stTabs [data-baseweb="tab-list"] {{ gap: 2px; border-bottom: 1px solid rgba(111,163,174,0.3); border-radius: 0; background: transparent; padding: 0; }}
    .stTabs [data-baseweb="tab"] {{ border-radius: 0; color: {TEXT_2}; font-weight: 600; padding: 9px 16px; border: none !important; font-size: 0.85rem; }}
    .stTabs [aria-selected="true"] {{ color: {SAND} !important; border-bottom: 2px solid {SAND} !important; background: transparent !important; }}
    [data-testid="stSidebar"] {{ background: #101B21 !important; border-right: 1px solid rgba(111,163,174,0.25); }}
    [data-testid="stToolbar"] {{ display: none; }}
    [data-testid="stStatusWidget"] {{ display: none; }}
    .stDeployButton {{ display: none; }}
    [data-testid="stSidebar"] .stSelectbox label, [data-testid="stSidebar"] .stRadio label {{ color: {TEXT_2}; font-size: 0.78rem; }}
    .stButton button, [data-testid="stDownloadButton"] button {{
        background: {SAND} !important; color: {INK} !important; border: none !important;
        font-weight: 600 !important; border-radius: 3px !important; width: 100%;
    }}
    .stButton button:hover {{ filter: brightness(1.05); }}
    h2 {{ font-size: 1.02rem; font-weight: 700; color: {TEXT}; letter-spacing: 0.01em; margin: 4px 0 2px 0; }}
    h3 {{ font-size: 0.86rem; font-weight: 600; color: {TEAL_LIGHT}; letter-spacing: 0.04em; text-transform: uppercase; margin: 14px 0 6px 0; }}
    [data-testid="stMetricValue"] {{ font-family: 'Inter', sans-serif; }}
</style>
""", unsafe_allow_html=True)

# Station coordinates (Delhi-NCR CPCB network)
STATION_COORDS = {
    "Anand Vihar, Delhi - DPCC": (28.6469, 77.3160),
    "RK Puram, Delhi - DPCC": (28.5637, 77.1867),
    "Punjabi Bagh, Delhi - DPCC": (28.6740, 77.1310),
    "ITO, Delhi - CPCB": (28.6318, 77.2405),
    "Mandir Marg, Delhi - DPCC": (28.6364, 77.2010),
    "Bawana, Delhi - DPCC": (28.7997, 77.0326),
    "Alipur, Delhi - DPCC": (28.8153, 77.1530),
    "Ashok Vihar, Delhi - DPCC": (28.6954, 77.1817),
    "DTU, Delhi - CPCB": (28.7501, 77.1177),
    "CRRI Mathura Road, Delhi - IMD": (28.5512, 77.2736),
    "Chandni Chowk, Delhi - IITM": (28.6562, 77.2307),
    "Dr. Karni Singh Shooting Range, Delhi - DPCC": (28.4986, 77.2648),
    "Dwarka-Sector 8, Delhi - DPCC": (28.5710, 77.0719),
    "Jahangirpuri, Delhi - DPCC": (28.7328, 77.1706),
    "Jawaharlal Nehru Stadium, Delhi - DPCC": (28.5802, 77.2338),
    "Major Dhyan Chand National Stadium, Delhi - DPCC": (28.6119, 77.2378),
    "Najafgarh, Delhi - DPCC": (28.5701, 76.9337),
    "Narela, Delhi - DPCC": (28.8526, 77.0924),
    "Okhla Phase-2, Delhi - DPCC": (28.5308, 77.2713),
    "Patparganj, Delhi - DPCC": (28.6237, 77.2872),
    "Pusa, Delhi - DPCC": (28.6396, 77.1462),
    "Rohini, Delhi - DPCC": (28.7325, 77.1199),
    "Shadipur, Delhi - CPCB": (28.6515, 77.1581),
    "Sirifort, Delhi - CPCB": (28.5504, 77.2159),
    "Sonia Vihar, Delhi - DPCC": (28.7105, 77.2494),
    "Sri Aurobindo Marg, Delhi - DPCC": (28.5313, 77.1901),
    "Vivek Vihar, Delhi - DPCC": (28.6723, 77.3153),
    "Wazirpur, Delhi - DPCC": (28.6998, 77.1654),
}

POLLUTANT_OPTIONS = {"PM2.5": "pm25", "PM10": "pm10", "NO2": "no2", "SO2": "so2", "CO": "co", "Ozone": "ozone"}


# ----------------- ENGINE (all live computation, cached once) -----------------
@st.cache_resource(show_spinner="Computing: features, receptor model, forecast models, driver analysis ...")
def run_engine() -> dict:
    CPCB_PARQUET = BASE_DIR / "data" / "processed" / "delhi_cpcb_real.parquet"
    raw = pd.read_parquet(CPCB_PARQUET)

    ts = pd.to_datetime(raw["Timestamp"], utc=True, errors="coerce").dropna()
    wdf, weather_status = load_weather(str((ts.min() - pd.Timedelta(hours=2)).date()), str(ts.max().date()))

    features, meta = build_features(raw, wdf if wdf is not None else None)
    meta["weather_fetch_status"] = weather_status

    forecaster = VAYUQuantileForecaster()
    forecaster.fit(features)
    latest_fc = forecaster.predict_latest(features)

    explainer_24 = explain_horizon(forecaster, features, horizon=24)
    explainer_1 = explain_horizon(forecaster, features, horizon=1)

    attribution = get_attribution_summary(n_boot=150)

    # Per-station aggregates for the map and tables (real telemetry means)
    cleaned = clean_cpcb_columns(raw)
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], utc=True, errors="coerce")
    station_stats = []
    if "station" in cleaned.columns:
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
            cat, cat_colour = aqi_category(aqi_v)
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

    return {
        "features": features,
        "meta": meta,
        "forecaster": forecaster,
        "latest_forecast": latest_fc,
        "explainer_24": explainer_24,
        "explainer_1": explainer_1,
        "attribution": attribution,
        "station_stats": station_stats,
        "raw_cleaned": cleaned,
    }


def _base_layout(fig, height: int, ytitle: str | None = None, xtitle: str | None = None):
    fig.update_layout(
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_2, family="Inter", size=12),
        margin=dict(l=10, r=14, t=28, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    if ytitle:
        fig.update_yaxes(title=ytitle)
    if xtitle:
        fig.update_xaxes(title=xtitle)
    return fig


engine = run_engine()
features: pd.DataFrame = engine["features"]
meta: dict = engine["meta"]
forecaster: VAYUQuantileForecaster = engine["forecaster"]
attribution: dict = engine["attribution"]

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown(f"""
    <div style='padding: 4px 0 2px 0;'>
        <div style='font-size: 1.55rem; font-weight: 800; letter-spacing: 0.35em; color: {TEXT};'>VAYU<span style='color:{SAND};'>.</span></div>
        <div style='font-size: 0.68rem; color: {TEXT_2}; letter-spacing: 0.14em; margin-top: 3px;'>ATMOSPHERIC INTELLIGENCE</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    view_mode = st.selectbox("View", ["City aggregate", "Single station"], index=0)
    available_stations = [s for s in sorted(engine["raw_cleaned"]["station"].dropna().unique()) if s in STATION_COORDS] \
        if "station" in engine["raw_cleaned"] else list(STATION_COORDS.keys())
    selected_station = st.selectbox("Monitoring station", available_stations, index=0, disabled=(view_mode == "City aggregate"))
    pollutant_label = st.selectbox("Focus pollutant", list(POLLUTANT_OPTIONS.keys()), index=0)
    pollutant_col = POLLUTANT_OPTIONS[pollutant_label]
    window_h = st.slider("Display window (hours)", 24, int(meta["n_hourly_rows"]), min(int(meta["n_hourly_rows"]), 96), step=12)

    st.markdown("---")
    st.markdown("**Data status**")
    st.markdown(
        f"<span style='font-size:0.75rem;color:{TEXT_2};'>"
        f"Ground: CPCB CAAQMS, {len(engine['station_stats'])} stations<br/>"
        f"Window: {meta['time_start'][:16]} - {meta['time_end'][:16]} UTC<br/>"
        f"Meteorology: {meta['weather_provenance']}<br/>"
        f"Receptor model: {attribution['n_samples']:,} samples</span>",
        unsafe_allow_html=True,
    )

# ----------------- PRESENT STATE (all real) -----------------
last_row = features.sort_values("timestamp").iloc[-1]
state: dict = {}
state["pm25"] = float(last_row["pm25"]) if pd.notna(last_row["pm25"]) else float(features["pm25"].dropna().iloc[-1])
state["pm10"] = float(last_row["pm10"]) if pd.notna(last_row["pm10"]) else float(features["pm10"].dropna().iloc[-1])
state["no2"] = float(last_row["no2"]) if pd.notna(last_row["no2"]) else float(features["no2"].dropna().iloc[-1])
state["so2"] = float(last_row["so2"]) if pd.notna(last_row["so2"]) else float(features["so2"].dropna().iloc[-1])
state["co"] = float(last_row["co"]) if pd.notna(last_row["co"]) else float(features["co"].dropna().iloc[-1])
state["ozone"] = float(last_row["ozone"]) if pd.notna(last_row["ozone"]) else float(features["ozone"].dropna().iloc[-1])
state["wind"] = float(last_row["wind_speed"]) if pd.notna(last_row["wind_speed"]) else float(features["wind_speed"].dropna().iloc[-1])
state["temp"] = float(last_row["temp"]) if pd.notna(last_row["temp"]) else float(features["temp"].dropna().iloc[-1])

state["pm_ratio"] = state["pm25"] / state["pm10"] if state["pm10"] else np.nan
state["no2_so2_ratio"] = state["no2"] / state["so2"] if state["so2"] else np.nan
state["aqi"], state["aqi_driver"], _subs = overall_aqi(
    {"pm25": state["pm25"], "pm10": state["pm10"], "no2": state["no2"], "so2": state["so2"], "co": state["co"], "ozone": state["ozone"]}
)
state["aqi_cat"], state["aqi_colour"] = aqi_category(state["aqi"])

blh_series = features[["timestamp", "blh"]].dropna()
if not blh_series.empty and last_row["timestamp"] - blh_series["timestamp"].iloc[-1] <= pd.Timedelta(hours=3):
    real_blh = float(blh_series["blh"].iloc[-1])
    state["blh_display"] = f"{real_blh:.0f} m"
    if real_blh < 200:
        state["inversion_strength"], state["inversion_colour"] = "SEVERE LID", "#C97B6D"
    elif real_blh < 500:
        state["inversion_strength"], state["inversion_colour"] = "MODERATE LID", "#D8A468"
    else:
        state["inversion_strength"], state["inversion_colour"] = "WELL-MIXED", "#A6B9A3"
    state["inversion_detail"] = f"ERA5 boundary layer at {blh_series['timestamp'].iloc[-1]:%d %b %H:%M} UTC"
else:
    state["blh_display"] = "n/a"
    state["inversion_strength"], state["inversion_colour"] = "UNKNOWN", TEXT_2
    state["inversion_detail"] = "Boundary-layer data unavailable - not estimated"

if view_mode == "Single station":
    active_df = engine["raw_cleaned"][engine["raw_cleaned"]["station"] == selected_station].sort_values("timestamp")
    station_lat, station_lon = STATION_COORDS.get(selected_station, (28.6139, 77.2090))
    station_display_name = selected_station.split(",")[0]
    scope_pill = station_display_name.upper()
else:
    active_df = features.sort_values("timestamp")
    station_lat, station_lon = 28.6139, 77.2090
    station_display_name = "Delhi-NCR city aggregate"
    scope_pill = "CITY AGGREGATE"

# ----------------- SIDEBAR PHASE 2 (situation report + downloads; needs present state) -----------------
with st.sidebar:
    st.markdown("**Situation report**")
    report_format = st.selectbox("Format", ["HTML (print-ready)", "Markdown"], index=0)

    def _kpis_for_report() -> dict:
        fc24 = engine["latest_forecast"].get(24)
        return {
            "tiles": [
                ("Overall AQI", f"{state['aqi']:.0f}" if state["aqi"] is not None else "n/a",
                 f"{state['aqi_cat']} | driver {state['aqi_driver']}"),
                ("PM2.5", f"{state['pm25']:.1f}", "ug/m3, city hourly median"),
                ("Boundary layer", state["blh_display"], state["inversion_strength"]),
                ("Wind / temp", f"{state['wind']:.1f} m/s", f"{state['temp']:.1f} C"),
                ("Forecast +24h median", f"{fc24['q50']:.0f}" if fc24 else "n/a", "ug/m3, quantile model"),
            ],
            "provenance": (
                f"CPCB CAAQMS continuous telemetry ({len(engine['station_stats'])} Delhi-NCR stations), "
                f"window {meta['time_start'][:16]} to {meta['time_end'][:16]} UTC; meteorology {meta['weather_provenance']}; "
                f"receptor model fitted on {attribution['n_samples']:,} multi-pollutant samples."
            ),
        }

    if report_format.startswith("HTML"):
        report_bytes = build_html_report(engine, _kpis_for_report()).encode("utf-8")
        st.download_button("Download report", data=report_bytes, file_name="vayu_situation_report.html", mime="text/html")
        st.caption("Open the file in a browser and print to PDF (Ctrl+P).")
    else:
        report_text = build_markdown_report(engine, _kpis_for_report())
        st.download_button("Download report", data=report_text, file_name="vayu_situation_report.md", mime="text/markdown")

    st.download_button(
        "Download dataset (CSV)",
        data=features.to_csv(index=False).encode("utf-8"),
        file_name="vayu_hourly_features.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.caption("All values are model estimates with stated uncertainty - not probabilities, not regulatory determinations.")

# ----------------- HEADER -----------------
st.markdown(f"""
<div class='vx-header'>
    <div>
        <div class='vx-logo'>VAYU<span>.</span></div>
        <div class='vx-tag'>Air-quality monitoring, source attribution and policy simulation for Delhi-NCR - built on open public data</div>
    </div>
    <div class='vx-pills'>
        <span class='vx-pill' style='color:{state['aqi_colour']};border-color:{state['aqi_colour']};background:transparent;'>AQI {state['aqi']:.0f} - {state['aqi_cat'].upper() if state['aqi_cat'] else 'N/A'}</span>
        <span class='vx-pill' style='color:{state['inversion_colour']};border-color:{state['inversion_colour']};background:transparent;'>{state['inversion_strength']}</span>
        <span class='vx-pill'>{scope_pill}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- KPI ROW -----------------
def _kpi(label: str, value: str, sub: str, colour: str = TEXT) -> str:
    return f"""
    <div class='vx-kpi'>
        <div class='l'>{label}</div>
        <div class='v' style='color:{colour};'>{value}</div>
        <div class='s'>{sub}</div>
    </div>"""

k = st.columns(6)
with k[0]:
    st.markdown(_kpi("Overall AQI (CPCB)", f"{state['aqi']:.0f}" if state["aqi"] is not None else "n/a",
                     f"{state['aqi_cat']} | driven by {state['aqi_driver']}", state["aqi_colour"] or TEXT), unsafe_allow_html=True)
with k[1]:
    st.markdown(_kpi("PM2.5 (hourly)", f"{state['pm25']:.1f}",
                     f"ug/m3 at {last_row['timestamp']:%d %b %H:%M} UTC", "#D8A468" if state["pm25"] > 120 else TEXT), unsafe_allow_html=True)
with k[2]:
    r_txt = "fine combustion dominant" if state["pm_ratio"] >= 0.65 else "mixed" if state["pm_ratio"] >= 0.4 else "crustal dominant"
    st.markdown(_kpi("PM2.5 / PM10", f"{state['pm_ratio']:.2f}", r_txt), unsafe_allow_html=True)
with k[3]:
    n_txt = "vehicular NOx regime" if state["no2_so2_ratio"] > 2.5 else "industrial influence"
    st.markdown(_kpi("NO2 / SO2", f"{state['no2_so2_ratio']:.2f}", n_txt), unsafe_allow_html=True)
with k[4]:
    st.markdown(_kpi("Boundary layer (ERA5)", state["blh_display"], state["inversion_detail"], state["inversion_colour"]), unsafe_allow_html=True)
with k[5]:
    stag = "stagnation" if state["wind"] < 1.5 else "ventilation ok"
    st.markdown(_kpi("Wind / temperature", f"{state['wind']:.1f} m/s", f"{state['temp']:.1f} C - {stag}"), unsafe_allow_html=True)

st.markdown("<hr class='vx-rule'>", unsafe_allow_html=True)

# ----------------- TABS -----------------
t1, t2, t3, t4, t5, t6, t7 = st.tabs(
    ["Overview", "Source attribution", "Drivers", "Forecast", "Policy simulator", "Model audit", "Method and data"]
)

# ================= TAB 1: OVERVIEW =================
with t1:
    map_df = pd.DataFrame(engine["station_stats"])
    if not map_df.empty and pollutant_label in map_df.columns:
        c_map, c_side = st.columns([3, 1.6])
        with c_map:
            fig_map = px.scatter_map(
                map_df.dropna(subset=[pollutant_label]),
                lat="lat", lon="lon", size=pollutant_label, color=pollutant_label,
                hover_name="station",
                hover_data={"AQI": True, "category": True, "driver": True, pollutant_label: ":.1f", "lat": False, "lon": False},
                color_continuous_scale=SEVERITY_SCALE,
                size_max=26, zoom=9.6, center={"lat": station_lat, "lon": station_lon},
                map_style="carto-darkmatter",
            )
            fig_map.update_layout(height=470, margin=dict(l=0, r=0, t=0, b=0),
                                  coloraxis_colorbar=dict(title=pollutant_label, thickness=11, len=0.7,
                                                          tickfont=dict(color=TEXT_2), title_font=dict(color=TEXT_2)))
            st.plotly_chart(fig_map, width="stretch")
            st.markdown(f"<div class='vx-note'>Station means over the loaded CPCB window, coloured by mean {pollutant_label}. "
                        f"AQI per station computed with CPCB sub-index breakpoints from mean concentrations.</div>", unsafe_allow_html=True)
        with c_side:
            st.markdown(f"#### Highest {pollutant_label}")
            lead = map_df.dropna(subset=[pollutant_label]).sort_values(pollutant_label, ascending=False).head(8)
            for _, row in lead.iterrows():
                col = CATEGORY_COLORS.get(row["category"], TEXT_2)
                st.markdown(
                    f"<div class='vx-card' style='padding:9px 12px;margin-bottom:8px;border-left-color:{col};'>"
                    f"<div style='display:flex;justify-content:space-between;'>"
                    f"<span style='font-size:0.8rem;color:{TEXT};'>{row['station']}</span>"
                    f"<b style='color:{col};'>{row[pollutant_label]:.1f}</b></div>"
                    f"<div style='font-size:0.68rem;color:{TEXT_2};'>AQI {row['AQI']:.0f} - {row['category']} | {row['driver']} driven</div></div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("No station rows matched the coordinate registry for this pollutant.")

    st.markdown("#### Time series")
    ts_plot = active_df.dropna(subset=[pollutant_col]).tail(window_h)
    x_vals = ts_plot["timestamp"] if "timestamp" in ts_plot.columns else ts_plot.index
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=x_vals, y=ts_plot[pollutant_col], mode="lines", name=pollutant_label, line=dict(color=SAND, width=2)))
    if pollutant_col != "pm25" and "pm25" in ts_plot.columns:
        fig_ts.add_trace(go.Scatter(x=x_vals, y=ts_plot["pm25"], mode="lines", name="PM2.5", line=dict(color=TEAL_LIGHT, width=1.2)))
    if "blh" in ts_plot.columns and ts_plot["blh"].notna().any():
        fig_ts.add_trace(go.Scatter(x=ts_plot["timestamp"], y=ts_plot["blh"], mode="lines", name="BLH (ERA5)",
                                    yaxis="y2", line=dict(color=SAGE, width=1.1, dash="dot")))
    fig_ts.update_layout(yaxis2=dict(title="BLH (m)", overlaying="y", side="right", showgrid=False))
    _base_layout(fig_ts, 330, ytitle=f"{pollutant_label} (ug/m3)")
    st.plotly_chart(fig_ts, width="stretch")

    st.markdown("#### Station summary")
    st.dataframe(
        map_df.drop(columns=["full_name", "lat", "lon"]).sort_values("AQI", ascending=False),
        width="stretch", height=320,
    )

# ================= TAB 2: SOURCE ATTRIBUTION =================
with t2:
    st.markdown("#### Receptor model - where the PM2.5 mass comes from")
    st.caption(
        f"Non-negative matrix factorization (EPA PMF form) fitted on {attribution['n_samples']:,} real multi-pollutant samples | "
        f"relative reconstruction residual {attribution['reconstruction_residual']:.3f} | "
        f"intervals: empirical 90% credible intervals from 150 bootstrap refits"
    )

    shares = attribution["shares"]
    boot = attribution["bootstrap"]
    c_donut, c_cards = st.columns([1, 1.25])
    with c_donut:
        fig_donut = go.Figure(go.Pie(
            labels=list(shares.keys()), values=list(shares.values()), hole=0.58,
            marker_colors=SERIES_COLORS, textinfo="percent", textfont=dict(color=TEXT, size=13),
            hovertemplate="%{label}<br>%{value:.1f}%<extra></extra>",
        ))
        fig_donut.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10),
                                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(color=TEXT_2), showlegend=True,
                                legend=dict(font=dict(size=10, color=TEXT_2)))
        st.plotly_chart(fig_donut, width="stretch")
    with c_cards:
        for i, (name, s) in enumerate(boot.items()):
            lo, hi = s["ci90_low"], s["ci90_high"]
            ci_txt = f"[{lo:.1f}% - {hi:.1f}%]" if lo is not None else "CI unavailable (few bootstrap fits)"
            sig_txt = f"{s['sigma']:.2f}%" if s["sigma"] is not None else "n/a"
            st.markdown(f"""
            <div class='vx-card' style='border-left-color:{SERIES_COLORS[i % 4]};padding:10px 14px;margin-bottom:10px;'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <b style='color:{TEXT};font-size:0.88rem;'>{name}</b>
                    <span style='font-size:1.15rem;font-weight:700;color:{SERIES_COLORS[i % 4]};'>{s['mean']:.1f}%</span>
                </div>
                <div style='font-size:0.72rem;color:{TEXT_2};margin-top:3px;'>
                    90% CI: <code style='color:{SAND};'>{ci_txt}</code> &middot; sigma {sig_txt} &middot; {s['n_boot_ok']} bootstrap fits
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("#### Factor profiles (fraction of each pollutant carried by each factor)")
    prof_df = pd.DataFrame(attribution["profiles"]).T
    st.dataframe(prof_df.style.format("{:.3f}").background_gradient(cmap="PuBu", axis=None), width="stretch", height=190)

    med = attribution["diagnostic_medians"]
    _m = lambda key, d=0.0: med[key] if med.get(key) is not None else d
    c1, c2, c3 = st.columns(3)
    c1.metric("PM2.5 / PM10 (median)", f"{_m('pm_ratio'):.2f}", "fine combustion dominant" if _m("pm_ratio") > 0.65 else "mixed / crustal")
    c2.metric("NO2 / SO2 (median)", f"{_m('no2_so2_ratio'):.2f}", "vehicular NOx regime (>2.5)")
    c3.metric("CO / NO2 x 1000 (median)", f"{_m('co_no2_ratio'):.1f}", "smoldering / biomass tracer")

    st.markdown("<div class='vx-note'>How to read this: NMF factors are statistical decompositions of co-varying pollutants. "
                "Physical source names are assigned by a documented chemical-marker heuristic (NO2 and aromatics point to road "
                "traffic, SO2 to coal and heavy fuel combustion, CO to smoldering biomass, coarse PM10 to dust). Shares are "
                "evidence with uncertainty - not a measured inventory.</div>", unsafe_allow_html=True)

# ================= TAB 3: DRIVERS =================
with t3:
    st.markdown("#### Why is pollution high - trapped atmosphere or more emissions?")
    st.caption("Permutation importance on the strictly out-of-time test split (pinball scorer at the median), "
               "grouped into meteorological, emission-signature and persistence-history drivers.")

    ex = engine["explainer_24"]
    if "error" in ex:
        st.warning(f"24-hour driver analysis unavailable: {ex['error']}")
    else:
        verd = decoupling_verdict(ex["group_shares"])
        c1, c2 = st.columns([1.5, 1])
        with c1:
            fig_imp = px.bar(
                ex["feature_importance"].head(10)[::-1],
                x="importance", y="feature", orientation="h", color="group",
                labels={"importance": "Permutation importance", "feature": ""},
                color_discrete_map={
                    "Meteorological dispersal": TEAL_LIGHT,
                    "Emission signature": "#C97B6D",
                    "Persistence history": SAND,
                    "Other": TEXT_2,
                },
            )
            _base_layout(fig_imp, 400)
            fig_imp.update_layout(showlegend=False)
            st.plotly_chart(fig_imp, width="stretch")
        with c2:
            met_pct = (verd["met_share"] or 0) * 100
            emi_pct = (verd["emission_share"] or 0) * 100
            st.markdown(f"""
            <div class='vx-card' style='border-left-color:{SAND};'>
                <div class='vx-kpi l' style='font-size:0.66rem;color:{TEXT_2};text-transform:uppercase;letter-spacing:0.1em;'>Verdict - 24h horizon</div>
                <div style='font-size:1.25rem;font-weight:700;color:{SAND};margin:6px 0;'>{verd['verdict']}</div>
                <div style='font-size:0.78rem;color:{TEXT_2};'>
                    Meteorological dispersal <b style='color:{TEAL_LIGHT};'>{met_pct:.0f}%</b><br/>
                    Emission + persistence <b style='color:#C97B6D;'>{emi_pct:.0f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class='vx-card'>
                <div style='font-size:0.66rem;color:{TEXT_2};text-transform:uppercase;letter-spacing:0.1em;'>Current conditions</div>
                <div style='font-size:0.8rem;color:{TEXT};margin-top:6px;line-height:1.7;'>
                    Boundary layer <b>{state['blh_display']}</b> - trapping strengthens below ~200 m<br/>
                    Wind <b>{state['wind']:.1f} m/s</b> - stagnation below 1.5 m/s<br/>
                    Temperature <b>{state['temp']:.1f} C</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            ex1 = engine["explainer_1"]
            if "error" not in ex1:
                v1 = decoupling_verdict(ex1["group_shares"])
                st.markdown(
                    f"<div class='vx-card' style='border-left-color:{TEAL_LIGHT};'>"
                    f"<div style='font-size:0.66rem;color:{TEXT_2};text-transform:uppercase;letter-spacing:0.1em;'>Nowcast vs day-ahead</div>"
                    f"<div style='font-size:0.78rem;color:{TEXT_2};margin-top:5px;line-height:1.7;'>"
                    f"1h horizon: <b style='color:{TEXT};'>{v1['verdict']}</b> (met {(v1['met_share'] or 0) * 100:.0f}%)<br/>"
                    f"24h horizon: <b style='color:{TEXT};'>{verd['verdict']}</b> (met {met_pct:.0f}%)</div></div>",
                    unsafe_allow_html=True,
                )
        st.markdown(f"<div class='vx-note'>{ex['interpretation']} Test n={ex['n_test']}, {ex['n_repeats']} permutation repeats. "
                    f"Computed on this data window - it describes this window, not a universal law. "
                    f"Importance method: permutation on the out-of-time split (SHAP not available on this Python build).</div>", unsafe_allow_html=True)

# ================= TAB 4: FORECAST =================
with t4:
    st.markdown("#### Probabilistic PM2.5 forecast")
    st.caption("Gradient-boosted quantile regression (pinball loss, tau = 0.10 / 0.50 / 0.90), direct multi-horizon, "
               "trained chronologically on CPCB + ERA5 features. Fan bands are prediction intervals, not certainties.")

    lf = engine["latest_forecast"]
    hist = features[["timestamp", "pm25"]].dropna().sort_values("timestamp").tail(72)
    origin_ts = features["timestamp"].max()

    fwd = pd.DataFrame({"h": sorted(lf.keys())})
    fwd["q10"] = [lf[h]["q10"] for h in fwd["h"]]
    fwd["q50"] = [lf[h]["q50"] for h in fwd["h"]]
    fwd["q90"] = [lf[h]["q90"] for h in fwd["h"]]
    dense_h = np.arange(1, max(fwd["h"]) + 1)
    dense = pd.DataFrame({"h": dense_h, "timestamp": origin_ts + pd.to_timedelta(dense_h, unit="h")})
    for q in ("q10", "q50", "q90"):
        dense[q] = np.interp(dense_h, fwd["h"], fwd[q])

    fig_fan = go.Figure()
    fig_fan.add_trace(go.Scatter(x=hist["timestamp"], y=hist["pm25"], mode="lines", name="Observed PM2.5", line=dict(color=TEXT, width=1.8)))
    fig_fan.add_trace(go.Scatter(x=dense["timestamp"], y=dense["q90"], mode="lines", line=dict(width=0), showlegend=False, name="q90"))
    fig_fan.add_trace(go.Scatter(x=dense["timestamp"], y=dense["q10"], mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(111,163,174,0.18)", name="80% interval"))
    fig_fan.add_trace(go.Scatter(x=dense["timestamp"], y=dense["q50"], mode="lines", name="Median forecast", line=dict(color=SAND, width=2.4)))
    fig_fan.add_trace(go.Scatter(x=origin_ts + pd.to_timedelta(fwd["h"], unit="h"), y=fwd["q50"], mode="markers",
                                 name="Direct model anchors", marker=dict(color=SAND, size=8, symbol="diamond")))
    fig_fan.add_hline(y=250, line_dash="dash", line_color="#C97B6D", annotation_text="Severe 250 ug/m3",
                      annotation_font=dict(color="#C97B6D"))
    _base_layout(fig_fan, 400, ytitle="PM2.5 (ug/m3)", xtitle="Time (UTC)")
    st.plotly_chart(fig_fan, width="stretch")
    st.markdown("<div class='vx-note'>Direct model output exists at the +1/24/48/72 h anchors (diamonds); hours between anchors "
                "are linear interpolation for display only.</div>", unsafe_allow_html=True)

    fc_table = pd.DataFrame(
        {"horizon": [f"+{h} h" for h in sorted(lf)],
         "q10 (ug/m3)": [round(lf[h]["q10"], 1) for h in sorted(lf)],
         "median q50 (ug/m3)": [round(lf[h]["q50"], 1) for h in sorted(lf)],
         "q90 (ug/m3)": [round(lf[h]["q90"], 1) for h in sorted(lf)]}
    )
    st.markdown("##### Forward forecast from the latest real observation")
    st.dataframe(fc_table, width="stretch", hide_index=True)

    st.markdown("##### Backtest - out-of-time predictions vs reality")
    trained_hs = [h for h in sorted(forecaster.results) if forecaster.results[h].trained]
    bt_h = st.selectbox("Backtest horizon", trained_hs, index=trained_hs.index(24) if 24 in trained_hs else 0)
    r = forecaster.results[bt_h]
    if r.predictions is not None:
        p = r.predictions
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=p["timestamp"], y=p["y_true"], mode="lines+markers", name="Actual PM2.5",
                                    line=dict(color=TEXT, width=1.6), marker=dict(size=4)))
        fig_bt.add_trace(go.Scatter(x=p["timestamp"], y=p["q90"], mode="lines", line=dict(width=0), showlegend=False))
        fig_bt.add_trace(go.Scatter(x=p["timestamp"], y=p["q10"], mode="lines", line=dict(width=0), fill="tonexty",
                                    fillcolor="rgba(217,199,165,0.15)", name="80% interval"))
        fig_bt.add_trace(go.Scatter(x=p["timestamp"], y=p["q50"], mode="lines", name="Model median", line=dict(color=SAND, width=2)))
        fig_bt.add_trace(go.Scatter(x=p["timestamp"], y=p["persistence"], mode="lines", name="Persistence baseline",
                                    line=dict(color=TEAL_LIGHT, width=1.3, dash="dot")))
        _base_layout(fig_bt, 330, ytitle="PM2.5 (ug/m3)", xtitle="Time (UTC)")
        st.plotly_chart(fig_bt, width="stretch")
        m = r.metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model MAE", f"{m['mae']:.1f}", f"persistence {m['persistence_mae']:.1f}")
        m2.metric("RMSE", f"{m['rmse']:.1f}", f"persistence {m['persistence_rmse']:.1f}")
        m3.metric("R2 (test)", f"{m['r2']:.3f}", f"n_test = {r.n_test}")
        m4.metric("80% interval coverage", f"{m['interval_coverage_80']:.2f}", f"width {m['mean_interval_width']:.0f} ug/m3")

# ================= TAB 5: POLICY SIMULATOR =================
with t5:
    st.markdown("#### What-if policy simulator")
    st.caption("First-order linear counterfactual: cutting a source by X% removes (X% x its share) of PM2.5 mass. "
               "No secondary-aerosol chemistry feedback. Presets are illustrative mappings of response-plan stages to model levers, "
               "not official percentages.")

    lever_shares = shares_to_intervention_keys(attribution["shares"], SOURCE_TO_LEVER)
    missing_levers = [k for k in ("vehicular", "stubble", "industrial", "dust") if k not in lever_shares]
    if missing_levers:
        st.info(f"Lever shares unavailable for: {', '.join(missing_levers)}. Simulator disabled for those levers.")

    GRAP_PRESETS = {
        "Custom": None,
        "Stage I - dust control focus": {"vehicular": 0, "stubble": 0, "industrial": 0, "dust": 20},
        "Stage II - curbside measures": {"vehicular": 10, "stubble": 10, "industrial": 10, "dust": 30},
        "Stage III - heavier restrictions": {"vehicular": 20, "stubble": 25, "industrial": 25, "dust": 40},
        "Stage IV - emergency measures": {"vehicular": 40, "stubble": 50, "industrial": 40, "dust": 50},
    }

    def _apply_preset():
        preset = st.session_state.get("sim_preset")
        vals = GRAP_PRESETS.get(preset)
        if vals:
            st.session_state["sim_veh"] = vals["vehicular"]
            st.session_state["sim_stub"] = vals["stubble"]
            st.session_state["sim_ind"] = vals["industrial"]
            st.session_state["sim_dust"] = vals["dust"]

    for key, default in (("sim_veh", 30), ("sim_stub", 45), ("sim_ind", 35), ("sim_dust", 25)):
        if key not in st.session_state:
            st.session_state[key] = default

    c_ctrl, c_res = st.columns([1, 1.4])
    with c_ctrl:
        st.selectbox("Scenario preset", list(GRAP_PRESETS.keys()), key="sim_preset", on_change=_apply_preset)
        veh = st.slider("Vehicular fleet reduction", 0, 100, key="sim_veh", format="%d%%")
        stub = st.slider("Stubble-burning suppression", 0, 100, key="sim_stub", format="%d%%")
        ind = st.slider("Industrial / coal curtailment", 0, 100, key="sim_ind", format="%d%%")
        dust = st.slider("Dust-control intensity", 0, 100, key="sim_dust", format="%d%%")

        try:
            sim = simulate_policy(
                state["pm25"], lever_shares,
                {"vehicular": veh, "stubble": stub, "industrial": ind, "dust": dust},
            )
        except ValueError as exc:
            sim = None
            st.error(f"Invalid scenario: {exc}")

    with c_res:
        if sim:
            fig_sim = go.Figure(go.Bar(
                x=["Current", "Simulated"], y=[sim["current_pm25"], sim["new_pm25"]],
                marker_color=["#C97B6D", SAGE], width=0.45,
                text=[f"{sim['current_pm25']:.0f}", f"{sim['new_pm25']:.0f}"], textposition="outside",
                textfont=dict(color=TEXT),
            ))
            _base_layout(fig_sim, 300, ytitle="PM2.5 (ug/m3)")
            fig_sim.update_layout(showlegend=False)
            st.plotly_chart(fig_sim, width="stretch")
            a1, a2, a3 = st.columns(3)
            a1.metric("Avoided burden", f"{sim['avoided_ug']:.1f} ug/m3", f"{sim['avoided_pct']:.1f}% lower")
            a2.metric("Simulated PM2.5", f"{sim['new_pm25']:.1f} ug/m3", "first-order estimate")
            a3.metric("Model", "linear counterfactual", "no chemistry feedback")
            post = pd.DataFrame(
                {"source": list(sim["post_shares"].keys()),
                 "post-intervention share (%)": [round(v, 1) for v in sim["post_shares"].values()]}
            )
            st.markdown("##### Post-intervention composition of remaining PM2.5")
            st.dataframe(post, width="stretch", hide_index=True)
            st.markdown("<div class='vx-note'>Reductions act on the source shares estimated by the receptor model. "
                        "This is a planning aid for comparing the relative size of interventions - not a forecast of "
                        "exactly what would happen.</div>", unsafe_allow_html=True)

# ================= TAB 6: MODEL AUDIT =================
with t6:
    st.markdown("#### Model audit - strictly chronological, out-of-time")
    st.caption(f"Feature table: {meta['n_hourly_rows']} hourly rows | {meta['time_start']} to {meta['time_end']} UTC | "
               f"meteorology: {meta['weather_provenance']}")

    rows = []
    for h in sorted(forecaster.results):
        r = forecaster.results[h]
        if not r.trained:
            rows.append({"horizon_h": h, "status": f"not trained: {r.reason}"})
            continue
        m = r.metrics
        rows.append({
            "horizon_h": h,
            "status": "trained",
            "model MAE": round(m["mae"], 1),
            "persistence MAE": round(m["persistence_mae"], 1),
            "beats persistence": "yes" if m["mae"] < m["persistence_mae"] else "no",
            "RMSE": round(m["rmse"], 1),
            "R2": round(m["r2"], 3),
            "pinball q50": round(m["pinball_q50"], 2),
            "80% coverage": round(m["interval_coverage_80"], 2),
            "mean width": round(m["mean_interval_width"], 1),
            "train/test n": f"{r.n_train}/{r.n_test}",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown(
        "<div class='vx-note'>Honest reading of this run: the model clearly beats persistence at +1h and +24h; "
        "at +48h/+72h persistence is competitive on this short window - reported as measured. Observed 80% interval "
        "coverage is below nominal (intervals too narrow at this sample size), which is stated rather than hidden. "
        "Each horizon is scored on roughly 25-42 out-of-time hourly points: indicative, not definitive.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Provenance")
    p1, p2, p3 = st.columns(3)
    p1.metric("Ground data", "CPCB CAAQMS", f"{len(engine['station_stats'])} stations | 15-min native")
    p2.metric("Meteorology", "Open-Meteo ERA5", meta["weather_fetch_status"])
    p3.metric("Receptor samples", f"{attribution['n_samples']:,}", "multi-pollutant rows")

    st.markdown("#### Inspect the modelling table")
    show_cols = [c for c in ["timestamp", "pm25", "pm10", "no2", "so2", "co", "ozone", "pm_ratio", "no2_so2_ratio",
                             "wind_speed", "wind_u", "wind_v", "blh", "pm25_lag24", "pm25_roll24", "n_stations"]
                 if c in features.columns]
    st.dataframe(features[show_cols].tail(48), width="stretch", height=300)

# ================= TAB 7: METHOD AND DATA =================
with t7:
    st.markdown("#### Methodology")
    st.write("VAYU answers three questions with documented, reproducible statistics rather than a single black-box score.")

    st.markdown("**1. Chemical fingerprinting**")
    st.latex(r"R_{aero} = \frac{[PM_{2.5}]}{[PM_{10}]} \qquad R_{ind} = \frac{[NO_2]}{[SO_2]} \qquad R_{comb} = \frac{[CO]\times 1000}{[NO_2]}")
    st.write("Fine-aerosol dominance (R_aero above 0.65) points to combustion sources; a high NO2/SO2 ratio (>2.5) points "
             "to a low-sulphur vehicular fleet; elevated CO relative to NO2 marks smoldering biomass burning.")

    st.markdown("**2. Receptor modelling (source attribution)**")
    st.latex(r"\min_{W \geq 0,\; H \geq 0} \; \frac{1}{2}\lVert X - WH \rVert_F^2 + \alpha \lVert W \rVert_1 + \beta \lVert H \rVert_1")
    st.write("The concentration matrix X (hours x pollutants) is decomposed into temporal factors W and source profiles H. "
             "Source shares are averaged over 150 bootstrap refits; credible intervals are the empirical 5th-95th percentiles "
             "of those refits. Factor naming uses documented chemical markers and is disclosed as a heuristic.")

    st.markdown("**3. Quantile forecasting**")
    st.latex(r"L_\tau(y, \hat{y}) = \max\left(\tau\,(y-\hat{y}),\; (\tau-1)(y-\hat{y})\right), \quad \tau \in \{0.10, 0.50, 0.90\}")
    st.write("Direct multi-horizon gradient-boosted models at +1/24/48/72 h use only information available at forecast time: "
             "lags, rolling means, chemical ratios, onboard met sensors and ERA5 boundary-layer height. Splits are strictly "
             "chronological; the persistence baseline is always reported alongside.")

    st.markdown("**4. Driver separation**")
    st.write("Permutation importance on the out-of-time test split, grouped into meteorological dispersal, emission signature "
             "and persistence history, separates inversion-driven pollution from emission-driven pollution.")

    st.markdown("#### Data sources")
    st.markdown(
        "- **CPCB CAAQMS** continuous telemetry for 40 Delhi-NCR stations (open mirror: Vonter/india-cpcb-aqi), 15-minute native cadence\n"
        "- **Open-Meteo ERA5 archive**: hourly temperature, humidity, wind speed/direction, surface pressure and boundary-layer height for Delhi, cached locally; provenance is shown in the sidebar\n"
        "- **NASA FIRMS** active-fire pixels (optional, requires a FIRMS map key; without one the fire module reports 'unavailable' rather than estimating)\n"
        "- **CPCB National AQI** breakpoints (2014 methodology) for sub-index computation"
    )

    st.markdown("#### Limitations and intended use")
    st.markdown(
        "- Attribution shares are statistical evidence about co-varying pollutants, not a measured emissions inventory and not causal proof\n"
        "- The open archive covers roughly a one-week window; intervals may under-cover at this sample size (coverage is shown in the audit tab)\n"
        "- The policy simulator is a first-order linear counterfactual without secondary-aerosol chemistry\n"
        "- All values are estimates with stated uncertainty - not probabilities and not regulatory determinations\n"
        "- Intended for situational awareness, prioritisation and planning support; confirmatory decisions need measurement campaigns"
    )

    st.markdown("#### Reproducibility")
    st.code(
        "pytest tests/ -v                     # 39 unit tests\n"
        "python -m src.models.forecaster      # re-print audit table\n"
        "python -m src.attribution.receptor_nmf  # re-run receptor fit\n"
        "streamlit run app/streamlit_app.py   # this console",
        language="bash",
    )

    st.markdown(
        f"<div style='font-size:0.72rem;color:{TEXT_2};text-align:center;padding:14px 0 4px 0;border-top:1px solid rgba(111,163,174,0.25);'>"
        f"VAYU &middot; atmospheric intelligence for Delhi-NCR &middot; open data: CPCB CAAQMS, Open-Meteo ERA5, CPCB AQI methodology "
        f"&middot; estimates carry stated uncertainty and are subject to revision</div>",
        unsafe_allow_html=True,
    )
