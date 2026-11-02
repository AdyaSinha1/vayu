# VAYU: Atmospheric Intelligence & Explainable Source Attribution System
**NEURAL STORM 2026 — Track: Green Planet | Problem Statement: GP-01**

> **Tagline:** *Understand the air. Trace the source. Quantify the uncertainty.*

---

## 📌 1. Project Overview & Problem Statement
* **Track:** Green Planet
* **Problem Statement:** GP-01 — Air Quality Monitoring & Source Attribution
* **Target Region:** Indo-Gangetic Plain & Delhi-NCR Smog Belt

### The Challenge
Every winter, the Indo-Gangetic Plain suffers from a hazardous toxic smog crisis. While standard monitoring systems report high PM2.5 values, **they fail to answer critical governance questions**:
1. *Who is contributing to this pollution?* (Vehicular fleet vs. transboundary stubble burning vs. industrial combustion vs. road dust).
2. *Is a spike caused by higher emissions or by thermal inversion?* (Planetary Boundary Layer collapse trapping normal emissions).
3. *What happens if specific interventions are implemented?*

### The VAYU Solution
**VAYU** is an explainable atmospheric intelligence platform built on **100% real, open data** — CPCB CAAQMS continuous telemetry (26,895 records, 40 Delhi-NCR stations) + Open-Meteo ERA5 reanalysis (real boundary-layer height and wind vectors). Rather than merely forecasting a scalar AQI value, VAYU implements:
* **Dual-Engine Source Attribution**: chemical diagnostic fingerprinting + unsupervised **Positive Matrix Factorization (NMF)** extracting emission profiles with **bootstrap credible intervals** — no fabricated labels.
* **Quantile (probabilistic) forecasting** evaluated **strictly out-of-time**, always compared against a persistence baseline — including where the baseline wins.
* **Inversion-vs-emission decoupling**: permutation importance separates meteorological trapping from emission signal.
* **Counterfactual "What-If" policy simulator** (documented first-order linear model) with illustrative response-stage presets.
* **CPCB AQI sub-index computation** (official 2014 breakpoints with linear interpolation) per station and city-wide, with the driving pollutant reported.
* **Printable situation reports** — self-contained HTML (open and Ctrl+P to PDF) or Markdown, generated from the live engine state, plus CSV export of the modelling table.
* **Honest failure reporting**: when data is unavailable (e.g. NASA FIRMS without an API key, Open-Meteo outage), the UI says "unavailable" — it never fabricates.

---

## 🔬 2. Scientific & Mathematical Methodology

### A. Chemical Diagnostic Ratios (Fingerprinting)
1. **Aerodynamic Combustion Ratio**: $R_{aero} = \frac{[PM_{2.5}]}{[PM_{10}]}$
   * $> 0.65$: fine combustion aerosols (vehicles, biomass). **Observed real Delhi median: 0.68**.
   * $< 0.40$: coarse crustal dust, road resuspension.
2. **Mobile vs. Stationary Signature**: $R_{ind} = \frac{[NO_2]}{[SO_2]}$
   * $> 2.5$: mobile fleet (BS-VI ultra-low-sulfur). **Observed real Delhi median: 4.31**.
   * $< 1.0$: sulfur-heavy coal/industrial combustion.
3. **Smoldering tracer**: $CO/NO_2 \times 1000$, observed real median **35.6**.

### B. Positive Matrix Factorization (Receptor Modeling via NMF)
For the multi-pollutant matrix $X \in \mathbb{R}^{n \times m}$ (PM2.5, PM10, NO2, SO2, CO, O3):
$$\min_{W \ge 0,\, H \ge 0} \tfrac{1}{2}\|X - WH\|_F^2 + \alpha\|W\|_1 + \beta\|H\|_1$$
Fitted on **16,892 real multi-pollutant samples** (relative reconstruction residual **0.075**). Source shares below are **means over 150 bootstrap refits**; intervals are empirical 90% credible intervals (5th–95th percentile of bootstrap shares):

| Source (heuristic assignment) | Share | 90% CI |
|---|---|---|
| Industrial & Thermal Combustion | 32.4% | [29.8, 33.8] |
| Vehicular Fleet Exhaust | 30.0% | [28.5, 37.9] |
| Agricultural Biomass / Stubble Burning | 24.2% | [21.3, 25.7] |
| Crustal & Resuspended Dust | 13.4% | [10.8, 14.1] |

> **Honesty note:** NMF factors are statistical constructs; the mapping of factors to physical source names is a **documented chemical-marker heuristic** (NO2/benzene→vehicular, SO2→industrial, CO→biomass, coarse PM10→dust), shown in the UI. Shares are evidence scores with uncertainty — not probabilities, not measured truth.

### C. Quantile (Pinball-Loss) Forecasting
$$L_\tau(y, \hat y) = \max\big(\tau (y-\hat y),\, (\tau-1)(y-\hat y)\big), \quad \tau \in \{0.10, 0.50, 0.90\}$$
Direct multi-horizon models (HistGradientBoosting, quantile loss) predict PM2.5 at +1/24/48/72 h from features known at forecast time (lags, rolling means, chemical ratios, CPCB onboard met sensors, ERA5 BLH + u/v winds). Quantile crossing is corrected by sorting; the forecast origin is always the latest real observation.

### D. Inversion-vs-Emission Decoupling
Permutation importance on the out-of-time test split (pinball scorer at τ=0.5), grouped into *Meteorological dispersal* / *Emission signature* / *Persistence history*. On this real window the 24h horizon is **meteorological-trapping dominant (74%)** while the 1h nowcast is **emission-signal dominant (86%)** — reported per run, not asserted.

---

## 💻 3. Tech Stack & Open Data
* **Language:** Python 3.10+ (developed on 3.14)
* **Real open data:**
  * CPCB CAAQMS continuous telemetry via the open mirror `Vonter/india-cpcb-aqi` (26,895 Delhi records, 40 stations, 15-min cadence)
  * Open-Meteo ERA5 archive API (temperature, RH, wind speed/direction, surface pressure, **boundary layer height**) with a local parquet cache and provenance reporting
  * NASA FIRMS VIIRS active fires (optional: set `FIRMS_MAP_KEY`; without it VAYU reports "unavailable" instead of faking fire data)
* **ML:** scikit-learn (NMF, HistGradientBoosting quantile regressors, permutation importance)
* **Data:** pandas, NumPy, pyarrow; **Viz/UI:** Streamlit, Plotly (MapLibre dark matter basemap)
* **Tests:** pytest (39 tests: ratio math, AQI breakpoints, NMF constraints, quantile monotonicity, no-leakage split, simulator bounds)

---

## 🚀 4. How to Run Locally
```bash
# 1. Clone
git clone <repo-url> && cd vayu

# 2. Environment
python -m venv venv && .\venv\Scripts\activate   # Windows (Linux: source venv/bin/activate)
pip install -r requirements.txt

# 3. Data (real CPCB download + Delhi slice)
python src/ingestion/download_cpcb_delhi.py

# 4. Run the test suite (no network needed)
pytest tests/ -v

# 5. Launch the console (fetches ERA5 weather once; caches it; works offline afterwards)
streamlit run app/streamlit_app.py
```
Open `http://localhost:8501`. The engine computes features, NMF bootstrap, quantile models and explainability on first load, then caches them for the session.

**Offline behaviour:** with no network, VAYU uses cached weather; without any cache it proceeds on CPCB's onboard met sensors and displays BLH as "unavailable" — honestly. Every UI panel shows its data provenance.

---

## 📊 5. Validation & Performance Benchmarks (measured, not asserted)
Feature table: **169 hourly city-aggregate rows** (2025-12-24 → 2025-12-31 UTC, 40 stations). Chronological 75/25 split; metrics on the strictly later test block; persistence = "predict last observed PM2.5".

| Horizon | Model MAE | Persistence MAE | Beats baseline | R² | 80% interval coverage |
|---|---|---|---|---|---|
| +1h | **16.8 µg/m³** | 20.8 | ✅ | **0.865** | 0.33 |
| +24h | **43.1 µg/m³** | 64.9 | ✅ | 0.225 | 0.30 |
| +48h | 42.1 µg/m³ | 35.7 | ❌ persistence wins | 0.271 | 0.61 |
| +72h | 40.9 µg/m³ | 38.9 | ❌ persistence wins | 0.292 | 0.48 |

**Honest reading of this run:** VAYU clearly beats persistence at nowcast (+1h) and day-ahead (+24h) ranges; at +48h/+72h on this short window persistence is competitive — we report that rather than hide it. Observed 80% interval coverage is below nominal (intervals too narrow on 7 days of data) and is displayed in the app's audit card. With ~25–42 out-of-time test points per horizon, these numbers are indicative, not definitive.

### Known limitations (stated up front)
* The open CPCB mirror provides a ~7-day rolling snapshot; a longer archive would strengthen multi-week validation.
* Source attribution shares come from receptor modelling of co-located pollutants — statistical evidence, **not** causal measurement.
* The policy simulator is a documented **first-order linear counterfactual** (no secondary aerosol chemistry feedback).
* Permutation importance is used instead of SHAP (the `shap` package does not yet support this Python build); this is labelled in the UI.

---

## 🤖 6. AI Tools & Disclosures
Per hackathon rules, AI assistants were used: **Antigravity (Google)** for early scaffolding and ideation, and **OpenCode (GLM)** for pipeline engineering, testing and this documentation. All models are trained and evaluated on real open datasets; every metric in this README is reproducible via `pytest tests/ -v` and the app's audit card.

---

## 👥 7. Team Information
* **Team Name:** VAYU Core Team
* **Track:** Green Planet (GP-01)
* **Hackathon:** NEURAL STORM 2026
