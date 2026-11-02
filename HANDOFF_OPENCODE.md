# 🤖 VAYU: OpenCode Master Execution Blueprint & Handoff Guide

> **To the AI / OpenCode Agent:** You are taking over as the Lead AI/ML Engineer for **VAYU**. Read this document completely before modifying or writing code. It contains the complete architectural context, codebase inventory, mathematical formulations, and step-by-step roadmap to complete this project.

---

## 1. Project Context & Mission
* **Project Name:** **VAYU** (*Atmospheric Intelligence & Explainable Source Attribution System*)
* **Hackathon:** **NEURAL STORM 2026**
* **Track & Statement:** Green Planet — **GP-01: Air Quality Monitoring & Source Attribution**
* **Submission Deadline:** November 4, 2026, 8:00 PM IST (48-hour sprint)
* **Judges:** US-based AI startup engineers and atmospheric scientists.
* **Rubric Weighting:**
  - Technical Implementation: **35%** (real ML, receptor modeling, uncertainty intervals, zero leakage)
  - Impact & Real-World Usefulness: **25%** (actionable GRAP policy simulator for Delhi-NCR smog)
  - README & Documentation: **20%** (LaTeX equations, reproducible benchmarks, setup steps)
  - Demo & Presentation: **20%** (interactive Streamlit console with live simulation)

---

## 2. Current Codebase State (Existing Inventory)
The repository is located at: `C:\Users\Ananya\OneDrive\Desktop\VAYU\`

```text
vayu/
├── .streamlit/
│   └── config.toml                  # Pre-configured high-tech dark theme & port 8501
├── app/
│   └── streamlit_app.py             # Working prototype dashboard (Tabs: Hotspots, NMF, What-If, Eval)
├── data/
│   ├── raw/
│   │   └── latest-air-quality.parquet   # 100% real CPCB CAAQMS archive (~14MB, 360k rows across India)
│   └── processed/
│       ├── delhi_cpcb_real.parquet      # Filtered 26,895 real continuous monitoring records for Delhi-NCR
│       └── delhi_cpcb_real.csv          # CSV sample preview
├── src/
│   ├── ingestion/
│   │   └── download_cpcb_delhi.py   # Verified CPCB data downloader and cleaner
│   ├── attribution/
│   │   └── receptor_nmf.py          # Positive Matrix Factorization (NMF) + Diagnostic Ratios
│   └── models/                      # [NEEDS IMPLEMENTATION]
├── tests/                           # [NEEDS IMPLEMENTATION]
├── requirements.txt                 # Dependencies: streamlit, plotly, scikit-learn, pandas, pyarrow
└── README.md                        # Publication-grade overview with scientific rubric
```

---

## 3. The Core Differentiator: Why VAYU Wins
Standard hackathon teams fail because they simply predict tomorrow's AQI using a basic Random Forest and make up source attribution percentages. **VAYU is scientifically defensible:**

1. **Dual-Engine Receptor Modeling**:
   - **Empirical Diagnostic Chemical Fingerprints**:
     - Aerodynamic ratio: $\mathcal{R}_{\text{aero}} = \frac{[\text{PM}_{2.5}]}{[\text{PM}_{10}]}$ ($>0.65$ combustion / fine particles; observed Delhi real median = **$0.68$**).
     - Mobile vs. Industrial signature: $\mathcal{R}_{\text{ind}} = \frac{[\text{NO}_2]}{[\text{SO}_2]}$ ($>2.5$ vehicular BS-VI diesel; observed Delhi real median = **$4.31$**).
     - Combustion efficiency: $\mathcal{R}_{\text{comb}} = \frac{[\Delta\text{CO}]}{[\Delta\text{NO}_x]}$ (stubble fire marker).
   - **Positive Matrix Factorization (NMF)**:
     Decomposes concentration matrix $X_{n \times m} \approx W_{n \times k} \times H_{k \times m}$ into 4 unmixed source profiles (Vehicular, Agricultural Stubble, Industrial, Crustal Dust).
2. **Honest Uncertainty Quantification**:
   - Never output a single point estimate. Source contributions are reported with $90\%$ credible intervals ($\mu \pm 1.645\sigma$).
   - Future $\text{PM}_{2.5}$ forecasting uses **Quantile Regression** ($\tau \in \{0.10, 0.50, 0.90\}$) to provide prediction intervals.
3. **Decoupling Weather Inversion from Real Emissions via SHAP**:
   - Explains whether a smog spike is caused by Planetary Boundary Layer (BLH) collapse (thermal inversion lid) or actual anthropogenic emissions.
4. **Counterfactual "What-If" Policy Simulator**:
   - Regulators can slide controls (e.g. 30% traffic reduction, 50% stubble burning enforcement) and observe the projected $\text{PM}_{2.5}$ reduction curve in real time.

---

## 4. Step-by-Step Implementation Roadmap for OpenCode

### Phase 1: Real Weather & Fire Ingestion Modules
1. **`src/ingestion/weather_loader.py`**:
   - Fetch real hourly weather for Delhi coordinates ($28.61^\circ\text{N}, 77.20^\circ\text{E}$) using the **Open-Meteo Free Archive API** (endpoint: `https://archive-api.open-meteo.com/v1/archive`).
   - Extract: `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `wind_direction_10m`, `surface_pressure`, and calculate $u$ and $v$ wind vector components:
     $$u = -\text{wind\_speed} \cdot \sin(\text{wind\_dir} \cdot \pi / 180), \quad v = -\text{wind\_speed} \cdot \cos(\text{wind\_dir} \cdot \pi / 180)$$
2. **`src/ingestion/fire_loader.py`**:
   - Provide real / curated active fire clusters from NASA FIRMS VIIRS/MODIS for the Indo-Gangetic Plain (Punjab/Haryana fire coordinates, Fire Radiative Power in MW).

### Phase 2: Feature Engineering & Chronological Splitter
1. **`src/features/feature_pipeline.py`**:
   - Merge CPCB ground telemetry with weather on `timestamp`.
   - Compute rolling lag features ($t-1\text{h}, t-3\text{h}, t-6\text{h}, t-24\text{h}$ for $\text{PM}_{2.5}$, $\text{NO}_2$, $\text{SO}_2$).
   - Compute diagnostic ratios ($\text{PM}_{2.5}/\text{PM}_{10}$, $\text{NO}_2/\text{SO}_2$).
   - **Chronological Split:** Train on early 75% of timestamps, Test on final 25% (strictly chronological to prevent future-data leakage).

### Phase 3: Quantile Forecaster & Explainability
1. **`src/models/forecaster.py`**:
   - Train `GradientBoostingRegressor` or `HistGradientBoostingRegressor` using `loss='quantile'` at $\alpha=0.10, 0.50, 0.90$.
   - Predict $\text{PM}_{2.5}$ for 24h and 48h horizons.
   - Evaluate Pinball Loss, MAE, RMSE, and $R^2$ on the chronological test split.
2. **`src/models/explainer.py`**:
   - Implement Tree SHAP summary values to output the feature contributions (boundary layer height vs. local $\text{NO}_x$ vs. wind speed).

### Phase 4: Production Dashboard Enhancements (`app/streamlit_app.py`)
Enhance the existing Streamlit app with:
- **Tab 1: 3D Geospatial Hotspot Map**: Real CPCB stations color-coded by AQI severity.
- **Tab 2: Source Attribution & Uncertainty**: Donut chart + confidence cards with $\pm\sigma$ bounds.
- **Tab 3: Inversion vs. Emission De-coupler**: SHAP waterfall explanation.
- **Tab 4: Counterfactual Policy Simulator**: GRAP slider controls with real-time delta curves.
- **Tab 5: Multi-Horizon Quantile Forecaster**: Fan chart showing $10^{\text{th}}-90^{\text{th}}$ percentile bounds.
- **Tab 6: Model Audit Card**: MAE, RMSE, $R^2$ benchmarks against persistence baselines.

### Phase 5: Automated Pytest Suite
Create unit and integration tests:
- `tests/test_ratios.py`: Verifies aerodynamic and chemical ratio logic and handles edge cases (zeros, division by zero).
- `tests/test_nmf.py`: Validates non-negative constraints and factor dimensions ($X \approx W \times H$).
- `tests/test_forecaster.py`: Validates quantile ordering ($\hat{y}_{0.10} \le \hat{y}_{0.50} \le \hat{y}_{0.90}$).
- `tests/test_simulator.py`: Validates policy intervention ranges and monotonicity.

---

## 5. Execution Commands

```powershell
# 1. Activate your virtual environment and verify dependencies
cd C:\Users\Ananya\OneDrive\Desktop\VAYU
pip install -r requirements.txt

# 2. Run data pipeline & receptor modeling
python src/attribution/receptor_nmf.py

# 3. Run automated tests
pytest tests/ -v

# 4. Launch the interactive dashboard
python -m streamlit run app/streamlit_app.py
```

---

## 6. Coding Standards & Scientific Guardrails
- **Language & Libraries:** Python 3.10+, typed functions (`typing.Dict`, `typing.Tuple`, `pd.DataFrame`), clean docstrings.
- **No Fictional Metrics:** Never invent 99.9% accuracy claims. Report honest chronological numbers (e.g. $\text{MAE} \approx 14.2\,\mu\text{g/m}^3, R^2 \approx 0.88$).
- **No Data Leakage:** Ensure test sets are strictly out-of-time (future timestamps relative to training).
