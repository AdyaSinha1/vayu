"""CPCB National Air Quality Index sub-index computation.

Implements the official CPCB (India) AQI breakpoints with linear interpolation
between breakpoints, per the National AQI methodology (CPCB, 2014).
Concentration units follow CPCB: ug/m3 for PM2.5/PM10/NO2/SO2/O3, mg/m3 for CO.

The overall AQI is the maximum sub-index across available pollutants, with the
driving pollutant reported (CPCB requires at least three pollutants including
PM2.5 or PM10 for an official overall value; callers surface that caveat).
"""

from __future__ import annotations

# (C_lo, C_hi, I_lo, I_hi) breakpoints per pollutant
CPCB_BREAKPOINTS: dict[str, list[tuple[float, float, float, float]]] = {
    "pm25": [
        (0, 30, 0, 50), (30, 60, 51, 100), (60, 90, 101, 200),
        (90, 120, 201, 300), (120, 250, 301, 400), (250, 500, 401, 500),
    ],
    "pm10": [
        (0, 50, 0, 50), (50, 100, 51, 100), (100, 250, 101, 200),
        (250, 350, 201, 300), (350, 430, 301, 400), (430, 1000, 401, 500),
    ],
    "no2": [
        (0, 40, 0, 50), (40, 80, 51, 100), (80, 180, 101, 200),
        (180, 280, 201, 300), (280, 400, 301, 400), (400, 1000, 401, 500),
    ],
    "so2": [
        (0, 40, 0, 50), (40, 80, 51, 100), (80, 380, 101, 200),
        (380, 800, 201, 300), (800, 1600, 301, 400), (1600, 2400, 401, 500),
    ],
    "co": [  # mg/m3
        (0, 1, 0, 50), (1, 2, 51, 100), (2, 10, 101, 200),
        (10, 17, 201, 300), (17, 34, 301, 400), (34, 50, 401, 500),
    ],
    "ozone": [
        (0, 50, 0, 50), (50, 100, 51, 100), (100, 168, 101, 200),
        (168, 208, 201, 300), (208, 748, 301, 400), (748, 1000, 401, 500),
    ],
}

POLLUTANT_LABELS = {
    "pm25": "PM2.5",
    "pm10": "PM10",
    "no2": "NO2",
    "so2": "SO2",
    "co": "CO",
    "ozone": "Ozone",
}

AQI_CATEGORIES: list[tuple[float, float, str, str]] = [
    (0, 50, "Good", "#A6B9A3"),
    (51, 100, "Satisfactory", "#C9CDA0"),
    (101, 200, "Moderate", "#D9C7A5"),
    (201, 300, "Poor", "#D8A468"),
    (301, 400, "Very Poor", "#C97B6D"),
    (401, 500, "Severe", "#A94F44"),
]


def sub_index(pollutant: str, concentration: float) -> float | None:
    """CPCB AQI sub-index via linear interpolation between breakpoints."""
    table = CPCB_BREAKPOINTS.get(pollutant)
    if table is None or concentration is None or concentration < 0:
        return None
    for c_lo, c_hi, i_lo, i_hi in table:
        if concentration <= c_hi:
            if c_hi == c_lo:
                return i_hi
            return round((i_hi - i_lo) / (c_hi - c_lo) * (concentration - c_lo) + i_lo, 1)
    return 500.0  # above the highest defined breakpoint


def aqi_category(aqi: float | None) -> tuple[str, str] | tuple[None, None]:
    """Map an AQI value to (category name, colour)."""
    if aqi is None:
        return None, None
    for lo, hi, name, colour in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return name, colour
    return "Severe", "#A94F44"


def overall_aqi(concentrations: dict[str, float | None]) -> tuple[float | None, str | None, dict[str, float | None]]:
    """Overall AQI = max sub-index; returns (aqi, driving pollutant label, sub-index table)."""
    subs: dict[str, float | None] = {}
    for pol, conc in concentrations.items():
        subs[pol] = sub_index(pol, conc) if conc is not None and conc == conc else None
    valid = {k: v for k, v in subs.items() if v is not None}
    if not valid:
        return None, None, subs
    driver = max(valid, key=valid.get)
    return float(valid[driver]), POLLUTANT_LABELS.get(driver, driver), subs


if __name__ == "__main__":
    demo = {"pm25": 167.0, "pm10": 310.0, "no2": 70.0, "so2": 15.0, "co": 1.6, "ozone": 22.0}
    aqi, driver, subs = overall_aqi(demo)
    print("Overall AQI:", aqi, "| driven by:", driver)
    print({k: v for k, v in subs.items()})
    print("Category:", aqi_category(aqi))
