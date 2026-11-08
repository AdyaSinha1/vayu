"""Shared station registry and pollutant options for VAYU (app + scripts)."""

from __future__ import annotations

# Station coordinates (Delhi-NCR CPCB network)
STATION_COORDS: dict[str, tuple[float, float]] = {
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

POLLUTANT_OPTIONS: dict[str, str] = {"PM2.5": "pm25", "PM10": "pm10", "NO2": "no2", "SO2": "so2", "CO": "co", "Ozone": "ozone"}
