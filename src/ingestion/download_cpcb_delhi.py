import os
import urllib.request
import pandas as pd

RAW_DIR = r"C:\Users\Ananya\OneDrive\Desktop\VAYU\data\raw"
PROCESSED_DIR = r"C:\Users\Ananya\OneDrive\Desktop\VAYU\data\processed"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

PARQUET_URL = "https://raw.githubusercontent.com/Vonter/india-cpcb-aqi/main/data/latest-air-quality.parquet"
LOCAL_PARQUET = os.path.join(RAW_DIR, "latest-air-quality.parquet")

print("1. Downloading latest CPCB air quality parquet dataset (~14 MB)...")
urllib.request.urlretrieve(PARQUET_URL, LOCAL_PARQUET)
print("Download complete! File size:", os.path.getsize(LOCAL_PARQUET), "bytes")

print("2. Reading parquet and filtering for Delhi-NCR stations...")
df = pd.read_parquet(LOCAL_PARQUET)
print("Total rows across India:", len(df))
print("Available columns:", df.columns.tolist())

# Check State / City values
if "City" in df.columns:
    delhi_df = df[df["City"].str.contains("Delhi", case=False, na=False)].copy()
elif "State" in df.columns:
    delhi_df = df[df["State"].str.contains("Delhi", case=False, na=False)].copy()
else:
    delhi_df = df.copy()

print(f"Delhi records found: {len(delhi_df)}")
if len(delhi_df) > 0:
    stations = delhi_df["Station Name"].unique() if "Station Name" in delhi_df.columns else delhi_df["Station ID"].unique()
    print("Unique Delhi stations sample:", stations[:10])
    print("Delhi sample row:\n", delhi_df.head(2))
    
    # Save processed Delhi slice
    delhi_parquet = os.path.join(PROCESSED_DIR, "delhi_cpcb_real.parquet")
    delhi_csv = os.path.join(PROCESSED_DIR, "delhi_cpcb_real.csv")
    delhi_df.to_parquet(delhi_parquet, index=False)
    delhi_df.head(1000).to_csv(delhi_csv, index=False)
    print(f"Saved real Delhi telemetry to {delhi_parquet} ({len(delhi_df)} records)")
