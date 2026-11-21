from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rainfall_features import (
    CITIES, NASA_COLS, NASA_PARAMS, OM_DAILY_VARS, OM_HOURLY_VARS, TARGET,
    INPUT_COLUMNS, ALL_NUMERIC, add_history_features, add_target, aggregate_hourly,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT_CSV = ROOT / "data" / "weather_pakistan.csv"
FULL_CSV = ROOT / "data" / "weather_pakistan_all_features.csv"
INFO_JSON = ROOT / "data" / "dataset_info.json"

START_YEAR, END_YEAR = 2010, 2024
# we keep 2020-2024 for the dataset
DATASET_START = "2020-01-01"
CHUNK_YEARS = 5
TZ = "Asia/Karachi"

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
NASA_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

HEADERS = {"User-Agent": "CT378-rainfall-student-project (educational use)"}


# API request with retries
def get_json(url: str, params: dict, retries: int = 6) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=120)
            if r.status_code == 200:
                return r.json()
            wait = 20 * (attempt + 1) if r.status_code == 429 else 5 * (attempt + 1)
            print(f"    HTTP {r.status_code} - retrying in {wait}s ...")
        except requests.RequestException as e:
            wait = 5 * (attempt + 1)
            print(f"    {type(e).__name__} - retrying in {wait}s ...")
        time.sleep(wait)
    raise RuntimeError(f"Failed after {retries} attempts: {url}")


# save API responses so we don't download twice
def cached(path: Path, fetch):
    if path.exists():
        return json.loads(path.read_text())
    data = fetch()
    path.write_text(json.dumps(data))
    time.sleep(1.5)
    return data


def parse_openmeteo(js: dict) -> pd.DataFrame:
    daily = pd.DataFrame(js["daily"]).rename(columns={"time": "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    hourly = aggregate_hourly(pd.DataFrame(js["hourly"]))
    out = daily.merge(hourly, on="date", how="left")
    out["elevation"] = js.get("elevation", np.nan)
    return out


# download Open-Meteo data for one city
def fetch_openmeteo(city: str, lat: float, lon: float) -> pd.DataFrame:
    parts = []
    for y0 in range(START_YEAR, END_YEAR + 1, CHUNK_YEARS):
        y1 = min(y0 + CHUNK_YEARS - 1, END_YEAR)
        path = RAW / f"openmeteo_{city}_{y0}_{y1}.json"
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": f"{y0}-01-01", "end_date": f"{y1}-12-31",
            "daily": ",".join(OM_DAILY_VARS),
            "hourly": ",".join(OM_HOURLY_VARS),
            "timezone": TZ,
        }
        print(f"  Open-Meteo {city} {y0}-{y1}")
        parts.append(parse_openmeteo(cached(path, lambda: get_json(OPEN_METEO_URL, params))))
    return pd.concat(parts, ignore_index=True)


def parse_nasa(js: dict) -> pd.DataFrame:
    p = js["properties"]["parameter"]
    df = pd.DataFrame({k: pd.Series(v) for k, v in p.items()})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.replace(-999.0, np.nan)
    df = df.rename(columns={k: "nasa_" + k.lower() for k in df.columns})
    df = df.reindex(columns=NASA_COLS)
    return df.rename_axis("date").reset_index()


# download NASA POWER data for one city
def fetch_nasa(city: str, lat: float, lon: float) -> pd.DataFrame:
    path = RAW / f"nasa_{city}.json"
    params = {
        "parameters": ",".join(NASA_PARAMS), "community": "AG",
        "latitude": lat, "longitude": lon,
        "start": f"{START_YEAR}0101", "end": f"{END_YEAR}1231", "format": "JSON",
    }
    print(f"  NASA POWER {city}")
    return parse_nasa(cached(path, lambda: get_json(NASA_URL, params)))


# merge both sources and add features and target
def build_dataset() -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    frames = []
    for city, (lat, lon) in CITIES.items():
        print(f"[{city}]")
        om = fetch_openmeteo(city, lat, lon)
        nasa = fetch_nasa(city, lat, lon)
        merged = om.merge(nasa, on="date", how="left")
        merged.insert(0, "city", city)
        frames.append(merged)

    df = pd.concat(frames, ignore_index=True)
    in_period = df["date"] >= DATASET_START
    raw_cols = [c for c in df.columns if c not in ("city", "date")]
    build_dataset.raw_missing = {c: int(n) for c, n in df.loc[in_period, raw_cols].isna().sum().items()}
    df = add_history_features(df)
    df = add_target(df)
    df = df[df["date"] >= DATASET_START]
    df = df[["date", "city"] + ALL_NUMERIC + [TARGET]].sort_values(["date", "city"])
    return df.reset_index(drop=True)


# build the dataset and save the CSV files
def main():
    full = build_dataset()
    full.to_csv(FULL_CSV, index=False)
    df = full[["date"] + INPUT_COLUMNS + [TARGET]]
    df.to_csv(OUT_CSV, index=False)
    n_features = len(INPUT_COLUMNS)
    info = {
        "file": OUT_CSV.name,
        "rows": int(len(df)),
        "input_features": n_features,
        "features": INPUT_COLUMNS,
        "all_collected_features": 1 + len(ALL_NUMERIC),
        "cities": list(CITIES),
        "period": f"{DATASET_START} to {END_YEAR}-12-31 (raw download {START_YEAR}-{END_YEAR})",
        "downloaded_on_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "target": f"{TARGET} = 1 if next-day precipitation >= 1 mm",
        "rain_rate": round(float(df[TARGET].mean()), 4),
        "sources": [
            {
                "name": "Open-Meteo Historical Weather API (ERA5 / ERA5-Land reanalysis)",
                "url": "https://open-meteo.com/en/docs/historical-weather-api",
                "method": "HTTP GET per city, daily + hourly variables, timezone Asia/Karachi",
                "licence": "CC BY 4.0 (https://open-meteo.com/en/licence)",
                "citation": "Zippenfenig, P. (2023). Open-Meteo.com Weather API. Zenodo. doi:10.5281/zenodo.7970649",
            },
            {
                "name": "NASA POWER Daily Point API (MERRA-2 / CERES)",
                "url": "https://power.larc.nasa.gov/docs/services/api/temporal/daily/",
                "method": "HTTP GET per city, community=AG, 10 parameters",
                "licence": "No restrictions on use; acknowledgement requested (https://power.larc.nasa.gov/docs/referencing/)",
                "citation": "NASA Langley Research Center (LaRC) POWER Project, funded through the NASA Earth Science Division.",
            },
        ],
        "personal_identifiers": "none (weather values per city and date only)",
        "raw_missing_values_in_period": {
            "total": int(sum(build_dataset.raw_missing.values())),
            "note": "NASA POWER -999 flags and Open-Meteo nulls, counted after merging, before feature engineering",
        },
    }
    INFO_JSON.write_text(json.dumps(info, indent=2))

    print("\nSaved", OUT_CSV, f"- {len(df):,} rows x {n_features} input features")
    print(f"Rain-tomorrow rate: {df[TARGET].mean():.1%}")
    assert len(df) >= 5000, "Guideline: at least 5,000 rows"
    assert n_features >= 15, "Guideline: at least 15 features"
    print("Guideline check passed: >= 5,000 rows and >= 15 features")


if __name__ == "__main__":
    main()
