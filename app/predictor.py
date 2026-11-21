"""
predictor.py - everything the app does EXCEPT the user interface:
loading the model, building an input row, live weather, prediction and explanation.
(Kept separate from the user interface so it can be tested on its own.)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests

# rainfall_features.py lives in ../src (or next to this file, if copied there)
_SRC = Path(__file__).resolve().parent.parent / "src"
if not (Path(__file__).resolve().parent / "rainfall_features.py").exists() and _SRC.exists():
    sys.path.insert(0, str(_SRC))

import rainfall_features as rf  # noqa: F401  (needed so joblib can rebuild the pipeline)
from rainfall_features import (
    CITIES, INPUT_COLUMNS, OM_DAILY_VARS, OM_HOURLY_VARS, RAIN_THRESHOLD_MM,
    add_history_features, aggregate_hourly, specific_humidity_gkg,
)

HERE = Path(__file__).resolve().parent
MODEL_FILE = "rainfall_pipeline.joblib"
META_FILE = "model_metadata.json"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
PKT = ZoneInfo("Asia/Karachi")

# Friendly names shown in the explanation chart
LABELS = {
    "temperature_2m_max": "Max temperature", "temperature_2m_min": "Min temperature",
    "temperature_2m_mean": "Mean temperature", "precipitation_sum": "Rain today",
    "rain_sum": "Rain today (liquid)", "precipitation_hours": "Hours of rain today",
    "wind_speed_10m_max": "Max wind speed", "wind_gusts_10m_max": "Max wind gust",
    "wind_direction_10m_dominant": "Wind direction", "shortwave_radiation_sum": "Solar radiation",
    "et0_fao_evapotranspiration": "Evapotranspiration", "sunshine_duration": "Sunshine duration",
    "rh_mean": "Humidity (mean)", "nasa_rh2m": "Humidity (NASA)", "rh_max": "Humidity (max)", "rh_min": "Humidity (min)",
    "dew_point_mean": "Dew point", "pressure_msl_mean": "Pressure (mean)",
    "pressure_msl_min": "Pressure (min)", "cloud_cover_mean": "Cloud cover",
    "nasa_t2m": "Temperature (NASA)", "nasa_t2mdew": "Dew point (NASA)", "nasa_rh2m": "Humidity (NASA)",
    "nasa_qv2m": "Specific humidity (NASA)", "nasa_ps": "Surface pressure (NASA)",
    "nasa_ws2m": "Wind 2 m (NASA)", "nasa_ws10m": "Wind 10 m (NASA)", "nasa_cloud_amt": "Cloud amount (NASA)",
    "nasa_allsky_sfc_sw_dwn": "Solar radiation (NASA)", "nasa_prectotcorr": "Rain today (NASA)",
    "precip_lag1": "Rain yesterday", "precip_3d_sum": "Rain last 3 days", "precip_7d_sum": "Rain last 7 days",
    "rain_days_7d": "Rain days last week", "pressure_change_24h": "Pressure change 24 h",
    "rh_change_24h": "Humidity change 24 h", "temp_change_24h": "Temperature change 24 h",
    "elevation": "Elevation", "month": "Month", "day_of_year": "Day of year", "city": "City",
}

ELEVATION_HINT = {  # used only if the city-month median table lacks elevation
    "Karachi": 10, "Hyderabad": 20, "Quetta": 1680, "Multan": 125, "Faisalabad": 185,
    "Lahore": 215, "Sialkot": 255, "Islamabad": 540, "Peshawar": 340, "Muzaffarabad": 740,
}


def _locate(name: str) -> Path:
    """Model files live in the project's models/ folder."""
    for p in (HERE / name, HERE.parent / "models" / name):
        if p.exists():
            return p
    raise FileNotFoundError(f"{name} not found in the models/ folder")


class RainPredictor:
    def __init__(self):
        self.model = joblib.load(_locate(MODEL_FILE))
        self.meta = json.loads(_locate(META_FILE).read_text(encoding="utf-8"))
        self.threshold = float(self.meta["threshold"])
        self.cities = self.meta.get("cities", list(CITIES))
        imp = self.meta.get("feature_importance", [])
        self.top_features = [d["feature"] for d in imp if d["feature"] not in ("month", "day_of_year")][:12]

    # ------------------------------------------------------------ defaults
    def typical(self, city: str, month: int) -> dict:
        """Median value of every input for this city and month in the training years."""
        base = dict(self.meta.get("train_medians", {}))
        base.update(self.meta.get("monthly_medians", {}).get(f"{city}|{month}", {}))
        base.setdefault("elevation", ELEVATION_HINT.get(city, np.nan))
        return base

    def row(self, city: str, month: int, day_of_year: int, values: dict) -> pd.DataFrame:
        r = self.typical(city, month)
        r.update({k: v for k, v in values.items() if v is not None})
        r.update({"city": city, "month": month, "day_of_year": day_of_year})
        X = pd.DataFrame([{c: r.get(c, np.nan) for c in INPUT_COLUMNS}])
        num = [c for c in INPUT_COLUMNS if c != "city"]
        X[num] = X[num].astype(float)  # e.g. wind direction typed as an integer
        return X

    # ------------------------------------------------------------ manual input
    def manual_row(self, city, month, tmax, humidity, dew, pressure, pressure_change, clouds,
                   wind, wind_dir, rain_today, rain_3d, rain_7d):
        """Returns (X for the model, dict of everything to display)."""
        doy = int(pd.Timestamp(2023, month, 15).dayofyear)
        vals = {
            "temperature_2m_max": tmax, "rh_mean": humidity, "dew_point_mean": dew,
            "pressure_msl_mean": pressure, "pressure_change_24h": pressure_change, "cloud_cover_mean": clouds,
            "wind_speed_10m_max": wind, "wind_direction_10m_dominant": wind_dir,
            "precipitation_sum": rain_today, "precip_3d_sum": rain_3d, "precip_7d_sum": rain_7d,
        }
        vals.update(self.nasa_from_openmeteo(city, month, vals))  # second-source (NASA POWER) inputs
        return self.row(city, month, doy, vals), vals

    def nasa_from_openmeteo(self, city: str, month: int, om: dict) -> dict:
        """Estimate the 4 NASA POWER inputs from Open-Meteo values.

        The two sources measure on slightly different scales (e.g. NASA cloud amount is usually higher), so
        copying values across would make an ordinary day look unusual. We use the delta method: start from the
        typical NASA value for this city and month, and add how far today's Open-Meteo value is from ITS typical
        value:  nasa_today = nasa_typical + (om_today - om_typical)."""
        typ = self.typical(city, month)

        def qv(dew, p_hpa):
            return float(specific_humidity_gkg(dew, p_hpa))

        def delta(nasa_key, om_now, om_key, scale=1.0, lo=None, hi=None):
            v = typ.get(nasa_key, np.nan) + scale * (om_now - typ.get(om_key, om_now))
            return float(np.clip(v, lo if lo is not None else -np.inf, hi if hi is not None else np.inf))

        p_typ = typ.get("pressure_msl_mean", om["pressure_msl_mean"])
        return {
            "nasa_cloud_amt": delta("nasa_cloud_amt", om["cloud_cover_mean"], "cloud_cover_mean", lo=0, hi=100),
            "nasa_rh2m": delta("nasa_rh2m", om["rh_mean"], "rh_mean", lo=0, hi=100),
            "nasa_prectotcorr": delta("nasa_prectotcorr", om["precipitation_sum"], "precipitation_sum", lo=0),
            "nasa_ps": delta("nasa_ps", om["pressure_msl_mean"], "pressure_msl_mean", scale=0.1),  # hPa -> kPa
            "nasa_qv2m": float(max(0.0, typ.get("nasa_qv2m", np.nan)
                                    + qv(om["dew_point_mean"], om["pressure_msl_mean"])
                                    - qv(typ.get("dew_point_mean", om["dew_point_mean"]), p_typ))),
        }

    # ------------------------------------------------------------ live weather
    def live_row(self, city: str):
        lat, lon = CITIES[city]
        params = {
            "latitude": lat, "longitude": lon, "timezone": "Asia/Karachi",
            "daily": ",".join(OM_DAILY_VARS),
            "hourly": ",".join(OM_HOURLY_VARS),
            "past_days": 7, "forecast_days": 1,
        }
        r = requests.get(FORECAST_URL, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()
        daily = pd.DataFrame(js["daily"]).rename(columns={"time": "date"})
        daily["date"] = pd.to_datetime(daily["date"])
        agg = aggregate_hourly(pd.DataFrame(js["hourly"]))
        d = daily.merge(agg, on="date", how="left")
        d["city"] = city
        d["elevation"] = js.get("elevation", np.nan)
        d = add_history_features(d)
        today = d.iloc[-1].to_dict()
        date = pd.Timestamp(today["date"])
        today.update(self.nasa_from_openmeteo(city, int(date.month), today))
        X = self.row(city, int(date.month), int(date.dayofyear),
                     {k: v for k, v in today.items() if k in INPUT_COLUMNS and not pd.isna(v)})
        history = d[["date", "precipitation_sum"]].tail(8).reset_index(drop=True)  # last 7 days + today
        return X, date.date(), history, today

    # ------------------------------------------------------------ predict + explain
    def predict(self, X: pd.DataFrame) -> float:
        return float(self.model.predict_proba(X)[:, 1][0])

    def explain(self, X: pd.DataFrame, k: int = 8) -> pd.DataFrame:
        """Local what-if explanation: for each important feature, replace this day's value with the
        typical value for the city and month and see how much the rain probability moves.
        Positive = this feature pushed the prediction towards rain."""
        city, month = X["city"].iloc[0], int(X["month"].iloc[0])
        typ = self.typical(city, month)
        feats = [f for f in self.top_features if f in X.columns and f in typ] or \
                [c for c in INPUT_COLUMNS if c in typ][:12]
        batch = pd.concat([X] * len(feats), ignore_index=True)
        for i, f in enumerate(feats):
            batch.loc[i, f] = typ[f]
        p0 = self.predict(X)
        effect = p0 - self.model.predict_proba(batch)[:, 1]
        out = pd.DataFrame({"feature": feats, "effect": effect,
                            "value": [X[f].iloc[0] for f in feats], "typical": [typ[f] for f in feats]})
        out["label"] = out["feature"].map(LABELS).fillna(out["feature"])
        return out.reindex(out["effect"].abs().sort_values(ascending=False).index).head(k)


def validate_manual(tmax, humidity, dew, pressure, clouds, wind, wind_dir, rain_today, rain_3d, rain_7d) -> list[str]:
    """Return a list of human-readable problems (empty list = input is valid)."""
    vals = dict(max_temperature=tmax, humidity=humidity, dew_point=dew, pressure=pressure, cloud_cover=clouds,
                wind=wind, wind_direction=wind_dir, rain_today=rain_today, rain_3_days=rain_3d, rain_7_days=rain_7d)
    missing = [k for k, v in vals.items() if v is None or (isinstance(v, float) and np.isnan(v))]
    if missing:
        return [f"Please fill in: {', '.join(missing)}"]
    errs = []
    if not -30 <= tmax <= 55:
        errs.append("Max temperature must be between −30 °C and 55 °C.")
    if dew > tmax:
        errs.append("Dew point cannot be higher than the maximum temperature.")
    if not 0 <= humidity <= 100 or not 0 <= clouds <= 100:
        errs.append("Humidity and cloud cover are percentages (0–100).")
    if not 930 <= pressure <= 1060:
        errs.append("Sea-level pressure should be between 930 and 1060 hPa.")
    if not 0 <= wind_dir <= 360:
        errs.append("Wind direction is in degrees (0–360).")
    if min(wind, rain_today, rain_3d, rain_7d) < 0:
        errs.append("Wind and rainfall cannot be negative.")
    if rain_3d + 1e-9 < rain_today:
        errs.append("Rain in the last 3 days must include today's rain.")
    if rain_7d + 1e-9 < rain_3d:
        errs.append("Rain in the last 7 days must include the last 3 days.")
    return errs


def now_pkt() -> datetime:
    return datetime.now(PKT)
