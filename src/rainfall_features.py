"""
rainfall_features.py
====================
Shared code for the RainCast Pakistan rainfall prediction project.

This one module is imported by:
  * src/download_data.py          (building the dataset)
  * notebooks/rainfall_prediction (training the model)
  * app/app.py + app/predictor.py (the Streamlit app)

Keeping it in ONE place guarantees that the features the model sees at
prediction time are computed exactly like the features it was trained on.
The saved joblib pipeline also refers to the classes below by the module
name `rainfall_features`, so the app must be able to import it (predictor.py adds src/ to the path).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# 1. Study area: 4 Pakistani cities with contrasting climates -
#    Karachi (coastal, dry), Lahore (monsoon plains), Islamabad (wettest, foothills),
#    Quetta (mountain plateau, winter rain from western disturbances).
# ---------------------------------------------------------------------------
CITIES: dict[str, tuple[float, float]] = {
    "Karachi": (24.8607, 67.0011),
    "Quetta": (30.1798, 66.9750),
    "Lahore": (31.5204, 74.3587),
    "Islamabad": (33.6844, 73.0479),
}

RAIN_THRESHOLD_MM = 1.0  # WMO convention: a "rain day" has >= 1 mm

# ---------------------------------------------------------------------------
# 2. Raw variables requested from each source
# ---------------------------------------------------------------------------
# Source 1 - Open-Meteo Historical Weather API (ERA5 reanalysis), daily
OM_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "sunshine_duration",
]
# Source 1 - hourly variables, aggregated to daily statistics by us
OM_HOURLY_VARS = ["relative_humidity_2m", "dew_point_2m", "pressure_msl", "cloud_cover"]

# Source 2 - NASA POWER daily point API (MERRA-2 / CERES), max 20 per request
NASA_PARAMS = [
    "T2M",                # temperature at 2 m (C)
    "T2MDEW",             # dew/frost point at 2 m (C)
    "RH2M",               # relative humidity at 2 m (%)
    "QV2M",               # specific humidity at 2 m (g/kg)
    "PS",                 # surface pressure (kPa)
    "WS2M",               # wind speed at 2 m (m/s)
    "WS10M",              # wind speed at 10 m (m/s)
    "CLOUD_AMT",          # cloud amount (%)
    "ALLSKY_SFC_SW_DWN",  # all-sky surface shortwave (kWh/m2/day)
    "PRECTOTCORR",        # bias-corrected precipitation (mm/day)
]
NASA_COLS = ["nasa_" + p.lower() for p in NASA_PARAMS]

# ---------------------------------------------------------------------------
# 3. Feature lists used by the model
# ---------------------------------------------------------------------------
HOURLY_AGG_COLS = [
    "rh_mean", "rh_max", "rh_min",
    "dew_point_mean",
    "pressure_msl_mean", "pressure_msl_min",
    "cloud_cover_mean",
]

HISTORY_COLS = [
    "precip_lag1",          # yesterday's precipitation (mm)
    "precip_3d_sum",        # precipitation over the last 3 days incl. today
    "precip_7d_sum",        # precipitation over the last 7 days incl. today
    "rain_days_7d",         # number of rain days in the last 7 days
    "pressure_change_24h",  # today's mean pressure - yesterday's
    "rh_change_24h",        # today's mean humidity - yesterday's
    "temp_change_24h",      # today's mean temperature - yesterday's
]

# Every column we collect/derive (40 features) - saved in data/weather_pakistan_all_features.csv
ALL_NUMERIC = (
    OM_DAILY_VARS + HOURLY_AGG_COLS + NASA_COLS + HISTORY_COLS
    + ["elevation", "month", "day_of_year"]
)

# Feature selection -> 16 model features (15 numeric + city), chosen ONLY from the training years
# (2020-2022) so the validation/test years cannot influence the choice. Rule (see notebook section 1.1):
#   1. permutation importance of a Random Forest fitted on 2020-2021 and scored on 2022 (avg of 3 seeds)
#   2. walk down the ranking; skip a feature with |r| >= 0.9 to one already chosen (training-year correlation)
#   3. stop when importance is no longer positive
#   4. `city` and `month` are always kept (seasonality differs by city)
SELECTED_NUMERIC = [
    "month",
    "nasa_cloud_amt", "nasa_prectotcorr", "pressure_change_24h", "precip_3d_sum", "nasa_qv2m",
    "cloud_cover_mean", "pressure_msl_mean", "nasa_ps", "precipitation_sum", "nasa_rh2m",
    "wind_speed_10m_max", "wind_direction_10m_dominant", "temperature_2m_max", "precip_7d_sum",
]
RAW_NUMERIC = SELECTED_NUMERIC
CATEGORICAL = ["city"]
INPUT_COLUMNS = CATEGORICAL + RAW_NUMERIC   # 16 features

# Columns created INSIDE the pipeline by WeatherFeatureEngineer
ENGINEERED = [
    "month_sin", "month_cos",   # cyclical month: December sits next to January
    "is_monsoon",           # July-September
    "rained_today",         # today's precipitation >= 1 mm
]
MODEL_NUMERIC = [c for c in RAW_NUMERIC if c != "month"] + ENGINEERED

TARGET = "rain_tomorrow"


# ---------------------------------------------------------------------------
# 4. Helpers used when building the dataset AND in the live app
# ---------------------------------------------------------------------------
def aggregate_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    """Turn an hourly Open-Meteo frame (column 'time' + OM_HOURLY_VARS) into
    one row per local day with the statistics in HOURLY_AGG_COLS."""
    h = hourly.copy()
    h["date"] = pd.to_datetime(h["time"]).dt.normalize()
    g = h.groupby("date")
    out = pd.DataFrame({
        "rh_mean": g["relative_humidity_2m"].mean(),
        "rh_max": g["relative_humidity_2m"].max(),
        "rh_min": g["relative_humidity_2m"].min(),
        "dew_point_mean": g["dew_point_2m"].mean(),
        "pressure_msl_mean": g["pressure_msl"].mean(),
        "pressure_msl_min": g["pressure_msl"].min(),
        "cloud_cover_mean": g["cloud_cover"].mean(),
    })
    return out.reset_index()


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add look-back features per city. Every value uses ONLY today and
    earlier days, so nothing from the future leaks into the inputs."""
    df = df.sort_values(["city", "date"]).copy()
    g = df.groupby("city", group_keys=False)
    p = df["precipitation_sum"]
    df["precip_lag1"] = g["precipitation_sum"].shift(1)
    df["precip_3d_sum"] = g["precipitation_sum"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    df["precip_7d_sum"] = g["precipitation_sum"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    df["_rainday"] = (p >= RAIN_THRESHOLD_MM).astype(float)
    df["rain_days_7d"] = g["_rainday"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    df["pressure_change_24h"] = g["pressure_msl_mean"].diff()
    df["rh_change_24h"] = g["rh_mean"].diff()
    df["temp_change_24h"] = g["temperature_2m_mean"].diff()
    df = df.drop(columns="_rainday")
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["day_of_year"] = pd.to_datetime(df["date"]).dt.dayofyear
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """rain_tomorrow = 1 if NEXT day's precipitation >= 1 mm (per city).
    The last day of each city has no 'tomorrow' and is dropped."""
    df = df.sort_values(["city", "date"]).copy()
    nxt = df.groupby("city")["precipitation_sum"].shift(-1)
    df[TARGET] = (nxt >= RAIN_THRESHOLD_MM).astype("float")
    df.loc[nxt.isna(), TARGET] = np.nan
    return df.dropna(subset=[TARGET]).assign(**{TARGET: lambda d: d[TARGET].astype(int)})


def specific_humidity_gkg(dew_c: float, pressure_hpa: float) -> float:
    """Specific humidity (g/kg) from dew point (C) and pressure (hPa),
    Magnus formula. Used by the live app to approximate NASA QV2M."""
    e = 6.112 * np.exp(17.67 * dew_c / (dew_c + 243.5))
    return 622.0 * e / (pressure_hpa - 0.378 * e)


# ---------------------------------------------------------------------------
# 5. Custom scikit-learn transformers (fitted on TRAINING data only)
# ---------------------------------------------------------------------------
class WeatherFeatureEngineer(BaseEstimator, TransformerMixin):
    """Row-wise feature engineering. Stateless, so it cannot leak."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        m = X["month"].astype(float)
        X["month_sin"] = np.sin(2 * np.pi * m / 12)
        X["month_cos"] = np.cos(2 * np.pi * m / 12)
        X["is_monsoon"] = m.isin([7, 8, 9]).astype(float)
        X["rained_today"] = (X["precipitation_sum"] >= RAIN_THRESHOLD_MM).astype(float)
        X.loc[X["precipitation_sum"].isna(), "rained_today"] = np.nan
        return X


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Outlier treatment (winsorising): learns the 1st and 99th percentile of
    each numeric column on the TRAINING data and clips new data to that range."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lo_, self.hi_)  # NaN stays NaN for the imputer


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("clip_outliers", QuantileClipper(0.01, 0.99)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numeric, MODEL_NUMERIC), ("cat", categorical, CATEGORICAL)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(model) -> Pipeline:
    """feature engineering -> clip/impute/scale/encode -> classifier."""
    return Pipeline([
        ("features", WeatherFeatureEngineer()),
        ("preprocess", build_preprocessor()),
        ("model", model),
    ])
