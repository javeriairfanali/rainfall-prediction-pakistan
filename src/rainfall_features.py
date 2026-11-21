from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CITIES: dict[str, tuple[float, float]] = {
    "Karachi": (24.8607, 67.0011),
    "Quetta": (30.1798, 66.9750),
    "Lahore": (31.5204, 74.3587),
    "Islamabad": (33.6844, 73.0479),
}

# a day counts as rainy if it gets at least 1 mm
RAIN_THRESHOLD_MM = 1.0

# daily variables downloaded from Open-Meteo
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
# hourly Open-Meteo variables, averaged to daily values
OM_HOURLY_VARS = ["relative_humidity_2m", "dew_point_2m", "pressure_msl", "cloud_cover"]

# daily variables downloaded from NASA POWER
NASA_PARAMS = [
    "T2M",
    "T2MDEW",
    "RH2M",
    "QV2M",
    "PS",
    "WS2M",
    "WS10M",
    "CLOUD_AMT",
    "ALLSKY_SFC_SW_DWN",
    "PRECTOTCORR",
]
NASA_COLS = ["nasa_" + p.lower() for p in NASA_PARAMS]

HOURLY_AGG_COLS = [
    "rh_mean", "rh_max", "rh_min",
    "dew_point_mean",
    "pressure_msl_mean", "pressure_msl_min",
    "cloud_cover_mean",
]

# features built from previous days
HISTORY_COLS = [
    "precip_lag1",
    "precip_3d_sum",
    "precip_7d_sum",
    "rain_days_7d",
    "pressure_change_24h",
    "rh_change_24h",
    "temp_change_24h",
]

# all 40 collected features
ALL_NUMERIC = (
    OM_DAILY_VARS + HOURLY_AGG_COLS + NASA_COLS + HISTORY_COLS
    + ["elevation", "month", "day_of_year"]
)

# the 16 features kept after feature selection on training years
SELECTED_NUMERIC = [
    "month",
    "nasa_cloud_amt", "nasa_prectotcorr", "pressure_change_24h", "precip_3d_sum", "nasa_qv2m",
    "cloud_cover_mean", "pressure_msl_mean", "nasa_ps", "precipitation_sum", "nasa_rh2m",
    "wind_speed_10m_max", "wind_direction_10m_dominant", "temperature_2m_max", "precip_7d_sum",
]
RAW_NUMERIC = SELECTED_NUMERIC
CATEGORICAL = ["city"]
INPUT_COLUMNS = CATEGORICAL + RAW_NUMERIC

# extra features created inside the pipeline
ENGINEERED = [
    "month_sin", "month_cos",
    "is_monsoon",
    "rained_today",
]
MODEL_NUMERIC = [c for c in RAW_NUMERIC if c != "month"] + ENGINEERED

TARGET = "rain_tomorrow"


# turn hourly data into daily averages
def aggregate_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
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


# rain and change features from the past few days
def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
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


# target: does it rain tomorrow (1) or not (0)
def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["city", "date"]).copy()
    nxt = df.groupby("city")["precipitation_sum"].shift(-1)
    df[TARGET] = (nxt >= RAIN_THRESHOLD_MM).astype("float")
    df.loc[nxt.isna(), TARGET] = np.nan
    return df.dropna(subset=[TARGET]).assign(**{TARGET: lambda d: d[TARGET].astype(int)})


# specific humidity from dew point and pressure
def specific_humidity_gkg(dew_c: float, pressure_hpa: float) -> float:
    e = 6.112 * np.exp(17.67 * dew_c / (dew_c + 243.5))
    return 622.0 * e / (pressure_hpa - 0.378 * e)


# adds month sin/cos, monsoon flag and rained today
class WeatherFeatureEngineer(BaseEstimator, TransformerMixin):

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


# clips extreme values to the 1st-99th percentile of training data
class QuantileClipper(BaseEstimator, TransformerMixin):

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
        return np.clip(X, self.lo_, self.hi_)


# clip, impute and scale numbers; one-hot encode city
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


# full pipeline: feature engineering + preprocessing + model
def build_pipeline(model) -> Pipeline:
    return Pipeline([
        ("features", WeatherFeatureEngineer()),
        ("preprocess", build_preprocessor()),
        ("model", model),
    ])
