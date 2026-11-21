"""
RainCast Pakistan - Streamlit app
Predicts whether it will rain tomorrow in Karachi, Lahore, Islamabad or Quetta.

Run locally (from the project folder):   streamlit run app/app.py
Deployed on Streamlit Community Cloud with app/app.py as the entrypoint.

Tabs
  1. Live forecast   - pick a city; today's weather + last 7 days come from Open-Meteo
  2. Compare cities  - tomorrow's chance of rain for all 4 cities, ranked
  3. Manual input    - type the weather yourself (validated form)
  4. About the model - metrics, features, data sources, evaluation figures
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(page_title="RainCast Pakistan — will it rain tomorrow?", page_icon="🌧️", layout="wide")

import ui  # noqa: E402
from predictor import RainPredictor, now_pkt, validate_manual  # noqa: E402

FIGURES = ROOT / "reports" / "figures"


# ----------------------------------------------------------------------------- loading (cached)
@st.cache_resource(show_spinner="Loading the model…")
def load_predictor() -> RainPredictor:
    return RainPredictor()


try:
    P = load_predictor()
    LOAD_ERROR = None
except Exception as e:  # show a clear message instead of a crashed app
    P, LOAD_ERROR = None, f"{type(e).__name__}: {e}"


@st.cache_data(ttl=900, show_spinner=False)
def live_weather(city: str):
    """Today's weather + last 7 days for one city (cached 15 minutes)."""
    return P.live_row(city)


def show(h: str) -> None:
    st.html(ui.compact(h))


def stretch(widget, *args, **kwargs):
    """Full-width widgets on any recent Streamlit version."""
    try:
        return widget(*args, width="stretch", **kwargs)
    except TypeError:
        return widget(*args, **kwargs)


# ----------------------------------------------------------------------------- page
st.html(f"<style>{ui.CSS}</style>")
if P is None:
    st.error(f"Model could not be loaded: {LOAD_ERROR}")
    st.stop()

CITIES = P.cities
DEFAULT_CITY = "Lahore" if "Lahore" in CITIES else CITIES[0]
show(ui.hero(P))

tab_live, tab_compare, tab_manual, tab_about = st.tabs(
    ["Live forecast", "Compare cities", "Manual input", "About the model"])

# ----------------------------------------------------------------------------- 1. live forecast
with tab_live:
    # city picker across the top, then results in two balanced columns
    city = st.segmented_control(
        "Choose a city", CITIES, default=DEFAULT_CITY, key="live_city")
    city = city or DEFAULT_CITY
    region = ui.CITY_META.get(city, "")
    st.caption(f"{region} · Today's observed + forecast weather and the past 7 days are fetched live from "
               "Open-Meteo to predict **tomorrow** (rain = at least 1 mm).")
    try:
        with st.spinner(f"Fetching today's weather for {city}…"):
            X, day, hist, shown = live_weather(city)
        p = P.predict(X)
        tomorrow = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%A, %d %B %Y")
        left, right = st.columns(2, gap="large")
        with left:
            show(ui.verdict_card(p, P.threshold, f"{city} · {region}", f"Forecast for {tomorrow}")
                 + ui.tiles(shown))
        with right:
            show(ui.history_bars(hist))
            st.pyplot(ui.explain_plot(P, X))
    except requests.RequestException:
        st.error("Couldn't reach the live weather service (Open-Meteo). Please try again in a minute, "
                 "or use the Manual input tab.")

# ----------------------------------------------------------------------------- 2. compare cities
with tab_compare:
    st.caption("Tomorrow's chance of rain for every city, using today's live weather. "
               "Cities are ranked from wettest to driest.")
    results, failed = [], []
    with st.spinner("Fetching live weather for all cities…"):
        for c in CITIES:
            try:
                Xc, _, _, vc = live_weather(c)
                results.append((c, P.predict(Xc), vc))
            except requests.RequestException:
                failed.append(c)
    if results:
        when = (now_pkt() + pd.Timedelta(days=1)).strftime("%A, %d %B %Y")
        show(ui.compare_html(results, P.threshold, when, failed))
    else:
        st.error("Couldn't reach the live weather service (Open-Meteo). Please try again in a minute.")

# ----------------------------------------------------------------------------- 3. manual input
with tab_manual:
    left, right = st.columns([6, 5], gap="large")
    with left:
        c1, c2 = st.columns(2)
        m_city = c1.selectbox("📍 City", CITIES, index=CITIES.index(DEFAULT_CITY), key="m_city")
        m_month = c2.selectbox("📅 Month", list(ui.MONTHS), index=now_pkt().month - 1,
                               format_func=ui.MONTHS.get, key="m_month")
        st.caption("The form starts with typical values for the chosen city and month — change any of them.")
        d = ui.typical_values(P, m_city, m_month)
        k = f"{m_city}_{m_month}"  # new city/month -> fresh typical defaults

        def clip(v, lo, hi):
            return float(min(max(v, lo), hi))

        with st.form("manual_form"):
            st.markdown("**🌡️ Temperature & moisture**")
            a, b = st.columns(2)
            tmax = a.number_input("Max temperature (°C)", -30.0, 55.0, clip(d[0], -30, 55), 0.5, key=f"tmax_{k}")
            dew = b.number_input("Dew point (°C)", -40.0, 40.0, clip(d[2], -40, 40), 0.5, key=f"dew_{k}")
            humidity = st.slider("Mean relative humidity (%)", 0, 100, int(round(clip(d[1], 0, 100))), key=f"rh_{k}")
            st.markdown("**🧭 Pressure, cloud & wind**")
            a, b = st.columns(2)
            pressure = a.number_input("Mean sea-level pressure (hPa)", 930.0, 1060.0, clip(d[3], 930, 1060), 0.5,
                                      key=f"p_{k}")
            p_change = b.number_input("Pressure change since yesterday (hPa)", -30.0, 30.0, clip(d[4], -30, 30), 0.1,
                                      key=f"dp_{k}")
            clouds = st.slider("Mean cloud cover (%)", 0, 100, int(round(clip(d[5], 0, 100))), key=f"cc_{k}")
            a, b = st.columns(2)
            wind = a.number_input("Max wind speed (km/h)", 0.0, 200.0, clip(d[6], 0, 200), 1.0, key=f"w_{k}")
            wind_dir = b.slider("Wind direction (° from north · 90 = E · 270 = W)", 0, 360,
                                int(round(clip(d[7], 0, 360) / 5) * 5), 5, key=f"wd_{k}")
            st.markdown("**☔ Recent rain**")
            a, b, c = st.columns(3)
            rain_today = a.number_input("Rain today (mm)", 0.0, 500.0, clip(d[8], 0, 500), 0.5, key=f"r0_{k}")
            rain_3d = b.number_input("Last 3 days (mm)", 0.0, 1000.0, clip(d[9], 0, 1000), 0.5, key=f"r3_{k}")
            rain_7d = c.number_input("Last 7 days (mm)", 0.0, 2000.0, clip(d[10], 0, 2000), 0.5, key=f"r7_{k}")
            submitted = stretch(st.form_submit_button, "🔮 Predict", type="primary")

        if submitted:
            problems = validate_manual(tmax, humidity, dew, pressure, clouds, wind, wind_dir,
                                       rain_today, rain_3d, rain_7d)
            if problems:
                st.session_state["manual"] = {"errors": problems}
            else:
                X, shown = P.manual_row(m_city, int(m_month), tmax, humidity, dew, pressure, p_change, clouds,
                                        wind, wind_dir, rain_today, rain_3d, rain_7d)
                st.session_state["manual"] = {"X": X, "shown": shown, "p": P.predict(X),
                                              "city": m_city, "month": int(m_month)}
        st.caption("The 5 NASA POWER inputs (cloud amount, rain, surface pressure, humidity, specific humidity) "
                   "are estimated from the values above.")

    with right:
        res = st.session_state.get("manual")
        if not res:
            show(ui.empty_state("Enter today's weather and press Predict.", "✍️"))
        elif "errors" in res:
            st.error("Please fix these inputs:\n\n" + "\n".join(f"- {e}" for e in res["errors"]))
        else:
            show(ui.verdict_card(res["p"], P.threshold, f"{res['city']} · {ui.MONTHS[res['month']]}",
                                 "Based on the values you entered") + ui.tiles(res["shown"]))
            st.pyplot(ui.explain_plot(P, res["X"]))

# ----------------------------------------------------------------------------- 4. about
with tab_about:
    show(ui.about_html(P))
    figs = [("09_roc_pr_curves.png", "ROC and precision–recall curves (validation 2023)"),
            ("12_test_confusion_roc.png", "Confusion matrix and ROC curve (test 2024)"),
            ("14_feature_importance.png", "Permutation feature importance"),
            ("06_seasonality_city_month.png", "Chance of rain by city and month")]
    figs = [(FIGURES / f, cap) for f, cap in figs if (FIGURES / f).exists()]
    if figs:
        st.markdown("**📈 Evaluation figures**")
        cols = st.columns(2)
        for i, (path, cap) in enumerate(figs):
            with cols[i % 2]:
                stretch(st.image, str(path), caption=cap)

show(ui.FOOTER)
