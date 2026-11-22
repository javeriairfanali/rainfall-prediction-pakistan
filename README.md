# 🌧️ RainCast Pakistan — Weather-Based Rainfall Prediction

Predicts whether it will **rain tomorrow (≥ 1 mm)** in 4 Pakistani cities (Karachi, Lahore, Islamabad, Quetta) from today's weather and the
last 7 days, using 5 years (2020–2024) of data combined from **Open-Meteo** and **NASA POWER**.

| | Link |
|---|---|
| 🚀 Live app (Streamlit Community Cloud) | https://raincast-pakistan.streamlit.app |

---

## Problem
- **Task:** binary classification. Target `rain_tomorrow` = 1 if next-day precipitation ≥ 1 mm.
- **User:** **farmers in Pakistan's main farming regions**, deciding whether to irrigate, spray pesticide or dry a harvest tomorrow. Irrigating before a rain day wastes water and money, and spraying before rain washes the pesticide off. A missed rain day is costly, so the metric focuses on the rain class.
- **Why these 4 cities:** each represents a different farming system and rain pattern.

| City | Farming region it represents | Why rain tomorrow matters |
|---|---|---|
| Lahore | Punjab irrigated plains | irrigation and spraying decisions; monsoon rice and cotton |
| Karachi | Lower Sindh coast | coastal vegetable farming; monsoon rain |
| Islamabad | Pothohar plateau (barani, rain-fed) | wheat and other crops depend almost entirely on rain |
| Quetta | Balochistan valleys | apple and grape orchards depend on winter rain |

- **Success metric:** F1-score of the rain class on the held-out 2024 test set, with ROC-AUC and PR-AUC.

## Data
| Source | What we take | Licence |
|---|---|---|
| [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) (ERA5 reanalysis) | 12 daily variables + 4 hourly variables aggregated to 7 daily statistics | CC BY 4.0 |
| [NASA POWER Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/) (MERRA-2 / CERES) | 10 daily variables | free, acknowledgement requested |

4 cities × 2020–2024 = **7,304 rows**. We collect 40 features and keep **16** (15 numeric + city), selected on the
**training years only** (permutation importance + removing |r| ≥ 0.9 duplicates), so the validation and test
years never influence the choice. Both sources are represented.
Full provenance (collection method, licences, download date) is in [`data/README.md`](data/README.md) and
`data/dataset_info.json`. No personal data.

## ML lifecycle (all in `notebooks/rainfall_prediction.ipynb`)
1. **EDA:** class balance, missing values, distributions, outliers, correlations (5 plots).
2. **Pre-processing in one scikit-learn `Pipeline`:** feature engineering → 1–99 % clipping → median imputation → scaling + one-hot. Fitted on training data only, so there is no leakage.
3. **Time-ordered split:** train 2020–2022, validation 2023, test 2024.
4. **Models:** dummy and persistence baselines, plus Logistic Regression, KNN, Random Forest and SVM (RBF kernel).
5. **Tuning:** `RandomizedSearchCV` with 5-fold forward-chaining time-series CV.
6. **Evaluation:** F1 / precision / recall / ROC-AUC / PR-AUC, threshold tuned on validation, test set used once, learning curve, error analysis, permutation importance.
7. **Saving:** `models/rainfall_pipeline.joblib` + metadata; versions pinned in `requirements.txt`.

## Repository structure
```
data/            README.md (data documentation) · weather_pakistan.csv · dataset_info.json  (raw/ is git-ignored)
src/             download_data.py (collects + merges the two sources) · rainfall_features.py (shared feature code)
notebooks/       rainfall_prediction.ipynb (EDA → model → evaluation → saving)
models/          rainfall_pipeline.joblib · model_metadata.json · result tables
reports/figures/ all plots used in the report
app/             app.py (Streamlit page) · ui.py (cards, charts, styling) · predictor.py (model, live weather, explanations)
                 requirements.txt (what Streamlit Community Cloud installs)
.streamlit/      config.toml (dark theme)
requirements.txt
```

## Setup and running (Windows, macOS or Linux; Python 3.13)
```bash
# 1. environment
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2. data (about 2–5 minutes, downloads from the two APIs)
python src/download_data.py

# 3. train and evaluate: open the notebook and choose Run All (about 10–25 minutes)
jupyter lab notebooks/rainfall_prediction.ipynb

# 4. run the app locally (opens http://localhost:8501)
streamlit run app/app.py
```

## Deployment — Streamlit Community Cloud
- **Live app:** https://raincast-pakistan.streamlit.app, deployed from this GitHub repository on Streamlit Community Cloud (free), entrypoint
  `app/app.py`, Python 3.13. Streamlit Cloud installs the pinned versions in `app/requirements.txt`, the same
  versions used for training, so the saved pipeline loads correctly.
- **The app has four tabs:**
  - **Live forecast:** pick a city; today's weather and the last 7 days are fetched from Open-Meteo.
  - **Compare cities:** tomorrow's chance of rain for all 4 cities, ranked.
  - **Manual input:** a validated form, pre-filled with typical values for the chosen city and month.
  - **About the model:** metrics, features, data sources and evaluation figures.

  Each prediction shows the rain probability against the decision threshold, farm advice, and a chart explaining
  which inputs pushed the prediction towards rain or dry weather.

## Results
**Best model:** Random Forest. Decision threshold 0.57, tuned for F1 on the 2023 validation year.

| Test set (2024, used once) | F1 | Precision | Recall | Accuracy | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Random Forest | 0.586 | 0.571 | 0.602 | 0.857 | 0.879 | 0.636 |
| Persistence baseline | 0.528 | 0.528 | 0.528 | 0.841 | 0.717 | 0.359 |

Validation (2023), F1 at threshold 0.5:

| Model | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| Random Forest | 0.590 | 0.886 | 0.625 |
| SVM (RBF kernel) | 0.492 | 0.876 | 0.601 |
| Logistic Regression | 0.578 | 0.875 | 0.599 |
| K-Nearest Neighbours | 0.455 | 0.864 | 0.587 |
| Baseline: Persistence | 0.533 | 0.714 | 0.371 |
| Baseline: Dummy (stratified) | 0.143 | 0.480 | 0.180 |

## References
- Zippenfenig, P. (2023). *Open-Meteo.com Weather API*. Zenodo. doi:10.5281/zenodo.7970649
- Hersbach, H. et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146, 1999–2049.
- NASA Langley Research Center (LaRC) POWER Project, funded through the NASA Earth Science Division. https://power.larc.nasa.gov
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
- Streamlit (2026). Streamlit — a faster way to build and share data apps. https://streamlit.io
