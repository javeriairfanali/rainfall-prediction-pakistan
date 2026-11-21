---
license: cc-by-4.0
library_name: sklearn
pipeline_tag: tabular-classification
tags:
  - sklearn
  - tabular-classification
  - weather
  - rainfall
  - pakistan
---

# Rainfall Prediction for Pakistan (next-day rain, binary)

Predicts whether it will rain **tomorrow** (≥ 1 mm) in a Pakistani city from today's weather and the last
7 days' history. Final model: **Random Forest** inside a scikit-learn `Pipeline`
(feature engineering → outlier clipping → median imputation → scaling / one-hot → classifier).

## Intended use
Decision support for **farmers** in Pakistan's main farming regions: whether to irrigate, spray or dry a
harvest tomorrow. Study cities represent Punjab irrigated plains (Lahore), lower Sindh (Karachi), rain-fed
Pothohar (Islamabad) and Balochistan orchards (Quetta). **Not** an official weather forecast — consult the Pakistan Meteorological Department for warnings.

## Data
| | |
|---|---|
| Sources | Open-Meteo Historical Weather API (ERA5, CC BY 4.0) + NASA POWER daily API (MERRA-2/CERES) |
| Cities | Islamabad, Karachi, Lahore, Quetta |
| Period | 2020–2024, 7,304 city-days, 16 input features |
| Target | `rain_tomorrow` = next-day precipitation ≥ 1 mm (rain rate 16.7%) |
| Split | time-ordered — train 2020-01-01 → 2022-12-31, validation 2023-01-01 → 2023-12-31, test 2024-01-01 → 2024-12-30 |

## Training
Baselines (stratified dummy, persistence) + Logistic Regression, KNN, Random Forest and an RBF-kernel SVM,
each tuned with `RandomizedSearchCV` and 5-fold forward-chaining time-series CV (scored by PR-AUC).
Decision threshold **0.573** chosen to maximise F1 on the validation year.

Validation (2023), threshold 0.5 for F1:

| Model | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| Random Forest | 0.590 | 0.886 | 0.625 |
| SVM (RBF kernel) | 0.492 | 0.876 | 0.601 |
| Logistic Regression | 0.578 | 0.875 | 0.599 |
| K-Nearest Neighbours | 0.455 | 0.863 | 0.586 |
| Baseline: Persistence | 0.533 | 0.714 | 0.371 |
| Baseline: Dummy (stratified) | 0.143 | 0.480 | 0.180 |

## Test results (2024, held out, evaluated once)
| Metric | Random Forest | Persistence baseline |
|---|---|---|
| F1 (rain) | 0.586 | 0.528 |
| Precision | 0.571 | 0.528 |
| Recall | 0.602 | 0.528 |
| ROC-AUC | 0.879 | — |
| PR-AUC | 0.636 | — |
| Accuracy | 0.857 | 0.841 |

Most important inputs (permutation importance): `nasa_cloud_amt`, `nasa_prectotcorr`, `precipitation_sum`, `cloud_cover_mean`, `wind_direction_10m_dominant`, `nasa_ps`, `precip_3d_sum`, `pressure_change_24h`.

## Limitations
- Inputs are reanalysis data (~25 km grid), not station observations; local storms can be missed.
- Trained on 4 cities only; other locations are handled (unknown city → no city effect) but less reliable.
- Rare dry-season rain is harder to predict than monsoon rain; see the error analysis in the notebook.
- Live predictions in the demo app use Open-Meteo forecast data for "today", and approximate the NASA POWER
  inputs from Open-Meteo equivalents.

## How to use
```python
import joblib, pandas as pd
from huggingface_hub import hf_hub_download
import rainfall_features  # must be importable (downloaded from this repo)
model = joblib.load(hf_hub_download("<user>/rainfall-prediction-pakistan", "rainfall_pipeline.joblib"))
proba = model.predict_proba(df_with_input_columns)[:, 1]
```
Requires scikit-learn 1.9.1 (same version used for training).

## Citation / acknowledgements
- Zippenfenig, P. (2023). *Open-Meteo.com Weather API*. Zenodo. doi:10.5281/zenodo.7970649 (data CC BY 4.0).
- Hersbach, H. et al. (2020). The ERA5 global reanalysis. *Q. J. R. Meteorol. Soc.* 146, 1999–2049.
- NASA Langley Research Center (LaRC) POWER Project, funded through the NASA Earth Science Division.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR* 12, 2825–2830.
