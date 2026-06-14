# 30-Day Readmission Risk Predictor (MIMIC-III Demo)

A machine learning pipeline that predicts 30-day hospital readmission risk
using structured EHR data (demographics, diagnoses, labs) from the
[MIMIC-III Clinical Database Demo](https://physionet.org/content/mimiciii-demo/1.4/),
with an interactive Streamlit app for exploring model predictions and
SHAP-based explanations.

## ⚠️ Disclaimer
This project uses the **open, de-identified MIMIC-III demo dataset**
(100 patients) for demonstration and educational purposes only. It is
**not validated for clinical use** and should not inform real patient care
decisions.

## Project overview

- **Task:** Predict whether a patient will be readmitted within 30 days
  of hospital discharge.
- **Data:** MIMIC-III demo (admissions, patients, diagnoses, labevents).
- **Models:** Logistic Regression (baseline) and XGBoost, evaluated with
  5-fold cross-validated AUROC.
- **Interpretability:** SHAP values for global feature importance and
  per-patient explanations.
- **App:** Streamlit interface to explore individual patient risk scores
  and the features driving each prediction.

## Pipeline

1. **Cohort definition** — adult inpatient admissions, excluding deaths,
   with a 30-day readmission label derived from each patient's next
   admission date.
2. **Feature engineering**
   - Demographics: age, gender, insurance, marital status, admission type
   - Diagnoses: comorbidity flags (diabetes, CHF, renal failure, COPD,
     liver disease, cancer) derived from ICD-9 codes
   - Prior admission count
   - Labs: creatinine, sodium, potassium, hemoglobin, WBC, glucose, BUN,
     bicarbonate, platelets — last/min/max/mean per admission, with
     missing-value indicators
3. **Modeling** — Logistic Regression and XGBoost, 5-fold CV AUROC
4. **Interpretability** — SHAP TreeExplainer (global + per-patient)

## Results (MIMIC-III demo subset)

| Model | CV AUROC (5-fold) |
|---|---|
| Logistic Regression | 0.745 ± 0.276 |
| XGBoost | 0.701 ± 0.211 |

**Cohort:** 88 admissions, 66 unique patients, 30-day readmission rate ≈ 8%.

## ⚠️ Limitations

- **Sample size:** The demo dataset contains only 100 patients (88
  qualifying admissions, 7 positive labels). Cross-validated AUROC
  estimates carry very wide uncertainty (±0.2–0.3) and should not be
  interpreted as production-grade performance.
- **Single-center data:** MIMIC-III is sourced from a single Boston
  teaching hospital (2001–2012); findings may not generalize.
- **Label leakage caution:** All features are computed using only data
  available up to discharge time to avoid leakage from future encounters.
- **Demographic features:** Some SHAP-identified associations (e.g.,
  gender) likely reflect small-sample noise rather than clinically
  meaningful relationships and should not be used to inform care
  decisions.
- **Scaling path:** This pipeline is designed to scale to the full
  MIMIC-III/IV databases (40,000+ admissions) via Google BigQuery, which
  would substantially improve statistical reliability.
