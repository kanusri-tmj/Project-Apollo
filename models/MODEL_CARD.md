# Model Card — Heart Disease Risk (Logistic Regression, Ridge (L2))

**Version:** `1.0.0`
**Artifact:** `ridge_l2_v1.0.0_20260926T062633Z.joblib`
**Generated:** 2026-09-26T06:26:33.814108+00:00

## Intended use
Decision-support for clinicians: estimate the probability that a patient has
coronary artery disease from routine, non-invasive clinical measurements.
**Not a diagnostic device.** Outputs must be interpreted by a qualified
clinician alongside the full clinical picture.

## Model
- Algorithm: **Ridge (L2)** — logistic regression with L2 (Ridge) penalty.
- Selected hyper-parameters: `classifier__C=3.1622776601683795`
- Preprocessing: median/mode imputation, one-hot encoding, standard scaling, optional engineered features.
- Trained on: UCI Cleveland Heart Disease dataset (303 records), binary target.

## Performance (held-out test set, 20% stratified split)
| Metric | Value |
| --- | --- |
| accuracy | 0.9016 |
| precision | 0.8667 |
| recall | 0.9286 |
| f1 | 0.8966 |
| roc_auc | 0.9654 |
| threshold | 0.5000 |

## Inputs
| Field | Type | Range / options | Description |
| --- | --- | --- | --- |
| age | number | 18 - 100 | Age in years |
| resting_bp | number | 80 - 220 | Resting blood pressure (mm Hg) |
| cholesterol | number | 100 - 600 | Serum cholesterol (mg/dl) |
| max_heart_rate | number | 60 - 220 | Maximum heart rate achieved |
| st_depression | number | 0.0 - 7.0 | ST depression induced by exercise (oldpeak) |
| sex | categorical | {'0': 'Female', '1': 'Male'} | Sex |
| chest_pain_type | categorical | {'1': 'Typical angina', '2': 'Atypical angina', '3': 'Non-anginal pain', '4': 'Asymptomatic'} | Chest pain type |
| fasting_blood_sugar | categorical | {'0': '<= 120 mg/dl', '1': '> 120 mg/dl'} | Fasting blood sugar |
| resting_ecg | categorical | {'0': 'Normal', '1': 'ST-T abnormality', '2': 'LV hypertrophy'} | Resting ecg |
| exercise_angina | categorical | {'0': 'No', '1': 'Yes'} | Exercise angina |
| st_slope | categorical | {'1': 'Upsloping', '2': 'Flat', '3': 'Downsloping'} | St slope |
| num_major_vessels | categorical | {'0': '0', '1': '1', '2': '2', '3': '3'} | Num major vessels |
| thalassemia | categorical | {'3': 'Normal', '6': 'Fixed defect', '7': 'Reversible defect'} | Thalassemia |

## Features used
age, resting_bp, cholesterol, max_heart_rate, st_depression, sex, chest_pain_type, fasting_blood_sugar, resting_ecg, exercise_angina, st_slope, num_major_vessels, thalassemia

## Limitations & ethical notes
- Trained on a small historical cohort (Cleveland, 1988) — not demographically representative.
- Risk probabilities are not calibrated to any local patient population.
- Do not use as the sole basis for any clinical decision.
