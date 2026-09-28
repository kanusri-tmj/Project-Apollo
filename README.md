# ❤️ Heart Disease Prediction — Ridge & Lasso Logistic Regression

Predict whether a patient has heart disease (**binary: present / absent**) from 13
routine clinical measurements, using **regularised logistic regression** as a
decision-support tool. The model returns a predicted class **and** a risk
probability, and is served through both a **Flask REST API** and a **Streamlit**
clinician UI.

> ⚠️ Research / decision-support artefact only. It is **not** a medical device and
> must not be used as the sole basis for a clinical decision.

---

## Contents

| Step | Where |
| --- | --- |
| 1. Problem definition | [`notebooks/heart_disease_analysis.ipynb`](notebooks/heart_disease_analysis.ipynb) |
| 2. Dataset loading | [`src/data_loader.py`](src/data_loader.py) |
| 3. Data understanding | [`src/eda.py`](src/eda.py) (`statistical_summary`, `report_missing`) |
| 4. Preprocessing | [`src/preprocessing.py`](src/preprocessing.py) |
| 5. Exploratory data analysis | [`src/eda.py`](src/eda.py) → `reports/figures/` |
| 6. Feature selection & engineering | [`src/preprocessing.py`](src/preprocessing.py), `train.lasso_feature_path` |
| 7. Train/test split | [`src/train.py`](src/train.py) (`run_training`) |
| 8. Model building | [`src/train.py`](src/train.py) |
| 9. Model evaluation | [`src/evaluate.py`](src/evaluate.py) |
| 10. Save model + model card | [`src/model_card.py`](src/model_card.py) |
| 11. Deployment | [`app/flask_app.py`](app/flask_app.py), [`app/streamlit_app.py`](app/streamlit_app.py) |
| Live model simulation | [`src/explain.py`](src/explain.py) + [`app/templates/index.html`](app/templates/index.html) |

---

## Project structure

```
.
├── README.md
├── requirements.txt            # full local/dev environment (training, notebook, Streamlit)
├── requirements-render.txt     # runtime-only deps installed by the Render build
├── render.yaml                 # Render blueprint: build, start, health check
├── wsgi.py                     # production entry point (`gunicorn wsgi:app`)
├── .python-version             # Python version Render builds with
├── pytest.ini
├── src/                        # reusable, importable pipeline code
│   ├── config.py               # paths, schema, hyper-parameter grids
│   ├── data_loader.py          # download + load the UCI Cleveland data
│   ├── preprocessing.py        # cleaning, validation, encoding, scaling, FE
│   ├── eda.py                  # exploratory plots + summary tables
│   ├── train.py                # 3 model variants, CV tuning, orchestration
│   ├── evaluate.py             # metrics, comparison table, diagnostic plots
│   ├── model_card.py           # versioned persistence + model card
│   ├── predict.py              # inference shared by both apps
│   └── explain.py              # real model internals for the Live Simulation
├── app/
│   ├── flask_app.py            # REST API + 3-section website
│   ├── streamlit_app.py        # Streamlit clinician UI (+ batch scoring)
│   └── templates/index.html    # Home / Live Simulation / About Us
├── notebooks/
│   └── heart_disease_analysis.ipynb   # the 11-step walkthrough
├── tests/                      # pytest suite (69 tests, incl. deployment contract)
├── data/                       # raw + processed CSVs (generated)
├── models/                     # versioned artifacts, metadata, MODEL_CARD.md
└── reports/                    # comparison tables + figures (generated)
```

---

## Dataset

**UCI Cleveland Heart Disease** — 303 records, 13 clinical features + target.

Fields: `age`, `sex`, `chest_pain_type`, `resting_bp`, `cholesterol`,
`fasting_blood_sugar`, `resting_ecg`, `max_heart_rate`, `exercise_angina`,
`st_depression`, `st_slope`, `num_major_vessels`, `thalassemia`.

`src/data_loader.py` downloads the official archive
(<https://archive.ics.uci.edu/static/public/45/heart+disease.zip>) and reads the
`processed.cleveland.data` member. The raw `num` severity column (0–4) is
binarised to **0 = no disease, 1 = disease present**. Six missing values
(`?`) live in `num_major_vessels` (4) and `thalassemia` (2) and are imputed in
the pipeline.

---

## Setup

```bash
# from the project root
python -m venv .venv
# Windows (Git Bash)
.venv/Scripts/python.exe -m pip install -r requirements.txt
# macOS / Linux
.venv/bin/python -m pip install -r requirements.txt
```

> **Windows note:** the pins `scipy>=1.15,<1.18` and `scikit-learn>=1.6,<1.8` are
> deliberate. Newer wheels (`scipy 1.18.x`, `scikit-learn 1.8/1.9`) ship native
> `.pyd` binaries that some Windows Application Control / Smart App Control
> policies block at import time. On other platforms you can relax them.

---

## Run end-to-end

```bash
# 1. Train everything: cleans data, runs EDA, tunes Ridge/Lasso, evaluates,
#    saves the best model + model card, and writes all reports.
python -m src.train

# 2. (optional) Explore interactively
jupyter notebook notebooks/heart_disease_analysis.ipynb
```

`python -m src.train` produces:

| Output | Description |
| --- | --- |
| `data/raw/heart_disease_raw.csv` | downloaded + binarised raw snapshot |
| `data/processed/heart_disease_clean.csv` | de-duplicated analysis frame |
| `reports/model_comparison.csv` | metrics for all three variants |
| `reports/figures/*.png` | histograms, box plots, heatmap, ROC, confusion matrices, Lasso path |
| `reports/lasso_feature_selection.csv` | features surviving L1 as `C` grows |
| `models/<variant>_v<version>_<ts>.joblib` | immutable versioned artifact |
| `models/<variant>_latest.joblib` | stable pointer used by the apps |
| `models/<variant>_metadata.json` | metrics, params, input spec |
| `models/MODEL_CARD.md` | human-readable model card |

---

## Models & results

Three variants are trained on the **same stratified 80/20 split**
(`random_state=42`), with `C` tuned by stratified 5-fold CV on ROC-AUC.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | best `C` | CV ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (no penalty) | 0.8852 | 0.8387 | 0.9286 | 0.8814 | 0.9610 | — | 0.8805 |
| **Ridge (L2)** | **0.9016** | **0.8667** | 0.9286 | **0.8966** | **0.9654** | 3.162 | 0.8974 |
| Lasso (L1) | 0.9016 | 0.8667 | 0.9286 | 0.8966 | 0.9632 | 3.162 | 0.9020 |

*(Numbers from the bundled run; re-running regenerates them — Ridge is selected
by test ROC-AUC.)*

**Regularisation behaves as expected:** on the 25 post-encoding features,
baseline and Ridge keep all coefficients non-zero, while **Lasso zeroes 7 of
them**, performing embedded feature selection. The strongest predictors are
`num_major_vessels`, `chest_pain_type`, `thalassemia`, `sex` and `st_slope`.

---

## Deployment

### Flask — REST API + clinician form

```bash
python app/flask_app.py          # http://localhost:5000
```

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | GET | clinician-facing HTML form |
| `/predict` (`/api/predict`) | POST | score one patient |
| `/explain` (`/api/explain`) | POST | full model internals: transformed features, coefficients, per-term contributions, intercept, logit, probability — per variant |
| `/api/predict/batch` | POST | score a list of patients |
| `/health` | GET | liveness/readiness probe |
| `/model-info` | GET | version, metrics, input specification |
| `/metrics` | GET | request/prediction counters + latency |

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"age": 57, "sex": 1, "chest_pain_type": 4, "resting_bp": 140,
        "cholesterol": 260, "fasting_blood_sugar": 0, "resting_ecg": 1,
        "max_heart_rate": 140, "exercise_angina": 1, "st_depression": 2.0,
        "st_slope": 2, "num_major_vessels": 2, "thalassemia": 7}}'
```

```json
{"prediction": 1, "label": "Disease present", "probability": 0.983,
 "risk_percent": 98.3, "threshold": 0.5, "model_version": "1.0.0"}
```

Out-of-range inputs return **422** with actionable `details`; a bad body returns
**400**; a missing model returns **503**.

### Website — Home / Live Simulation / About Us

The Flask app serves a three-section single-page site (sticky header, smooth
scroll, scroll-spy navigation):

1. **Home** — the 13-field patient form plus a colour-coded result card with an
   animated risk gauge, risk probability and verdict (present / absent).
2. **Live Simulation** — an animated, step-by-step reveal of the *actual*
   computation for the patient just submitted, powered by `POST /explain`:
   * standardised inputs (z-scores) as signed bars,
   * the logistic-regression equation built term by term with a running logit,
   * **Ridge (L2)** coefficients against their unpenalised baseline magnitudes to
     show smooth dampening,
   * **Lasso (L1)** coefficients, with exactly-zero features fading out and
     marked *dropped* — embedded feature selection made visible,
   * the sigmoid curve with a dot animating to the patient's logit,
   * the probability animating into the verdict (synced with the Home card).

   A Ridge/Lasso toggle re-runs the animation for either model, and **Replay**
   re-triggers it. Every number comes from the trained pipeline, so the visuals
   cannot drift from the model. Open [`/?demo=1`](http://localhost:5000/?demo=1)
   to auto-load and score an example patient.
3. **About Us** — project title, abstract, team details, guide, course and
   institution.

The page also surfaces uncaught JavaScript errors in a visible banner instead of
failing silently.

### Streamlit — clinician UI

```bash
streamlit run app/streamlit_app.py   # http://localhost:8501
```

Single-patient form (with a decision-threshold slider for recall/precision
trade-offs), a CSV batch-scoring tab, and an About tab rendering the model card.

### Render — one-click production deploy

The repo ships a [`render.yaml`](render.yaml) blueprint, so someone with a
Render account can deploy the Flask API + website with *New → Blueprint* and no
manual configuration:

| Setting | Value |
| --- | --- |
| Build command | `pip install -r requirements-render.txt` |
| Start command | `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 --access-logfile - wsgi:app` |
| Health check path | `/health` |
| Python | `3.14.3` ([`.python-version`](.python-version), mirrored by `PYTHON_VERSION`) |

```bash
# reproduce the production process locally (Linux/macOS)
pip install -r requirements-render.txt
gunicorn --bind 0.0.0.0:5000 wsgi:app

# or validate the entry point on any OS, without a WSGI server
PORT=5000 python wsgi.py
```

Why it is wired this way — each item is a deploy failure that has been designed
out, and covered by [`tests/test_deployment.py`](tests/test_deployment.py):

* **`wsgi:app`** — [`wsgi.py`](wsgi.py) puts the project root on `sys.path` and
  re-exports `app.flask_app.app`, so `gunicorn` never depends on the working
  directory.
* **`$PORT`** — Render injects the port. The server binds `0.0.0.0:$PORT` and the
  app reads `$PORT` (defaulting to 5000 locally); a hardcoded port is the classic
  "no open ports detected" failure.
* **`requirements-render.txt`** — the request path imports only `src.config`,
  `src.predict`, `src.explain` and `src.preprocessing`, so matplotlib, seaborn,
  jupyter and streamlit stay out of the build. Its versions are pinned
  *exactly* to the locally tested stack (numpy 2.5.3, pandas 3.0.6, scipy
  1.16.3, scikit-learn 1.7.2, flask 3.1.3), and that set resolves to prebuilt
  manylinux `cp314` wheels, so a deploy cannot silently drift onto a
  pandas/scipy the test suite never ran against — or start compiling from source.
* **Committed artifacts** — `models/<variant>_latest.joblib` and
  `models/<variant>_metadata.json` are deliberately **not** gitignored. Render's
  disk is ephemeral and every deploy rebuilds from Git, so the model must live in
  the repository. The timestamped `*_v<version>_*.joblib` audit copies stay
  ignored.
* **`/health`** — returns **503** until a model is loaded, so a deploy that ships
  without artifacts fails its health check loudly instead of quietly answering
  503 to every user.

Notes: free instances spin down when idle, so the first request after a pause
takes ~30–50 s (`/health` warms it back up). Nothing is written to disk at
runtime, so no persistent disk or volume is needed. Only the Flask app is
deployed; the Streamlit UI is a local/separate-host interface.

### Cloud deployment & monitoring

* **Container/process:** run the Flask app behind a WSGI server
  (`waitress` on Windows, `gunicorn` on Linux — see the Render section above) and
  terminate TLS at the platform ingress. Streamlit Community Cloud or a
  container host both work as-is.
* **Config:** artifacts are resolved relative to the project root, so mount
  `models/` as a volume or bake it into the image.
* **Probes:** wire `/health` to liveness/readiness checks.
* **Metrics:** `/metrics` exposes counters and latency for a basic dashboard.
  For production, export the same numbers via `prometheus_flask_exporter` and ship
  logs (each request is already logged with method, path, status and latency).
* **Versioning:** `/model-info` reports the served model version so you can track
  which artifact answered a given request — useful during rollouts.
* **Retraining:** re-run `python -m src.train`; the versioned + `_latest`
  artifacts let you promote or roll back without downtime.

---

## Tests

```bash
python -m pytest
```

69 tests cover cleaning, outlier detection, range validation, feature
engineering, pipeline output (finite, correctly shaped, standardised numerics),
inference validation, threshold behaviour, every Flask endpoint, the deployment
contract (WSGI entry point, blueprint start command and health check, runtime
requirements, that the model artifacts are not gitignored, and that the request
path imports no development-only package), and —
importantly — that the Live Simulation maths matches the model exactly:
`explain.explain_record` probabilities are asserted against
`LogisticRegression.predict_proba`, and the per-term contributions are checked to
reconstruct the logit. API tests auto-skip if no trained artifact exists yet.

The site was additionally smoke-tested in headless Chrome against a live server:
all six simulation steps complete, the sigmoid curve draws, the gauge animates,
Lasso dropout rows render, and no JavaScript error surfaces.

---

## Reproducibility & limitations

* Fixed `random_state=42` throughout; the split is stratified.
* Small historical cohort (Cleveland, 1988) — not demographically representative
  and not calibrated to any local population.
* Probabilities are model outputs, not clinically validated risk scores.
* Outliers in `cholesterol` / `resting_bp` are **flagged, not deleted**, because
  they can be genuine high-risk patients. Adjust in `src/preprocessing.py` if your
  protocol requires winsorising.
