"""Streamlit deployment: interactive clinician-facing risk calculator.

Run with::

    streamlit run app/streamlit_app.py

The app loads the same serialised pipeline as the Flask API, so predictions are
identical between the two interfaces. A CSV batch-scoring tab is included for
retrospective review of a patient list.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when Streamlit runs this file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import config, predict  # noqa: E402

st.set_page_config(page_title="Heart Disease Risk", page_icon="❤️", layout="wide")


@st.cache_resource(show_spinner="Loading model ...")
def get_model():
    """Load and cache the trained pipeline for the life of the session."""
    model, info = predict.load_model()
    return model, info


st.title("❤️ Heart Disease Risk Calculator")
st.caption(
    "Decision support using regularised logistic regression (Ridge/Lasso) trained "
    "on the UCI Cleveland cohort. **Not a diagnostic device.**"
)

try:
    model, info = get_model()
except predict.ModelNotFoundError as exc:
    st.error(str(exc))
    st.stop()

metadata = info.get("metadata", {})
metrics = metadata.get("metrics", {})
model_version = info.get("version", "unknown")

# --------------------------------------------------------------------------- #
# Sidebar — model transparency
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Model")
    st.write(f"**Variant:** {metadata.get('label', metadata.get('variant', 'n/a'))}")
    st.write(f"**Version:** {model_version}")
    if metadata.get("best_params"):
        st.write("**Hyper-parameters:**")
        st.json(metadata["best_params"])
    if metrics:
        st.write("**Held-out test metrics:**")
        st.dataframe(
            pd.DataFrame(
                {"metric": list(metrics.keys()), "value": [round(v, 4) for v in metrics.values()]}
            ),
            hide_index=True,
            width="stretch",
        )

    st.divider()
    threshold = st.slider(
        "Decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Lower the threshold to favour recall (catch more disease) at the cost of precision.",
    )

tab_single, tab_batch, tab_about = st.tabs(["Single patient", "Batch (CSV)", "About"])

# --------------------------------------------------------------------------- #
# Single patient
# --------------------------------------------------------------------------- #
with tab_single:
    with st.form("patient_form"):
        st.subheader("Patient measurements")
        numeric_values: dict[str, float] = {}
        categorical_values: dict[str, int] = {}

        columns = st.columns(3)
        defaults = {
            "age": 54.0,
            "resting_bp": 130.0,
            "cholesterol": 240.0,
            "max_heart_rate": 150.0,
            "st_depression": 1.0,
        }
        for index, feature in enumerate(config.NUMERIC_FEATURES):
            low, high = config.VALID_RANGES.get(feature, (0.0, 1000.0))
            with columns[index % 3]:
                numeric_values[feature] = st.number_input(
                    feature.replace("_", " ").capitalize(),
                    min_value=float(low),
                    max_value=float(high),
                    value=defaults.get(feature, float(low)),
                    step=1.0,
                )

        columns = st.columns(3)
        for index, feature in enumerate(config.CATEGORICAL_FEATURES):
            options = config.CATEGORY_LEVELS.get(feature, {})
            labels = [f"{label} ({code})" for code, label in options.items()]
            with columns[index % 3]:
                choice = st.selectbox(feature.replace("_", " ").capitalize(), labels)
                categorical_values[feature] = int(choice.split("(")[-1].rstrip(")"))

        submitted = st.form_submit_button("Calculate risk", width="stretch")

    if submitted:
        record = {**numeric_values, **categorical_values}
        errors = predict.validate_record(record)
        if errors:
            st.error(" ".join(errors))
        else:
            result = predict.predict_patient(
                record, model=model, threshold=threshold, validate=False
            )
            left, right = st.columns([1, 2])
            with left:
                st.metric("Risk of heart disease", f"{result['risk_percent']}%")
                if result["prediction"] == 1:
                    st.error(f"🟥 {result['label']}")
                else:
                    st.success(f"🟩 {result['label']}")
            with right:
                st.progress(min(1.0, result["probability"]))
                st.caption(
                    f"Predicted probability = {result['probability']} · "
                    f"decision threshold = {result['threshold']}"
                )
            st.info(
                "Interpret alongside the full clinical picture. The model was trained "
                "on a small historical cohort and is not calibrated to your population."
            )

# --------------------------------------------------------------------------- #
# Batch scoring
# --------------------------------------------------------------------------- #
with tab_batch:
    st.subheader("Score a patient list")
    st.caption(
        "Upload a CSV containing the 13 clinical columns. Missing or extra columns are "
        "handled gracefully (missing numeric fields fall back to the training median)."
    )
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        frame = pd.read_csv(uploaded)
        st.write("Preview", frame.head())
        if st.button("Score all patients", width="stretch"):
            records = frame.to_dict(orient="records")
            results = predict.predict_batch(records, model=model, threshold=threshold)
            output = pd.concat([frame.reset_index(drop=True), pd.DataFrame(results)], axis=1)
            st.write("Predictions", output)
            st.download_button(
                "Download results CSV",
                output.to_csv(index=False).encode("utf-8"),
                file_name="heart_disease_predictions.csv",
                mime="text/csv",
            )

# --------------------------------------------------------------------------- #
# About
# --------------------------------------------------------------------------- #
with tab_about:
    st.subheader("About this model")
    st.markdown(
        f"""
        - **Algorithm:** {metadata.get('label', 'Regularised logistic regression')}
        - **Features:** {len(config.FEATURES)} routine clinical measurements
        - **Training data:** UCI Cleveland Heart Disease (303 records, binary target)
        - **Version:** `{model_version}`
        - **Transformed feature count:** {len(metadata.get('transformed_features', []))}

        ### Input specification
        """
    )
    spec = metadata.get("input_spec", [])
    if spec:
        st.dataframe(pd.DataFrame(spec), hide_index=True, width="stretch")

    st.warning(
        "This tool is for research and decision support only. It does not replace "
        "clinical assessment and must not be used as the sole basis for diagnosis."
    )
