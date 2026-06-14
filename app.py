import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="30-Day Readmission Risk",
    page_icon="🏥",
    layout="wide"
)

ARTIFACTS_DIR = "artifacts/"


@st.cache_resource
def load_artifacts():
    model = joblib.load(ARTIFACTS_DIR + "model.pkl")
    explainer = joblib.load(ARTIFACTS_DIR + "shap_explainer.pkl")
    X_test = pd.read_parquet(ARTIFACTS_DIR + "X_test.parquet")
    y_test = pd.read_parquet(ARTIFACTS_DIR + "y_test.parquet")
    with open(ARTIFACTS_DIR + "metrics.json") as f:
        metrics = json.load(f)
    return model, explainer, X_test, y_test, metrics


model, explainer, X_test, y_test, metrics = load_artifacts()

st.title("🏥 30-Day Readmission Risk Predictor")
st.caption(
    "Trained on the MIMIC-III Clinical Database demo subset "
    f"({metrics['cohort_size']} admissions, {metrics['unique_patients']} patients). "
    "For demonstration purposes only — not for clinical use."
)

tab1, tab2 = st.tabs(["Patient Risk Explorer", "Model Performance"])

# ---------------------------------------------------------------------
# TAB 1: Patient Risk Explorer
# ---------------------------------------------------------------------
with tab1:
    st.subheader("Select a patient (test set)")

    patient_idx = st.selectbox(
        "Patient (by test set row index)",
        options=list(range(len(X_test))),
        format_func=lambda i: f"Patient {i}"
    )

    patient_features = X_test.iloc[[patient_idx]]
    actual_label = y_test.iloc[patient_idx, 0]

    risk_score = model.predict_proba(patient_features)[0, 1]

    if risk_score >= 0.5:
        risk_tier = "High"
        risk_color = "🔴"
    elif risk_score >= 0.2:
        risk_tier = "Medium"
        risk_color = "🟡"
    else:
        risk_tier = "Low"
        risk_color = "🟢"

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted 30-day readmission risk", f"{risk_score:.1%}")
    col2.metric("Risk tier", f"{risk_color} {risk_tier}")
    col3.metric("Actual outcome (held-out label)", "Readmitted" if actual_label == 1 else "Not readmitted")

    st.divider()
    st.subheader("What's driving this prediction?")

    shap_values = explainer.shap_values(patient_features)

    # Handle both old (array) and new (Explanation) SHAP API return types
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
        base_value = explainer.expected_value[1]
    else:
        sv = shap_values[0]
        base_value = explainer.expected_value

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(
        shap.Explanation(
            values=sv,
            base_values=base_value,
            data=patient_features.iloc[0].values,
            feature_names=patient_features.columns.tolist()
        ),
        max_display=10,
        show=False
    )
    st.pyplot(fig, bbox_inches="tight")
    plt.close(fig)

    st.caption(
        "Red bars push the predicted risk higher; blue bars push it lower. "
        "Values shown are the patient's actual feature values."
    )

    with st.expander("View raw feature values for this patient"):
        st.dataframe(patient_features.T.rename(columns={patient_features.index[0]: "value"}))

# ---------------------------------------------------------------------
# TAB 2: Model Performance
# ---------------------------------------------------------------------
with tab2:
    st.subheader("Cohort & model summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Admissions in cohort", metrics["cohort_size"])
    col2.metric("Unique patients", metrics["unique_patients"])
    col3.metric("30-day readmission rate", f"{metrics['positive_rate']:.1%}")

    st.divider()
    st.subheader("Cross-validated AUROC (5-fold)")

    col1, col2 = st.columns(2)
    col1.metric(
        "Logistic Regression",
        f"{metrics['logreg_auroc_cv']:.3f}",
        help=f"± {metrics['logreg_auroc_cv_std']:.3f} across folds"
    )
    col2.metric(
        "XGBoost",
        f"{metrics['xgb_auroc_cv']:.3f}",
        help=f"± {metrics['xgb_auroc_cv_std']:.3f} across folds"
    )

    st.divider()
    st.subheader("Global feature importance (SHAP)")

    shap_values_all = explainer.shap_values(X_test)
    if isinstance(shap_values_all, list):
        sv_all = shap_values_all[1]
    else:
        sv_all = shap_values_all

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    shap.summary_plot(sv_all, X_test, max_display=15, show=False)
    st.pyplot(fig2, bbox_inches="tight")
    plt.close(fig2)

    st.divider()
    st.warning(
        "**Limitations:** This model is trained on the MIMIC-III demo subset "
        f"(n={metrics['cohort_size']} admissions, {metrics['unique_patients']} patients), "
        "which is far too small for reliable clinical metrics — cross-validated AUROC "
        "estimates carry wide uncertainty (±0.2–0.3). This app is a methodology "
        "demonstration only. A production model would be trained on the full "
        "MIMIC-III/IV database (40,000+ admissions) with external validation."
    )
