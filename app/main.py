import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.explainability.shap_explainer import (
    compute_shap_values,
    create_shap_explainer,
    generate_decision_report,
    plot_global_feature_importance,
    plot_local_explanation,
)
from src.features.feature_engineering import load_raw_data, prepare_features
from src.reports.decision_report_generator import decision_label, generate_decision_report_file

st.set_page_config(page_title="Explainable Credit Risk System", layout="wide")

st.title("Explainable Credit Risk Decision System")
st.markdown(
    """
    This application demonstrates how machine learning models can support
    **transparent, explainable, and responsible credit decisions**.
    """
)


@st.cache_data
def load_and_train():
    df = prepare_features(load_raw_data())
    X = df.drop(columns=["credit_risk"])
    y = df["credit_risk"]

    X_scaled = StandardScaler().fit_transform(X)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_scaled, y)
    return X, X_scaled, model


X, X_scaled, model = load_and_train()
explainer = create_shap_explainer(model, X_scaled)
shap_values = compute_shap_values(explainer, X_scaled)

st.sidebar.header("Select Applicant")
index = st.sidebar.slider("Applicant Index", min_value=0, max_value=len(X) - 1, value=0)

risk_prob = model.predict_proba(X_scaled[index].reshape(1, -1))[0, 1]

st.subheader("Risk Assessment")
st.metric("Default Risk Probability", f"{risk_prob:.2%}")
st.markdown(f"### Decision: **{decision_label(risk_prob)}**")

st.subheader("Global Risk Drivers")
st.pyplot(plot_global_feature_importance(shap_values, X), clear_figure=True)

st.subheader("Local Explanation")
st.pyplot(plot_local_explanation(explainer, shap_values, X, index), clear_figure=True)

st.subheader("Decision Justification")
report = generate_decision_report(X, shap_values, index, threshold=0.02)
st.text(report)

st.subheader("Audit & Compliance")
if st.button("Generate Audit Decision Report"):
    report_path = generate_decision_report_file(
        applicant_index=index,
        risk_prob=risk_prob,
        justification_text=report,
    )
    st.success("Decision report generated successfully.")
    st.code(str(report_path))
