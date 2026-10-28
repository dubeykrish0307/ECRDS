import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.features.feature_engineering import prepare_features
from src.explainability.shap_explainer import (
    create_shap_explainer,
    compute_shap_values,
    plot_global_feature_importance,
    plot_local_explanation,
    generate_decision_report
)

from src.reports.decision_report_generator import generate_decision_report_file

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
    df_raw = pd.read_csv(
        "data/raw/german_credit.data",
        sep=" ",
        header=None,
        names=[
            "checking_account_status",
            "loan_duration_months",
            "credit_history",
            "loan_purpose",
            "credit_amount",
            "savings_account",
            "employment_duration",
            "installment_rate",
            "personal_status_sex",
            "other_debtors",
            "residence_duration",
            "property",
            "age",
            "other_installment_plans",
            "housing",
            "existing_credits",
            "job",
            "num_dependents",
            "telephone",
            "foreign_worker",
            "credit_risk"
        ]
    )

    df = prepare_features(df_raw)

    X = df.drop(columns=["credit_risk"])
    y = df["credit_risk"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_scaled, y)

    return df_raw, df, X, X_scaled, model, scaler

df_raw, df, X, X_scaled, model, scaler = load_and_train()
explainer = create_shap_explainer(model, X_scaled)
shap_values = compute_shap_values(explainer, X_scaled)

st.sidebar.header("Select Applicant")
index = st.sidebar.slider(
    "Applicant Index",
    min_value=0,
    max_value=len(df_raw) - 1,
    value=0
)

applicant_scaled = X_scaled[index].reshape(1, -1)
risk_prob = model.predict_proba(applicant_scaled)[0, 1]

st.subheader("Risk Assessment")
st.metric("Default Risk Probability", f"{risk_prob:.2%}")

if risk_prob < 0.3:
    decision = "APPROVE"
elif risk_prob < 0.6:
    decision = "MANUAL REVIEW"
else:
    decision = "REJECT"

st.markdown(f"### Decision: **{decision}**")


st.subheader("Global Risk Drivers")
plot_global_feature_importance(shap_values, X)
st.pyplot(plt.gcf(), clear_figure=True)

st.subheader("Local Explanation")
plot_local_explanation(explainer, shap_values, X, index)
st.pyplot(plt.gcf(), clear_figure=True)


st.subheader("Decision Justification")
report = generate_decision_report(X, shap_values, index, threshold=0.02)
st.text(report)

st.subheader("Audit & Compliance")

if st.button("Generate Audit Decision Report"):
    report_path = generate_decision_report_file(
        applicant_index=index,
        risk_prob=risk_prob,
        justification_text=report
    )

    st.success("Decision report generated successfully.")
    st.code(str(report_path))

