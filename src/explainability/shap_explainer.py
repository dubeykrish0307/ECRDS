import matplotlib.pyplot as plt
import shap


def create_shap_explainer(model, X_train):
    return shap.LinearExplainer(model, X_train, feature_perturbation="interventional")


def compute_shap_values(explainer, X):
    return explainer.shap_values(X)


def plot_global_feature_importance(shap_values, X):
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    return plt.gcf()


def plot_local_explanation(explainer, shap_values, X, index):
    shap.force_plot(
        explainer.expected_value,
        shap_values[index],
        X.iloc[index],
        matplotlib=True,
        show=False,
    )
    plt.title(f"Local Explanation for Applicant #{index}")
    return plt.gcf()


def generate_decision_report(X, shap_values, index, threshold=0.0):
    """Plain-text list of the features that pushed this applicant's risk up or down."""
    row = X.iloc[index]
    lines = ["Decision Justification:\n"]

    for feature, value, impact in zip(row.index, row.values, shap_values[index]):
        if abs(impact) > threshold:
            direction = "increases" if impact > 0 else "reduces"
            lines.append(f"- {feature}: value={value} {direction} risk (impact={impact:.3f})")

    return "\n".join(lines)
