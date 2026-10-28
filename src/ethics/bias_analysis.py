SENSITIVE_FEATURES = ["age", "personal_status_sex", "foreign_worker"]


def outcome_rate_by_group(df, feature, target="credit_risk"):
    return df.groupby(feature)[target].mean().sort_values(ascending=False)


def prediction_rate_by_group(df, feature, prediction_col):
    return df.groupby(feature)[prediction_col].mean().sort_values(ascending=False)


def generate_bias_report(df, prediction_col="predicted_risk"):
    """Compare real default rates with the model's predicted risk for each sensitive group."""
    report = ["Bias & Fairness Assessment\n"]

    for feature in SENSITIVE_FEATURES:
        report += [
            f"\nFeature: {feature}",
            "Observed Default Rate:",
            str(outcome_rate_by_group(df, feature)),
            "\nModel Predicted Risk Rate:",
            str(prediction_rate_by_group(df, feature, prediction_col)),
        ]

    return "\n".join(report)
