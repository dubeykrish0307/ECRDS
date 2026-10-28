import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.features.feature_engineering import load_raw_data, prepare_features


def main():
    df = prepare_features(load_raw_data())
    X = df.drop(columns=["credit_risk"])
    y = df["credit_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # logistic regression gets scaled inputs, the forest doesn't need them
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    log_model.fit(X_train_scaled, y_train)
    log_pred = log_model.predict(X_test_scaled)
    log_prob = log_model.predict_proba(X_test_scaled)[:, 1]

    rf_model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]

    results = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy": [accuracy_score(y_test, log_pred), accuracy_score(y_test, rf_pred)],
        "ROC_AUC": [roc_auc_score(y_test, log_prob), roc_auc_score(y_test, rf_prob)],
    })
    print(results)

    print("Logistic Regression Report:\n")
    print(classification_report(y_test, log_pred))
    print("\nRandom Forest Report:\n")
    print(classification_report(y_test, rf_pred))


if __name__ == "__main__":
    main()
