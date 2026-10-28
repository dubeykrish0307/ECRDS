import pandas as pd

DATA_PATH = "data/raw/german_credit.data"

# german_credit.data has no header row, so the column order comes from german.doc
COLUMN_NAMES = [
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
    "credit_risk",
]

NUMERICAL_FEATURES = [
    "loan_duration_months",
    "credit_amount",
    "installment_rate",
    "residence_duration",
    "age",
    "existing_credits",
    "num_dependents",
]

# A65 is "unknown / no savings account", treated like the lowest band
ORDINAL_CATEGORICAL_FEATURES = {
    "employment_duration": {"A71": 0, "A72": 1, "A73": 2, "A74": 3, "A75": 4},
    "savings_account": {"A61": 0, "A62": 1, "A63": 2, "A64": 3, "A65": 0},
}

NOMINAL_CATEGORICAL_FEATURES = [
    "checking_account_status",
    "credit_history",
    "loan_purpose",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, sep=" ", header=None, names=COLUMN_NAMES)


def transform_target(df: pd.DataFrame) -> pd.DataFrame:
    # the dataset codes good credit as 1 and bad as 2; we predict default, so bad -> 1
    df = df.copy()
    df["credit_risk"] = df["credit_risk"].map({1: 0, 2: 1})
    return df


def encode_ordinal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for feature, mapping in ORDINAL_CATEGORICAL_FEATURES.items():
        df[feature] = df[feature].map(mapping)
    return df


def encode_nominal_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=NOMINAL_CATEGORICAL_FEATURES, drop_first=True)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = transform_target(df)
    df = encode_ordinal_features(df)
    return encode_nominal_features(df)
