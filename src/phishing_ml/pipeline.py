#Preprocessing + model pipeline: one-hot encoding for categorical URL
#features, standard scaling for numeric ones, feeding a classifier.

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .email_features import (
    CATEGORICAL_FEATURES as EMAIL_CATEGORICAL_FEATURES,
    NUMERIC_FEATURES as EMAIL_NUMERIC_FEATURES,
    extract_email_features_batch,
)
from .features import CATEGORICAL_FEATURES as URL_CATEGORICAL_FEATURES
from .features import NUMERIC_FEATURES as URL_NUMERIC_FEATURES
from .features import extract_features_batch


def build_feature_frame(urls) -> pd.DataFrame:
    return pd.DataFrame(extract_features_batch(urls))


def build_email_feature_frame(df) -> pd.DataFrame:
    return pd.DataFrame(extract_email_features_batch(df))


def build_preprocessor(
    numeric_features=URL_NUMERIC_FEATURES,
    categorical_features=URL_CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def _make_classifier(classifier: str):
    if classifier == "logistic_regression":
        return LogisticRegression(max_iter=1000)
    if classifier == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    raise ValueError(f"Unknown classifier: {classifier}")


def build_model_pipeline(
    classifier: str = "logistic_regression",
    numeric_features=URL_NUMERIC_FEATURES,
    categorical_features=URL_CATEGORICAL_FEATURES,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("classifier", _make_classifier(classifier)),
        ]
    )


def build_email_model_pipeline(classifier: str = "logistic_regression") -> Pipeline:
    return build_model_pipeline(
        classifier,
        numeric_features=EMAIL_NUMERIC_FEATURES,
        categorical_features=EMAIL_CATEGORICAL_FEATURES,
    )
