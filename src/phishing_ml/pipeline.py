"""Preprocessing + model pipeline.

`StandardScaler` on numeric columns, `OneHotEncoder` on categorical ones,
feeding a classifier. Which column goes where comes from the `FeatureSet`,
so the pipeline stays agnostic to whether it is scoring URLs or emails.

High-cardinality categoricals (`sender_domain` has 24k+ distinct values) are
handled by the encoder's own `min_frequency` grouping: domains seen fewer
than `min_frequency` times in the *training split* collapse into a single
"infrequent" category, and unseen domains at score time join them. That
keeps frequent domains as real signal instead of discarding them against a
hardcoded allowlist, while bounding the encoded width and avoiding a
category explosion.
"""
from typing import Optional

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .featureset import FeatureSet

CLASSIFIERS = ("logistic_regression", "random_forest", "hist_gradient_boosting")

# Domains rarer than this in the training split are pooled rather than each
# getting their own near-empty column. Swept over the full email corpus at 5 /
# 20 / 100: every classifier scored better the lower this went (logistic
# regression most sharply, 0.855 -> 0.900 ROC AUC), so keep it low enough to
# retain real sender signal. Below ~5 the encoded width grows fast for
# negligible gain.
DEFAULT_MIN_FREQUENCY = 5


def build_preprocessor(
    feature_set: FeatureSet,
    min_frequency: Optional[int] = DEFAULT_MIN_FREQUENCY,
) -> ColumnTransformer:
    encoder = OneHotEncoder(
        handle_unknown="infrequent_if_exist" if min_frequency else "ignore",
        min_frequency=min_frequency,
        sparse_output=False,
    )
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(feature_set.numeric)),
            ("cat", encoder, list(feature_set.categorical)),
        ]
    )


def build_classifier(name: str, random_state: int = 42):
    """Instantiate a classifier by name.

    `class_weight="balanced"` on the two that support it keeps the minority
    class from being ignored when a corpus is lopsided.
    """
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced")
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1, class_weight="balanced"
        )
    if name == "hist_gradient_boosting":
        # Histogram-based boosting: the strongest of the three on large,
        # mostly-numeric feature frames, and it scales near-linearly in rows.
        return HistGradientBoostingClassifier(random_state=random_state)
    raise ValueError(f"Unknown classifier {name!r}; expected one of {CLASSIFIERS}")


def build_pipeline(
    feature_set: FeatureSet,
    classifier: str = "logistic_regression",
    min_frequency: Optional[int] = DEFAULT_MIN_FREQUENCY,
    random_state: int = 42,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_set, min_frequency)),
            ("classifier", build_classifier(classifier, random_state)),
        ]
    )
