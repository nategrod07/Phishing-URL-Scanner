"""Train and evaluate a phishing classifier (URL or email mode).

Usage:
    python -m phishing_ml.train --mode url --classifier random_forest

    python -m phishing_ml.train --mode email --classifier hist_gradient_boosting \
        --out models/email_model.joblib
"""
import argparse
import os
import time
from typing import Tuple

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

from .data import (
    DEFAULT_CHUNKSIZE,
    find_kagglehub_email_dataset,
    generate_synthetic_dataset,
    load_email_features,
    load_url_csv,
)
from .email_features import EMAIL_FEATURES
from .featureset import FeatureSet
from .pipeline import CLASSIFIERS, build_pipeline
from .url_features import URL_FEATURES

TARGET_NAMES = ["legit", "phishing"]


def load_url_training_data(args) -> Tuple[pd.DataFrame, pd.Series, FeatureSet]:
    if args.csv:
        records = load_url_csv(args.csv)
    else:
        print("No --csv given; using the synthetic demo dataset.")
        records = generate_synthetic_dataset(n_per_class=args.n_per_class)
    features = URL_FEATURES.build(records)
    return features, records["label"], URL_FEATURES


def load_email_training_data(args) -> Tuple[pd.DataFrame, pd.Series, FeatureSet]:
    dataset_dir = args.email_dir or find_kagglehub_email_dataset()
    print(f"Streaming email corpus from {dataset_dir}")
    started = time.perf_counter()
    features, labels, report = load_email_features(
        dataset_dir,
        chunksize=args.chunksize,
        drop_duplicates=not args.keep_duplicates,
        sample_size=args.sample_size,
    )
    print(f"Ingest: {report.summary()} in {time.perf_counter() - started:.1f}s")
    print(f"Label balance: {labels.value_counts().sort_index().to_dict()}")
    return features, labels, EMAIL_FEATURES


def evaluate(pipeline, X_test, y_test) -> None:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, target_names=TARGET_NAMES, digits=3))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC AUC:  {roc_auc_score(y_test, y_proba):.4f}")
    print(f"PR  AUC:  {average_precision_score(y_test, y_proba):.4f}")
    print(f"Log loss: {log_loss(y_test, y_proba):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="url", choices=["url", "email"])
    parser.add_argument("--classifier", default="logistic_regression", choices=CLASSIFIERS)
    parser.add_argument("--out", default="models/phishing_model.joblib")
    parser.add_argument("--cv", type=int, default=0,
                        help="If >1, also run this many cross-validation folds")

    url_group = parser.add_argument_group("url mode")
    url_group.add_argument("--csv", help="Labelled CSV with url,label columns")
    url_group.add_argument("--n-per-class", type=int, default=800,
                           help="Rows per class for the synthetic dataset")

    email_group = parser.add_argument_group("email mode")
    email_group.add_argument("--email-dir", help="Dataset directory "
                                                 "(defaults to the kagglehub cache)")
    email_group.add_argument("--sample-size", type=int, default=None,
                             help="Cap total rows, balanced across classes "
                                  "(default: use every labelled row)")
    email_group.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE,
                             help="Rows per ingest chunk")
    email_group.add_argument("--keep-duplicates", action="store_true",
                             help="Keep duplicate emails (they leak across the "
                                  "train/test split and inflate scores)")
    args = parser.parse_args()

    if args.mode == "url":
        X, y, feature_set = load_url_training_data(args)
    else:
        X, y, feature_set = load_email_training_data(args)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(feature_set, args.classifier)
    started = time.perf_counter()
    pipeline.fit(X_train, y_train)
    print(f"\nMode: {args.mode} | Classifier: {args.classifier} | "
          f"Trained on {len(X_train):,} rows in {time.perf_counter() - started:.1f}s")

    evaluate(pipeline, X_test, y_test)

    if args.cv > 1:
        scores = cross_val_score(pipeline, X, y, cv=args.cv, scoring="roc_auc", n_jobs=-1)
        print(f"\n{args.cv}-fold CV ROC AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(pipeline, args.out)
    print(f"\nSaved model to {args.out}")


if __name__ == "__main__":
    main()
