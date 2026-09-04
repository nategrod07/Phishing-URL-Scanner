"""Train and evaluate a phishing classifier (URL or email mode).

Usage:
    # URL mode (synthetic demo data, or --csv url,label)
    python -m phishing_ml.train --mode url --classifier random_forest

    # Email mode, using the Kaggle phishing-email-dataset
    python -m phishing_ml.train --mode email --classifier random_forest \
        --email-dir ~/.cache/kagglehub/datasets/naserabdullahalam/phishing-email-dataset/versions/1
"""
import argparse
import os

import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from .data import (
    find_kagglehub_email_dataset,
    generate_synthetic_dataset,
    load_csv,
    load_email_dataset,
)
from .pipeline import (
    build_email_feature_frame,
    build_email_model_pipeline,
    build_feature_frame,
    build_model_pipeline,
)


def train_url_mode(args):
    if args.csv:
        df = load_csv(args.csv)
    else:
        print("No --csv given; using synthetic demo dataset.")
        df = generate_synthetic_dataset(n_per_class=args.n_per_class)

    X = build_feature_frame(df["url"])
    y = df["label"]
    return build_model_pipeline(args.classifier), X, y, ["legit", "phishing"]


def train_email_mode(args):
    email_dir = args.email_dir or find_kagglehub_email_dataset()
    print(f"Loading email dataset from {email_dir}")
    df = load_email_dataset(email_dir, sample_size=args.sample_size)
    print(f"Loaded {len(df)} emails, label counts: {df['label'].value_counts().to_dict()}")

    X = build_email_feature_frame(df)
    y = df["label"]
    return build_email_model_pipeline(args.classifier), X, y, ["legit", "phishing"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="url", choices=["url", "email"])
    parser.add_argument("--csv", help="[url mode] Path to a labeled CSV (columns: url,label)")
    parser.add_argument("--classifier", default="logistic_regression",
                         choices=["logistic_regression", "random_forest"])
    parser.add_argument("--n-per-class", type=int, default=800,
                         help="[url mode] Rows per class for the synthetic dataset")
    parser.add_argument("--email-dir", help="[email mode] Path to the downloaded Kaggle dataset "
                                             "directory (defaults to the kagglehub cache)")
    parser.add_argument("--sample-size", type=int, default=20000,
                         help="[email mode] Cap on total rows used (balanced across classes)")
    parser.add_argument("--out", default="models/phishing_model.joblib")
    args = parser.parse_args()

    if args.mode == "url":
        pipeline, X, y, target_names = train_url_mode(args)
    else:
        pipeline, X, y, target_names = train_email_mode(args)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print(f"\nMode: {args.mode} | Classifier: {args.classifier}")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(pipeline, args.out)
    print(f"\nSaved model to {args.out}")


if __name__ == "__main__":
    main()
