"""Score a URL or email with a trained model.

Usage:
    python -m phishing_ml.predict --model models/phishing_model.joblib "http://paypal-secure-login.xyz/verify"
    python -m phishing_ml.predict --mode email --model models/email_model.joblib \
        --sender "support@paypal-secure.xyz" --subject "Verify your account now" \
        --body "Click here to verify your account or it will be suspended."
"""
import argparse

import joblib

from .email_features import extract_email_features
from .pipeline import build_feature_frame


def predict_urls(model_path: str, urls: list) -> list:
    pipeline = joblib.load(model_path)
    X = build_feature_frame(urls)
    proba = pipeline.predict_proba(X)[:, 1]
    pred = pipeline.predict(X)
    return [
        {"url": u, "prediction": "phishing" if p else "legit", "phishing_probability": float(prob)}
        for u, p, prob in zip(urls, pred, proba)
    ]


def predict_email(model_path: str, sender: str, subject: str, body: str) -> dict:
    import pandas as pd

    pipeline = joblib.load(model_path)
    feats = extract_email_features(sender=sender, subject=subject, body=body)
    X = pd.DataFrame([feats])
    proba = pipeline.predict_proba(X)[:, 1][0]
    pred = pipeline.predict(X)[0]
    return {
        "sender": sender,
        "subject": subject,
        "prediction": "phishing" if pred else "legit",
        "phishing_probability": float(proba),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urls", nargs="*", help="[url mode] One or more URLs to score")
    parser.add_argument("--mode", default="url", choices=["url", "email"])
    parser.add_argument("--model", default="models/phishing_model.joblib")
    parser.add_argument("--sender", default="", help="[email mode] Sender address")
    parser.add_argument("--subject", default="", help="[email mode] Subject line")
    parser.add_argument("--body", default="", help="[email mode] Email body")
    args = parser.parse_args()

    if args.mode == "url":
        results = predict_urls(args.model, args.urls)
        for r in results:
            print(f"{r['prediction']:>8}  (p={r['phishing_probability']:.3f})  {r['url']}")
    else:
        r = predict_email(args.model, args.sender, args.subject, args.body)
        print(f"{r['prediction']:>8}  (p={r['phishing_probability']:.3f})  "
              f"from={r['sender']!r} subject={r['subject']!r}")


if __name__ == "__main__":
    main()
