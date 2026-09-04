"""Score URLs or emails with a trained model.

Usage:
    python -m phishing_ml.predict --model models/phishing_model.joblib \
        "http://paypal-secure-login.xyz/verify"

    python -m phishing_ml.predict --mode email --model models/email_model.joblib \
        --sender "support@paypal-secure.xyz" --subject "Verify your account" \
        --body "Click here to confirm your password."
"""
import argparse
from typing import List, Sequence

import joblib
import pandas as pd

from .email_features import EMAIL_FEATURES
from .url_features import URL_FEATURES


def _score(pipeline, features: pd.DataFrame) -> List[dict]:
    proba = pipeline.predict_proba(features)[:, 1]
    predictions = pipeline.predict(features)
    return [
        {"prediction": "phishing" if p else "legit", "phishing_probability": float(prob)}
        for p, prob in zip(predictions, proba)
    ]


def predict_urls(model_path: str, urls: Sequence[str]) -> List[dict]:
    pipeline = joblib.load(model_path)
    features = URL_FEATURES.build(pd.DataFrame({"url": list(urls)}))
    return [{"url": url, **result} for url, result in zip(urls, _score(pipeline, features))]


def predict_emails(model_path: str, emails: Sequence[dict]) -> List[dict]:
    """Score a batch of `{sender, subject, body}` dicts in one pass."""
    pipeline = joblib.load(model_path)
    records = pd.DataFrame(list(emails), columns=["sender", "subject", "body"]).fillna("")
    features = EMAIL_FEATURES.build(records)
    return [{**email, **result} for email, result in zip(emails, _score(pipeline, features))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urls", nargs="*", help="[url mode] URLs to score")
    parser.add_argument("--mode", default="url", choices=["url", "email"])
    parser.add_argument("--model", default="models/phishing_model.joblib")
    parser.add_argument("--sender", default="", help="[email mode] Sender address")
    parser.add_argument("--subject", default="", help="[email mode] Subject line")
    parser.add_argument("--body", default="", help="[email mode] Body text")
    args = parser.parse_args()

    if args.mode == "url":
        if not args.urls:
            parser.error("url mode needs at least one URL")
        for result in predict_urls(args.model, args.urls):
            print(f"{result['prediction']:>8}  (p={result['phishing_probability']:.3f})  "
                  f"{result['url']}")
    else:
        email = {"sender": args.sender, "subject": args.subject, "body": args.body}
        result = predict_emails(args.model, [email])[0]
        print(f"{result['prediction']:>8}  (p={result['phishing_probability']:.3f})  "
              f"from={result['sender']!r} subject={result['subject']!r}")


if __name__ == "__main__":
    main()
