"""Vectorised feature extraction for emails (sender/subject/body/urls).

Mirrors url_features.py: numeric columns get scaled, categorical columns get
one-hot encoded, and everything is computed with column-wide pandas string
ops so cost scales with the corpus rather than with Python loop overhead.

Two deliberate choices preserve signal that an earlier per-row version
discarded:

* The dataset's `urls` column is a **binary flag**, not a count. It is read
  as `has_urls`, and `num_urls` is always counted from the body text, so the
  actual link volume survives.
* `sender_domain` keeps the raw domain. Bucketing it against a hardcoded
  allowlist collapsed ~99% of 24k+ distinct senders into a single "other"
  category that correlated with the phishing label purely as an artifact of
  which corpora supplied the legitimate mail. Rare domains are grouped by
  the encoder instead, learned from the training split (see pipeline.py).
"""
import re

import pandas as pd

from .featureset import FeatureSet

SUSPICIOUS_WORDS = (
    "urgent", "verify", "suspend", "password", "click here", "act now",
    "confirm", "bank", "account", "limited time", "winner", "congratulations",
    "wire transfer", "social security", "login",
)

_SUSPICIOUS_RE = "|".join(re.escape(word) for word in SUSPICIOUS_WORDS)
_URL_RE = r"https?://"
_DOMAIN_RE = r"@([\w.-]+)"
_UPPERCASE_WORD_RE = r"\b[A-Z]{2,}\b"

TEXT_COLUMNS = ("sender", "subject", "body")

NUMERIC_FEATURES = [
    "subject_length",
    "body_length",
    "num_words",
    "num_urls",
    "num_digits",
    "num_exclamations",
    "num_uppercase_words",
    "uppercase_ratio",
    "num_suspicious_words",
    "has_urls",
    "has_reply_subject",
]

CATEGORICAL_FEATURES = ["sender_domain"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _text_column(emails: pd.DataFrame, name: str) -> pd.Series:
    """Return `name` as clean strings, or empty strings if the column is absent.

    Enron and Ling ship without sender/urls columns; treating those as empty
    keeps their ~32k rows usable instead of dropping them.
    """
    if name not in emails.columns:
        return pd.Series("", index=emails.index, dtype="object")
    return emails[name].fillna("").astype(str)


def _sender_domain(sender: pd.Series) -> pd.Series:
    domain = sender.str.lower().str.extract(_DOMAIN_RE, expand=False)
    return domain.str.strip(">").fillna("unknown").replace("", "unknown")


def extract_email_features(emails: pd.DataFrame) -> pd.DataFrame:
    """Build the email feature frame from a DataFrame of raw records."""
    sender = _text_column(emails, "sender")
    subject = _text_column(emails, "subject")
    body = _text_column(emails, "body")
    text = subject.str.cat(body, sep=" ")

    num_words = text.str.split().str.len().fillna(0)
    num_uppercase_words = text.str.count(_UPPERCASE_WORD_RE)
    num_urls = body.str.count(_URL_RE)

    features = pd.DataFrame(index=emails.index)
    features["subject_length"] = subject.str.len()
    features["body_length"] = body.str.len()
    features["num_words"] = num_words
    features["num_urls"] = num_urls
    features["num_digits"] = text.str.count(r"\d")
    features["num_exclamations"] = text.str.count("!")
    features["num_uppercase_words"] = num_uppercase_words
    features["uppercase_ratio"] = (num_uppercase_words / num_words.where(num_words > 0)).fillna(0.0)
    features["num_suspicious_words"] = text.str.lower().str.count(_SUSPICIOUS_RE)

    # Prefer the dataset's own link flag where it exists; fall back to whether
    # the body actually contains a link.
    if "urls" in emails.columns:
        flag = pd.to_numeric(emails["urls"], errors="coerce")
        features["has_urls"] = flag.fillna((num_urls > 0).astype(int)).gt(0).astype(int)
    else:
        features["has_urls"] = (num_urls > 0).astype(int)

    features["has_reply_subject"] = (
        subject.str.strip().str.lower().str.startswith(("re:", "fwd:", "fw:")).astype(int)
    )
    features["sender_domain"] = _sender_domain(sender)
    return features


def extract_email_features_one(sender: str = "", subject: str = "", body: str = "") -> dict:
    """Score-time convenience wrapper: one email in, one feature dict out."""
    record = pd.DataFrame([{"sender": sender, "subject": subject, "body": body}])
    return extract_email_features(record).iloc[0].to_dict()


EMAIL_FEATURES = FeatureSet(
    name="email",
    numeric=NUMERIC_FEATURES,
    categorical=CATEGORICAL_FEATURES,
    extract=extract_email_features,
)
