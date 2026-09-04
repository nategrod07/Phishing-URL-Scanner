"""Feature extraction for phishing emails (sender/subject/body/urls).

Mirrors features.py's split: numeric features get StandardScaler'd,
categorical features get OneHotEncoder'd.
"""
import re

URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
EMAIL_DOMAIN_PATTERN = re.compile(r"@([\w.-]+)")

SUSPICIOUS_WORDS = (
    "urgent", "verify", "suspend", "password", "click here", "act now",
    "confirm", "bank", "account", "limited time", "winner", "congratulations",
    "wire transfer", "social security", "login",
)

# Free/very common providers collapse the domain cardinality; anything else
# is bucketed as "other" or "unknown" so OneHotEncoder(handle_unknown="ignore")
# doesn't choke on thousands of rare corporate domains.
COMMON_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "enron.com", "msn.com", "excite.com",
}

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

CATEGORICAL_FEATURES = [
    "sender_domain",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _sender_domain(sender) -> str:
    if not isinstance(sender, str) or not sender:
        return "unknown"
    match = EMAIL_DOMAIN_PATTERN.search(sender.lower())
    if not match:
        return "unknown"
    domain = match.group(1).strip(">").strip()
    return domain if domain in COMMON_DOMAINS else "other"


def extract_email_features(sender, subject, body, urls=None) -> dict:
    subject = subject if isinstance(subject, str) else ""
    body = body if isinstance(body, str) else ""
    text = f"{subject} {body}"

    words = text.split()
    uppercase_words = [w for w in words if w.isupper() and len(w) > 1]

    if urls is not None and not (isinstance(urls, float)):
        try:
            num_urls = int(urls)
        except (TypeError, ValueError):
            num_urls = len(URL_PATTERN.findall(body))
    else:
        num_urls = len(URL_PATTERN.findall(body))

    return {
        "subject_length": len(subject),
        "body_length": len(body),
        "num_words": len(words),
        "num_urls": num_urls,
        "num_digits": sum(c.isdigit() for c in text),
        "num_exclamations": text.count("!"),
        "num_uppercase_words": len(uppercase_words),
        "uppercase_ratio": len(uppercase_words) / len(words) if words else 0.0,
        "num_suspicious_words": sum(w in text.lower() for w in SUSPICIOUS_WORDS),
        "has_urls": int(num_urls > 0),
        "has_reply_subject": int(subject.lower().strip().startswith(("re:", "fwd:"))),
        "sender_domain": _sender_domain(sender),
    }


def extract_email_features_batch(df) -> list:
    has_sender = "sender" in df.columns
    has_urls = "urls" in df.columns
    rows = []
    for _, row in df.iterrows():
        rows.append(
            extract_email_features(
                sender=row["sender"] if has_sender else None,
                subject=row.get("subject"),
                body=row.get("body"),
                urls=row["urls"] if has_urls else None,
            )
        )
    return rows
