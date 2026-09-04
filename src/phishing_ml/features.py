"""Lexical/host-based feature extraction for URLs.

Produces a mix of numeric features (lengths, character counts) that get
StandardScaler'd and categorical features (protocol, TLD) that get
OneHotEncoder'd downstream in pipeline.py.
"""
import re
from urllib.parse import urlparse

SUSPICIOUS_WORDS = (
    "login", "verify", "update", "secure", "account", "banking",
    "confirm", "signin", "webscr", "password",
)

IP_PATTERN = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")

NUMERIC_FEATURES = [
    "url_length",
    "hostname_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_digits",
    "num_special_chars",
    "num_subdomains",
    "num_query_params",
    "digit_ratio",
    "has_ip_address",
    "has_at_symbol",
    "has_https",
    "num_suspicious_words",
]

CATEGORICAL_FEATURES = [
    "protocol",
    "tld",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _tld(hostname: str) -> str:
    parts = hostname.split(".")
    if len(parts) < 2:
        return "none"
    return parts[-1].lower()


def extract_features(url: str) -> dict:
    """Extract a flat dict of features for a single URL string."""
    url = url.strip()
    parsed = urlparse(url if "://" in url else f"http://{url}")
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    digits = sum(c.isdigit() for c in url)

    return {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_digits": digits,
        "num_special_chars": sum(url.count(c) for c in "@%=&?~#"),
        "num_subdomains": max(hostname.count(".") - 1, 0),
        "num_query_params": query.count("=") if query else 0,
        "digit_ratio": digits / len(url) if url else 0.0,
        "has_ip_address": int(bool(IP_PATTERN.match(hostname))),
        "has_at_symbol": int("@" in url),
        "has_https": int(parsed.scheme == "https"),
        "num_suspicious_words": sum(w in url.lower() for w in SUSPICIOUS_WORDS),
        "protocol": parsed.scheme or "none",
        "tld": _tld(hostname),
    }


def extract_features_batch(urls) -> list:
    return [extract_features(u) for u in urls]
