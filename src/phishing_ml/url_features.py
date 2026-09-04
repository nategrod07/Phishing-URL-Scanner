"""Vectorised lexical/host feature extraction for URLs.

Every feature is computed with pandas string ops over whole columns rather
than per-row Python, so throughput stays flat as the corpus grows.

`extract_url_features` is the single implementation. `extract_url_features_one`
wraps it for scoring one URL, so the batch and single-item paths cannot
drift apart.

URLs are parsed with regex rather than `urllib.parse` to stay vectorised.
Unlike a previous per-row version, a URL with no scheme reports
`protocol="none"` instead of being silently rewritten to `http`: the absence
of a scheme is itself signal, and inventing one discards it.
"""
import re
from typing import Iterable, Union

import pandas as pd

from .featureset import FeatureSet

SUSPICIOUS_WORDS = (
    "login", "verify", "update", "secure", "account", "banking",
    "confirm", "signin", "webscr", "password",
)

# Alternation over the whole vocabulary, counted in one pass per column.
_SUSPICIOUS_RE = "|".join(re.escape(word) for word in SUSPICIOUS_WORDS)

# scheme://  |  authority = everything up to the first / ? #  |  path  |  query
_SCHEME_RE = r"^([a-zA-Z][a-zA-Z0-9+.\-]*)://"
_AUTHORITY_RE = r"^(?:[a-zA-Z][a-zA-Z0-9+.\-]*://)?([^/?#]*)"
_PATH_RE = r"^(?:[a-zA-Z][a-zA-Z0-9+.\-]*://)?[^/?#]*([^?#]*)"
_QUERY_RE = r"\?([^#]*)"
_IPV4_RE = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"

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

CATEGORICAL_FEATURES = ["protocol", "tld"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _as_url_series(urls: Union[pd.DataFrame, pd.Series, Iterable]) -> pd.Series:
    """Accept a DataFrame with a `url` column, a Series, or any iterable."""
    if isinstance(urls, pd.DataFrame):
        urls = urls["url"]
    if not isinstance(urls, pd.Series):
        urls = pd.Series(list(urls), dtype="object")
    return urls.fillna("").astype(str).str.strip()


def _hostname(authority: pd.Series) -> pd.Series:
    """Strip `userinfo@` and `:port` from an authority component."""
    return authority.str.rsplit("@", n=1).str[-1].str.split(":").str[0].str.lower()


def _tld(hostname: pd.Series) -> pd.Series:
    """Last dotted label, or "none" for hostnames without a dot."""
    return hostname.str.rsplit(".", n=1).str[-1].where(hostname.str.contains(".", regex=False), "none")


def extract_url_features(urls: Union[pd.DataFrame, pd.Series, Iterable]) -> pd.DataFrame:
    """Build the URL feature frame. Input may be a frame, Series or iterable."""
    url = _as_url_series(urls)

    scheme = url.str.extract(_SCHEME_RE, expand=False).fillna("none").str.lower()
    authority = url.str.extract(_AUTHORITY_RE, expand=False).fillna("")
    hostname = _hostname(authority)
    path = url.str.extract(_PATH_RE, expand=False).fillna("")
    query = url.str.extract(_QUERY_RE, expand=False).fillna("")

    length = url.str.len()
    num_digits = url.str.count(r"\d")

    features = pd.DataFrame(index=url.index)
    features["url_length"] = length
    features["hostname_length"] = hostname.str.len()
    features["path_length"] = path.str.len()
    features["num_dots"] = url.str.count(r"\.")
    features["num_hyphens"] = url.str.count("-")
    features["num_underscores"] = url.str.count("_")
    features["num_slashes"] = url.str.count("/")
    features["num_digits"] = num_digits
    features["num_special_chars"] = url.str.count(r"[@%=&?~#]")
    features["num_subdomains"] = (hostname.str.count(r"\.") - 1).clip(lower=0)
    features["num_query_params"] = query.str.count("=")
    # `where` masks the zero-length rows to NaN so the division never warns.
    features["digit_ratio"] = (num_digits / length.where(length > 0)).fillna(0.0)
    features["has_ip_address"] = hostname.str.match(_IPV4_RE).fillna(False).astype(int)
    features["has_at_symbol"] = url.str.contains("@", regex=False).astype(int)
    features["has_https"] = (scheme == "https").astype(int)
    features["num_suspicious_words"] = url.str.lower().str.count(_SUSPICIOUS_RE)
    features["protocol"] = scheme
    features["tld"] = _tld(hostname)
    return features


def extract_url_features_one(url: str) -> dict:
    """Score-time convenience wrapper: one URL in, one feature dict out."""
    return extract_url_features([url]).iloc[0].to_dict()


URL_FEATURES = FeatureSet(
    name="url",
    numeric=NUMERIC_FEATURES,
    categorical=CATEGORICAL_FEATURES,
    extract=extract_url_features,
)
