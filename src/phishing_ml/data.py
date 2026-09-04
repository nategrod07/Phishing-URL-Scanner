"""Dataset loading.

The email corpus is ingested in chunks: each chunk of raw records is turned
into features immediately and the bulky text is released before the next
chunk is read. Peak memory therefore tracks the chunk size, not the corpus
size, so the loader behaves the same on the bundled ~82k emails as it would
on several million.

Nothing is silently discarded. Rows are dropped only for reasons that make
them unusable (no label), duplicates are dropped only when asked, and every
loader returns an `IngestReport` accounting for the difference between rows
read and rows kept.
"""
import glob
import os
import random
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd

from .email_features import extract_email_features

# Files from kaggle.com/datasets/naserabdullahalam/phishing-email-dataset that
# carry structured columns. The flattened `phishing_email.csv` is excluded: it
# has only `text_combined`, so it cannot supply sender_domain and duplicates
# content already present in the files below.
EMAIL_DATASET_FILES = (
    "CEAS_08.csv",
    "Enron.csv",
    "Ling.csv",
    "Nazario.csv",
    "Nigerian_Fraud.csv",
    "SpamAssasin.csv",
)

# Columns worth reading; `receiver`/`date` are skipped to keep the read narrow.
EMAIL_SOURCE_COLUMNS = ("sender", "subject", "body", "urls", "label")

# Used to identify duplicate emails across corpora.
_DEDUPE_KEY = ["sender", "subject", "body"]

DEFAULT_CHUNKSIZE = 50_000


@dataclass
class IngestReport:
    """Accounting for every row between the CSVs and the feature matrix."""

    rows_read: int = 0
    dropped_missing_label: int = 0
    dropped_duplicates: int = 0
    dropped_sampling: int = 0
    rows_per_file: Dict[str, int] = field(default_factory=dict)

    @property
    def rows_kept(self) -> int:
        return (
            self.rows_read
            - self.dropped_missing_label
            - self.dropped_duplicates
            - self.dropped_sampling
        )

    @property
    def retention(self) -> float:
        return self.rows_kept / self.rows_read if self.rows_read else 0.0

    def summary(self) -> str:
        return (
            f"read {self.rows_read:,} rows -> kept {self.rows_kept:,} "
            f"({self.retention:.1%}); dropped: "
            f"{self.dropped_missing_label:,} unlabelled, "
            f"{self.dropped_duplicates:,} duplicate, "
            f"{self.dropped_sampling:,} sampled out"
        )


# --------------------------------------------------------------------------
# URL data
# --------------------------------------------------------------------------

LEGIT_DOMAINS = [
    "github.com", "wikipedia.org", "google.com", "amazon.com",
    "nytimes.com", "stackoverflow.com", "python.org", "mozilla.org",
    "cloudflare.com", "microsoft.com",
]
LEGIT_PATHS = ["", "/docs", "/about", "/search?q=test", "/en/wiki/Main_Page", "/pricing"]

PHISHING_TLDS = ["xyz", "top", "ru", "info", "click", "tk"]
PHISHING_WORDS = ["login", "secure", "verify", "update", "account", "signin", "confirm"]


def _random_legit_url(rng: random.Random) -> str:
    return f"https://{rng.choice(LEGIT_DOMAINS)}{rng.choice(LEGIT_PATHS)}"


def _random_phishing_url(rng: random.Random) -> str:
    word = rng.choice(PHISHING_WORDS)
    brand = rng.choice(["paypal", "amazon", "bankofamerica", "appleid", "netflix"])
    suffix = "".join(rng.choices("abcdefghij0123456789", k=6))
    if rng.random() < 0.15:
        host = ".".join(str(rng.randint(0, 255)) for _ in range(4))
    else:
        host = f"{word}-{brand}-{suffix}.{rng.choice(PHISHING_TLDS)}"
    scheme = "http" if rng.random() < 0.7 else "https"
    return f"{scheme}://{host}/{word}.php?id={rng.randint(1000, 9999)}"


def generate_synthetic_dataset(n_per_class: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate a balanced synthetic URL dataset for demo/dev purposes.

    The two classes are separable almost by construction, so scores on this
    data say only that the pipeline runs -- not that the model is good. Use a
    real labelled corpus for anything else.
    """
    rng = random.Random(seed)
    rows = [{"url": _random_legit_url(rng), "label": 0} for _ in range(n_per_class)]
    rows += [{"url": _random_phishing_url(rng), "label": 1} for _ in range(n_per_class)]
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


def load_url_csv(path: str, url_col: str = "url", label_col: str = "label") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(columns={url_col: "url", label_col: "label"})[["url", "label"]]


# --------------------------------------------------------------------------
# Email data
# --------------------------------------------------------------------------


def find_kagglehub_email_dataset() -> str:
    """Locate the newest cached copy of the phishing-email dataset."""
    pattern = os.path.expanduser(
        "~/.cache/kagglehub/datasets/naserabdullahalam/phishing-email-dataset/versions/*"
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            "No cached kagglehub download found. Run:\n"
            "  import kagglehub\n"
            '  kagglehub.dataset_download("naserabdullahalam/phishing-email-dataset")'
        )
    return matches[-1]


def _existing_dataset_files(dataset_dir: str) -> List[str]:
    files = [f for f in EMAIL_DATASET_FILES if os.path.exists(os.path.join(dataset_dir, f))]
    if not files:
        raise FileNotFoundError(
            f"No known dataset CSVs under {dataset_dir}. Expected any of: "
            f"{', '.join(EMAIL_DATASET_FILES)}"
        )
    return files


def iter_email_chunks(
    dataset_dir: str, chunksize: int = DEFAULT_CHUNKSIZE
) -> Iterator[Tuple[str, pd.DataFrame]]:
    """Yield `(filename, chunk)` pairs of raw email records.

    Only the columns in EMAIL_SOURCE_COLUMNS are read, and files missing some
    of them (Enron and Ling have no sender/urls) still yield their rows --
    the extractor treats absent columns as empty rather than dropping ~32k
    otherwise-usable emails.
    """
    for filename in _existing_dataset_files(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        available = set(pd.read_csv(path, nrows=0).columns)
        usecols = [c for c in EMAIL_SOURCE_COLUMNS if c in available]
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, dtype=object):
            yield filename, chunk


def load_email_features(
    dataset_dir: str,
    chunksize: int = DEFAULT_CHUNKSIZE,
    drop_duplicates: bool = True,
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, IngestReport]:
    """Stream the email corpus into a feature matrix.

    Returns `(features, labels, report)`. `sample_size` caps the total rows,
    balanced across classes; leaving it as None -- the default -- uses every
    labelled row.
    """
    report = IngestReport()
    seen_hashes: set = set()
    feature_frames: List[pd.DataFrame] = []
    label_frames: List[pd.Series] = []

    for filename, chunk in iter_email_chunks(dataset_dir, chunksize):
        report.rows_read += len(chunk)
        report.rows_per_file[filename] = report.rows_per_file.get(filename, 0) + len(chunk)

        labels = pd.to_numeric(chunk.get("label"), errors="coerce")
        labelled = labels.notna()
        report.dropped_missing_label += int((~labelled).sum())
        chunk = chunk[labelled]
        labels = labels[labelled].astype(int)

        if drop_duplicates and len(chunk):
            key = pd.util.hash_pandas_object(
                pd.DataFrame({c: _column_or_blank(chunk, c) for c in _DEDUPE_KEY}),
                index=False,
            )
            fresh = ~key.isin(seen_hashes) & ~key.duplicated()
            report.dropped_duplicates += int((~fresh).sum())
            seen_hashes.update(key[fresh].tolist())
            chunk, labels = chunk[fresh], labels[fresh]

        if not len(chunk):
            continue

        # Features are computed here so the chunk's text can be freed now.
        feature_frames.append(extract_email_features(chunk))
        label_frames.append(labels)

    if not feature_frames:
        raise ValueError(f"No labelled email rows found under {dataset_dir}")

    features = pd.concat(feature_frames, ignore_index=True)
    labels = pd.concat(label_frames, ignore_index=True)

    if sample_size and sample_size < len(features):
        keep = _balanced_sample_index(labels, sample_size, seed)
        report.dropped_sampling = len(features) - len(keep)
        features = features.loc[keep].reset_index(drop=True)
        labels = labels.loc[keep].reset_index(drop=True)

    return features, labels, report


def _column_or_blank(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series("", index=frame.index, dtype="object")
    return frame[name].fillna("").astype(str)


def _balanced_sample_index(labels: pd.Series, sample_size: int, seed: int) -> pd.Index:
    """Pick up to `sample_size` rows split evenly across the label classes."""
    classes = labels.unique()
    per_class = max(sample_size // len(classes), 1)
    picks = [
        labels[labels == cls].sample(
            n=min(per_class, int((labels == cls).sum())), random_state=seed
        ).index
        for cls in classes
    ]
    return pd.Index([i for idx in picks for i in idx]).sort_values()
