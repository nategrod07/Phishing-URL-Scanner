"""Dataset loading utilities.

For a real project, point `load_csv` at a labeled URL dataset (columns:
`url`, `label` with label in {0, 1}, 1 = phishing). A small synthetic
generator is included so the pipeline runs end-to-end with no external
data.
"""
import glob
import os
import random
from typing import Optional

import pandas as pd

# Files from kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
# that carry structured columns (sender/subject/body/urls) rather than the
# single pre-flattened `text_combined` column in phishing_email.csv.
EMAIL_DATASET_FILES = [
    "CEAS_08.csv",
    "Enron.csv",
    "Ling.csv",
    "Nazario.csv",
    "Nigerian_Fraud.csv",
    "SpamAssasin.csv",
]
EMAIL_COLUMNS = ["sender", "subject", "body", "urls", "label"]

LEGIT_DOMAINS = [
    "github.com", "wikipedia.org", "google.com", "amazon.com",
    "nytimes.com", "stackoverflow.com", "python.org", "mozilla.org",
    "cloudflare.com", "microsoft.com",
]
LEGIT_PATHS = ["", "/docs", "/about", "/search?q=test", "/en/wiki/Main_Page", "/pricing"]

PHISHING_TLDS = ["xyz", "top", "ru", "info", "click", "tk"]
PHISHING_WORDS = ["login", "secure", "verify", "update", "account", "signin", "confirm"]


def _random_legit_url() -> str:
    domain = random.choice(LEGIT_DOMAINS)
    path = random.choice(LEGIT_PATHS)
    return f"https://{domain}{path}"


def _random_phishing_url() -> str:
    word = random.choice(PHISHING_WORDS)
    brand = random.choice(["paypal", "amazon", "bankofamerica", "appleid", "netflix"])
    tld = random.choice(PHISHING_TLDS)
    suffix = "".join(random.choices("abcdefghij0123456789", k=6))
    use_ip = random.random() < 0.15
    if use_ip:
        host = f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}"
    else:
        host = f"{word}-{brand}-{suffix}.{tld}"
    scheme = "http" if random.random() < 0.7 else "https"
    return f"{scheme}://{host}/{word}.php?id={random.randint(1000,9999)}"


def generate_synthetic_dataset(n_per_class: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate a balanced synthetic dataset for demo/dev purposes.

    Replace with `load_csv` pointed at a real labeled dataset for anything
    beyond pipeline development.
    """
    rng = random.Random(seed)
    random.seed(seed)

    rows = []
    for _ in range(n_per_class):
        rows.append({"url": _random_legit_url(), "label": 0})
    for _ in range(n_per_class):
        rows.append({"url": _random_phishing_url(), "label": 1})

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_csv(path: str, url_col: str = "url", label_col: str = "label") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(columns={url_col: "url", label_col: "label"})[["url", "label"]]


def load_email_dataset(dataset_dir: str, sample_size: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    """Load and combine the structured CSVs from the Kaggle phishing-email
    dataset (kaggle.com/datasets/naserabdullahalam/phishing-email-dataset).

    Only files with sender/subject/body/urls columns are used (Enron/Ling
    lack sender+urls; those columns are filled with NaN for them). The
    flattened `phishing_email.csv` (text_combined + label only) is skipped
    since it doesn't support the categorical (sender_domain) feature.
    """
    frames = []
    for fname in EMAIL_DATASET_FILES:
        fpath = os.path.join(dataset_dir, fname)
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath)
        for col in EMAIL_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df["source"] = fname
        frames.append(df[EMAIL_COLUMNS + ["source"]])

    if not frames:
        raise FileNotFoundError(
            f"No known dataset CSVs found under {dataset_dir}. "
            f"Expected one of: {EMAIL_DATASET_FILES}"
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["label"]).reset_index(drop=True)
    combined["label"] = combined["label"].astype(int)

    if sample_size and sample_size < len(combined):
        combined = combined.groupby("label", group_keys=False)[combined.columns.tolist()].apply(
            lambda g: g.sample(
                n=min(len(g), sample_size // combined["label"].nunique()),
                random_state=seed,
            )
        )
        combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    return combined


def find_kagglehub_email_dataset() -> str:
    """Locate the most recently downloaded copy of the phishing-email
    dataset in the local kagglehub cache."""
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
