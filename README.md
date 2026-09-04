# Phishing Detection (URL + Email)

ML pipelines that classify URLs and emails as phishing or legitimate using
engineered lexical/structural features — one-hot encoding for categorical
fields and feature scaling for numeric fields feeding a classifier.

## Approach

Both modes follow the same pattern: a `FeatureSet` binds the numeric and
categorical column names to the extractor that produces them, and the
pipeline runs `StandardScaler` over the numeric block and `OneHotEncoder`
over the categorical block before the classifier. Because the column lists
travel with the extractor, a new feature cannot silently go unscaled or
unencoded.

Feature extraction is fully vectorised — column-wide pandas string ops, no
per-row Python — so throughput stays flat as the corpus grows. The email
corpus is ingested in chunks: each chunk becomes features immediately and
the bulky text is released before the next is read, so peak memory tracks
the chunk size rather than the corpus size.

**URL mode** ([url_features.py](src/phishing_ml/url_features.py)): numeric
features (length, dot/hyphen/digit counts, subdomain count, suspicious
keyword count) and categorical features (`protocol`, `tld`) parsed from the
URL string alone — no network calls.

**Email mode** ([email_features.py](src/phishing_ml/email_features.py)):
numeric features (subject/body length, word/digit/URL counts, uppercase
ratio, suspicious-keyword count, reply-prefix flag) plus a categorical
`sender_domain`. Trained on the Kaggle
[phishing-email-dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)
(CEAS_08, Enron, Ling, Nazario, Nigerian_Fraud, SpamAssasin).

## Results

Random forest on all 82,486 labelled emails (80/20 split, full corpus, no
sampling):

| metric | value |
|---|---|
| accuracy | 0.917 |
| ROC AUC | 0.9756 |
| PR AUC | 0.9792 |
| log loss | 0.2045 |

Ingesting and featurising the whole corpus takes ~6s; training ~12s.

Classifier / `min_frequency` sweep on the full corpus:

| classifier | min_freq 5 | 20 | 100 |
|---|---|---|---|
| logistic_regression | 0.9004 | 0.8809 | 0.8548 |
| random_forest | **0.9756** | 0.9749 | 0.9735 |
| hist_gradient_boosting | 0.9641 | 0.9634 | 0.9621 |

*(ROC AUC.)* Every classifier improves as `min_frequency` drops — retaining
sender-domain detail is worth real accuracy, which is why the default is 5.

## Preserving signal

Three earlier choices were discarding information and were measured out:

- **`sender_domain` was bucketed against a hardcoded allowlist**, collapsing
  ~99% of 24k+ distinct senders into one `"other"` category that correlated
  with the label mostly as an artifact of which corpora supplied the ham.
  Domains are now kept raw, and rare ones are pooled by the encoder's
  `min_frequency` using counts learned from the *training split only*.
  Permutation importance now ranks `sender_domain` the **most informative
  feature** (+0.075 ROC AUC).
- **The dataset's `urls` column is a binary flag, not a count**, but was
  being read as `num_urls`. It now sets `has_urls`, and `num_urls` is
  counted from the body, so link volume survives.
- **Training defaulted to a 20,000-row cap** (24% of the corpus). Default is
  now the full corpus; `--sample-size` is opt-in. Every loader returns an
  `IngestReport` accounting for the gap between rows read and rows kept.

## Known limitation: out-of-distribution mail

On held-out data from its own distribution the model is reliable (89.9% of
legitimate and 88.4% of phishing emails classified correctly). On mail
unlike anything in the corpora it is not, and it fails **confidently**.

A modern GitHub notification scores `phishing` at p=0.95. Diagnosing it:
`github.com` appears in 0 training rows, so the model's strongest feature is
uninformative; the message has no `Re:` prefix, the second-strongest feature
and a marker of genuine threads; and at 66 characters it is shorter than the
10th percentile of *both* classes. The corpora are largely 2001–2008 mail,
so machine-generated notification digests are simply unrepresented.

This is extrapolation, not a bug — but it means the probabilities should not
be trusted on mail far from the training distribution. Fixing it needs more
diverse legitimate mail, not more feature engineering.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Train

**URL mode** (synthetic demo data by default, or a `url,label` CSV via `--csv`):

```bash
python -m phishing_ml.train --mode url --classifier random_forest
```

The synthetic classes are separable by construction, so a perfect score
there means the pipeline runs — not that the model is good.

**Email mode** — download the dataset once:

```python
import kagglehub
kagglehub.dataset_download("naserabdullahalam/phishing-email-dataset")
```

```bash
python -m phishing_ml.train --mode email --classifier random_forest --out models/email_model.joblib
```

Useful flags: `--email-dir` to point at the dataset explicitly (otherwise
the kagglehub cache is used), `--sample-size` to cap rows (default: use all
of them), `--chunksize` to tune ingest memory, `--cv N` for cross-validated
ROC AUC, and `--keep-duplicates` to skip dedupe (duplicates leak across the
train/test split and inflate scores).

## Predict

```bash
python -m phishing_ml.predict --model models/url_model.joblib \
  "http://paypal-secure-login.xyz/verify" "https://github.com/login"

python -m phishing_ml.predict --mode email --model models/email_model.joblib \
  --sender "support@paypal-secure.xyz" --subject "Verify your account" \
  --body "Click here to confirm your password."
```

`predict_emails()` takes a batch of records and scores them in one
vectorised pass; the CLI is a thin wrapper over it.

## Test

```bash
pytest
```

72 tests covering URL parsing edge cases (userinfo, ports, IP hosts,
missing schemes, fragments), email extraction robustness (NaN fields,
absent columns, flag-vs-count handling), ingest accounting (chunk-boundary
dedupe, chunk-size invariance, balanced sampling, retention reporting), and
pipeline behaviour (rare-category grouping, unseen categories, save/load
roundtrip, per-classifier fit).

On macOS/Python 3.9 with numpy 2.0.2, `LogisticRegression` emits spurious
`matmul` RuntimeWarnings from the Accelerate BLAS backend. It reproduces on
plain random data with no project code involved.

## Next steps

- Source more diverse and more recent legitimate mail to address the
  out-of-distribution weakness above.
- Add domain-age / reputation features so unseen senders degrade gracefully
  rather than falling back on weak lexical cues.
- Calibrate probabilities (`CalibratedClassifierCV`) so the confidence
  reported on unfamiliar mail is closer to honest.
- Compare against a TF-IDF text baseline to quantify what the engineered
  features are worth.
