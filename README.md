# Phishing Detection (URL + Email)

ML pipelines that classify URLs and emails as phishing or legitimate using
engineered lexical/structural features — one-hot encoding for categorical
fields and feature scaling for numeric fields feeding a classifier.

## Approach

Both modes follow the same pattern: extract a mix of numeric and
categorical features, run them through a `ColumnTransformer`
(`StandardScaler` on numerics, `OneHotEncoder` on categoricals), and feed
a classifier (logistic regression or random forest).

**URL mode** ([src/phishing_ml/features.py](src/phishing_ml/features.py)):
numeric features (URL length, dot/hyphen/digit counts, subdomain count,
suspicious keyword count, etc.) and categorical features (`protocol`,
`tld`) pulled from the URL string alone — no network calls. Dataset
([src/phishing_ml/data.py](src/phishing_ml/data.py)) defaults to a synthetic generator; swap in a
real one (columns `url`, `label`) via `--csv`.

**Email mode** ([src/phishing_ml/email_features.py](src/phishing_ml/email_features.py)): numeric
features (subject/body length, word/digit/URL counts, uppercase ratio,
suspicious-keyword count, etc.) and a categorical `sender_domain` feature
(bucketed into common free-mail domains vs. `other`/`unknown`). Trained on
the Kaggle [phishing-email-dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)
(CEAS_08, Enron, Ling, Nazario, Nigerian_Fraud, SpamAssasin — ~82k emails
combined).

**Known limitation (email mode):** the training corpora mostly contain
legitimate mail from a handful of corporate domains (e.g. Enron), so
`sender_domain="other"` is heavily correlated with phishing in this
dataset. A legitimate email from an uncommon domain (e.g. a real company's
own domain) can get a false-positive phishing score. Worth addressing with
a larger/more diverse legitimate-mail sample or dropping `sender_domain`
in favor of domain-age/reputation features.

## Setup

```bash
cd phishing-url-ml
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Train

**URL mode** (synthetic demo data by default, or `--csv data/raw/urls.csv` with `url,label` columns):

```bash
python -m phishing_ml.train --mode url --classifier random_forest
```

**Email mode**, using the Kaggle dataset:

```python
import kagglehub
path = kagglehub.dataset_download("naserabdullahalam/phishing-email-dataset")
```

```bash
python -m phishing_ml.train --mode email --classifier random_forest --out models/email_model.joblib
```

`--email-dir` can point at the dataset explicitly; otherwise it looks in
the local kagglehub cache. `--sample-size` (default 20000) caps how many
rows get used, balanced across classes, since the combined dataset is
~82k rows.

## Predict

```bash
python -m phishing_ml.predict --model models/phishing_model.joblib \
  "http://paypal-secure-login.xyz/verify" "https://github.com/login"

python -m phishing_ml.predict --mode email --model models/email_model.joblib \
  --sender "support@paypal-secure.xyz" --subject "Verify your account now" \
  --body "Click here to verify your account or it will be suspended."
```

## Test

```bash
pytest
```

## Next steps

- Swap the synthetic dataset for a real one (e.g. PhishTank, Kaggle
  "Phishing Site URLs", or an internal feed of flagged domains).
- Add WHOIS/domain-age and DNS-based features (requires network calls at
  inference time — cache aggressively).
- Try gradient boosting (XGBoost/LightGBM) and compare against the
  logistic regression / random forest baselines.
- Add cross-validation and hyperparameter search (`GridSearchCV`) in
  `train.py`.
