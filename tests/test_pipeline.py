import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from phishing_ml.data import generate_synthetic_dataset
from phishing_ml.email_features import EMAIL_FEATURES
from phishing_ml.pipeline import CLASSIFIERS, build_classifier, build_pipeline, build_preprocessor
from phishing_ml.url_features import URL_FEATURES


@pytest.fixture
def url_training_data():
    records = generate_synthetic_dataset(n_per_class=60)
    return URL_FEATURES.build(records), records["label"]


class TestPreprocessor:
    def test_scales_numeric_columns(self, url_training_data):
        X, _ = url_training_data
        transformed = build_preprocessor(URL_FEATURES).fit_transform(X)
        n_numeric = len(URL_FEATURES.numeric)
        numeric_block = transformed[:, :n_numeric]
        # Constant columns stay at 0; the rest are standardised.
        assert np.allclose(numeric_block.mean(axis=0), 0, atol=1e-9)

    def test_one_hot_widens_categorical_columns(self, url_training_data):
        X, _ = url_training_data
        transformed = build_preprocessor(URL_FEATURES).fit_transform(X)
        assert transformed.shape[1] > len(URL_FEATURES.columns)

    def test_unseen_category_does_not_raise(self, url_training_data):
        X, _ = url_training_data
        preprocessor = build_preprocessor(URL_FEATURES).fit(X)
        unseen = URL_FEATURES.build(pd.DataFrame({"url": ["https://novel.invalidtld"]}))
        assert preprocessor.transform(unseen).shape[1] == preprocessor.transform(X).shape[1]

    def test_rare_categories_are_grouped_not_dropped(self):
        # 30 common senders and 3 one-off ones. With min_frequency=5 the rare
        # domains share an "infrequent" column instead of vanishing.
        senders = [f"a@common{i % 2}.com" for i in range(30)] + [
            "a@rare1.com", "a@rare2.com", "a@rare3.com"
        ]
        X = EMAIL_FEATURES.build(pd.DataFrame({"sender": senders, "subject": "s", "body": "b"}))
        encoder = build_preprocessor(EMAIL_FEATURES, min_frequency=5).fit(X).named_transformers_["cat"]
        categories = encoder.get_feature_names_out()
        assert any("infrequent" in name for name in categories)
        assert len(categories) < len(set(senders))

    def test_min_frequency_none_keeps_every_category(self):
        senders = [f"a@d{i}.com" for i in range(6)]
        X = EMAIL_FEATURES.build(pd.DataFrame({"sender": senders, "subject": "s", "body": "b"}))
        encoder = build_preprocessor(EMAIL_FEATURES, min_frequency=None).fit(X).named_transformers_["cat"]
        assert len(encoder.get_feature_names_out()) == 6


class TestClassifierFactory:
    @pytest.mark.parametrize("name", CLASSIFIERS)
    def test_every_advertised_classifier_builds(self, name):
        assert build_classifier(name) is not None

    def test_unknown_classifier_raises(self):
        with pytest.raises(ValueError, match="Unknown classifier"):
            build_classifier("svm")


class TestPipeline:
    @pytest.mark.parametrize("name", CLASSIFIERS)
    def test_fits_and_predicts_for_each_classifier(self, url_training_data, name):
        X, y = url_training_data
        pipeline = build_pipeline(URL_FEATURES, name).fit(X, y)
        proba = pipeline.predict_proba(X)[:, 1]
        assert isinstance(pipeline, Pipeline)
        assert set(pipeline.predict(X)) <= {0, 1}
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_learns_the_synthetic_signal(self, url_training_data):
        X, y = url_training_data
        pipeline = build_pipeline(URL_FEATURES, "random_forest").fit(X, y)
        assert pipeline.score(X, y) > 0.9

    def test_scores_a_single_row(self, url_training_data):
        X, y = url_training_data
        pipeline = build_pipeline(URL_FEATURES).fit(X, y)
        one = URL_FEATURES.build(pd.DataFrame({"url": ["http://verify-paypal.xyz/login.php"]}))
        assert pipeline.predict_proba(one).shape == (1, 2)

    def test_survives_a_save_load_roundtrip(self, url_training_data, tmp_path):
        X, y = url_training_data
        pipeline = build_pipeline(URL_FEATURES).fit(X, y)
        path = tmp_path / "model.joblib"
        joblib.dump(pipeline, path)
        np.testing.assert_allclose(
            joblib.load(path).predict_proba(X), pipeline.predict_proba(X)
        )

    def test_email_pipeline_trains_end_to_end(self):
        records = pd.DataFrame(
            {
                "sender": ["a@corp.com"] * 10 + ["x@phish.xyz"] * 10,
                "subject": ["Re: agenda"] * 10 + ["URGENT verify account"] * 10,
                "body": ["notes attached"] * 10 + ["click here http://x.co confirm password"] * 10,
                "urls": [0] * 10 + [1] * 10,
            }
        )
        y = pd.Series([0] * 10 + [1] * 10)
        pipeline = build_pipeline(EMAIL_FEATURES, "random_forest")
        pipeline.fit(EMAIL_FEATURES.build(records), y)
        assert pipeline.score(EMAIL_FEATURES.build(records), y) == 1.0
