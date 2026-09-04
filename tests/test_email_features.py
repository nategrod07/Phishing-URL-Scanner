import numpy as np
import pandas as pd
import pytest

from phishing_ml.email_features import (
    EMAIL_FEATURES,
    extract_email_features,
    extract_email_features_one,
)


def frame(**columns) -> pd.DataFrame:
    """Build a one-row email frame from the given columns."""
    return pd.DataFrame({key: [value] for key, value in columns.items()})


class TestUrlFeatures:
    def test_num_urls_counted_from_body(self):
        feats = extract_email_features_one(body="click http://a.com or https://b.com now")
        assert feats["num_urls"] == 2
        assert feats["has_urls"] == 1

    def test_urls_column_is_a_flag_not_a_count(self):
        # The dataset ships `urls` as 0/1. It must set has_urls without
        # overwriting the real link count parsed from the body.
        features = extract_email_features(
            frame(body="see http://a.com and http://b.com", urls=1)
        )
        assert features.iloc[0]["has_urls"] == 1
        assert features.iloc[0]["num_urls"] == 2

    def test_urls_flag_zero_still_counts_body_links(self):
        features = extract_email_features(frame(body="http://a.com", urls=0))
        assert features.iloc[0]["has_urls"] == 0
        assert features.iloc[0]["num_urls"] == 1

    def test_missing_urls_column_falls_back_to_body(self):
        features = extract_email_features(frame(body="http://a.com"))
        assert features.iloc[0]["has_urls"] == 1

    def test_unparseable_urls_flag_falls_back_to_body(self):
        features = extract_email_features(frame(body="http://a.com", urls="not-a-number"))
        assert features.iloc[0]["has_urls"] == 1


class TestSenderDomain:
    def test_raw_domain_is_preserved(self):
        # Rare domains must survive extraction; the encoder groups them
        # later using frequencies learned from the training split.
        assert extract_email_features_one(sender="bob@some-corp.io")["sender_domain"] == "some-corp.io"
        assert extract_email_features_one(sender="a@gmail.com")["sender_domain"] == "gmail.com"

    def test_display_name_form(self):
        feats = extract_email_features_one(sender="Bob <bob@Example.COM>")
        assert feats["sender_domain"] == "example.com"

    def test_missing_sender_is_unknown(self):
        assert extract_email_features_one(sender="")["sender_domain"] == "unknown"
        assert extract_email_features(frame(subject="hi")).iloc[0]["sender_domain"] == "unknown"

    def test_malformed_sender_is_unknown(self):
        assert extract_email_features_one(sender="no-at-sign")["sender_domain"] == "unknown"


class TestTextFeatures:
    def test_suspicious_word_count_includes_phrases(self):
        feats = extract_email_features_one(
            subject="URGENT: verify your account",
            body="click here to confirm your password",
        )
        # urgent, verify, account, click here, confirm, password
        assert feats["num_suspicious_words"] == 6

    def test_uppercase_ratio(self):
        feats = extract_email_features_one(subject="FREE MONEY", body="now")
        assert feats["num_uppercase_words"] == 2
        assert feats["uppercase_ratio"] == pytest.approx(2 / 3)

    def test_empty_text_does_not_divide_by_zero(self):
        feats = extract_email_features_one()
        assert feats["uppercase_ratio"] == 0.0
        assert feats["num_words"] == 0

    def test_reply_prefixes(self):
        assert extract_email_features_one(subject="Re: lunch")["has_reply_subject"] == 1
        assert extract_email_features_one(subject="  FWD: report")["has_reply_subject"] == 1
        assert extract_email_features_one(subject="lunch")["has_reply_subject"] == 0

    def test_lengths_are_per_field(self):
        feats = extract_email_features_one(subject="abc", body="de")
        assert feats["subject_length"] == 3
        assert feats["body_length"] == 2


class TestRobustness:
    def test_nan_fields_are_treated_as_empty_not_dropped(self):
        records = pd.DataFrame(
            {"sender": [np.nan], "subject": [np.nan], "body": [np.nan], "urls": [np.nan]}
        )
        features = extract_email_features(records)
        assert len(features) == 1
        assert features.iloc[0]["body_length"] == 0
        assert features.iloc[0]["sender_domain"] == "unknown"

    def test_missing_sender_and_urls_columns_still_extract(self):
        # Enron/Ling ship with only subject/body; those ~32k rows must survive.
        features = extract_email_features(pd.DataFrame({"subject": ["hi"], "body": ["there"]}))
        assert len(features) == 1

    def test_non_string_body_is_coerced(self):
        features = extract_email_features(frame(sender="a@b.com", subject=1234, body=5678))
        assert features.iloc[0]["subject_length"] == 4

    def test_preserves_input_index(self):
        records = pd.DataFrame({"body": ["a", "b"]}, index=[7, 9])
        assert list(extract_email_features(records).index) == [7, 9]


class TestFeatureSetContract:
    def test_build_returns_declared_columns_in_order(self):
        features = EMAIL_FEATURES.build(frame(sender="a@b.com", subject="s", body="b"))
        assert list(features.columns) == EMAIL_FEATURES.columns

    def test_no_nulls_in_output(self):
        records = pd.DataFrame(
            {
                "sender": ["a@b.com", None, ""],
                "subject": ["s", None, "!!!"],
                "body": ["b", None, "http://x.com"],
                "urls": [1, None, "x"],
            }
        )
        assert not EMAIL_FEATURES.build(records).isna().any().any()

    def test_batch_matches_single(self):
        emails = [
            {"sender": "a@b.com", "subject": "Re: hi", "body": "http://x.com verify"},
            {"sender": "", "subject": "", "body": ""},
        ]
        batch = extract_email_features(pd.DataFrame(emails))
        for i, email in enumerate(emails):
            assert batch.iloc[i].to_dict() == extract_email_features_one(**email)
