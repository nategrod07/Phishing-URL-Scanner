import pandas as pd
import pytest

from phishing_ml.url_features import (
    URL_FEATURES,
    extract_url_features,
    extract_url_features_one,
)


class TestParsing:
    """The regex parser has to match urllib.parse on the awkward cases."""

    def test_scheme_and_tld(self):
        feats = extract_url_features_one("https://example.com/path")
        assert feats["protocol"] == "https"
        assert feats["tld"] == "com"
        assert feats["has_https"] == 1

    def test_missing_scheme_is_not_invented(self):
        # Reporting "none" keeps the absence of a scheme as signal rather
        # than rewriting it to http and losing the distinction.
        feats = extract_url_features_one("example.com/path")
        assert feats["protocol"] == "none"
        assert feats["has_https"] == 0
        assert feats["hostname_length"] == len("example.com")

    def test_userinfo_is_stripped_from_hostname(self):
        # http://user@evil.com/ has hostname evil.com, not user@evil.com.
        feats = extract_url_features_one("http://user@evil.com/login")
        assert feats["hostname_length"] == len("evil.com")
        assert feats["tld"] == "com"
        assert feats["has_at_symbol"] == 1

    def test_port_is_stripped_from_hostname(self):
        feats = extract_url_features_one("http://example.com:8080/a")
        assert feats["hostname_length"] == len("example.com")
        assert feats["tld"] == "com"

    def test_ip_host_detected(self):
        feats = extract_url_features_one("http://192.168.1.10/login")
        assert feats["has_ip_address"] == 1

    def test_ip_lookalike_in_path_is_not_an_ip_host(self):
        feats = extract_url_features_one("http://example.com/192.168.1.10")
        assert feats["has_ip_address"] == 0

    def test_hostname_without_dot_has_no_tld(self):
        assert extract_url_features_one("http://localhost/admin")["tld"] == "none"

    def test_path_and_query_are_separated(self):
        feats = extract_url_features_one("https://example.com/a/b?x=1&y=2")
        assert feats["path_length"] == len("/a/b")
        assert feats["num_query_params"] == 2

    def test_fragment_excluded_from_path(self):
        feats = extract_url_features_one("https://example.com/a#section")
        assert feats["path_length"] == len("/a")

    def test_subdomain_count(self):
        assert extract_url_features_one("http://a.b.example.com")["num_subdomains"] == 2
        assert extract_url_features_one("http://example.com")["num_subdomains"] == 0


class TestCounts:
    def test_digit_ratio(self):
        feats = extract_url_features_one("http://a1.com/22")
        assert feats["digit_ratio"] == pytest.approx(feats["num_digits"] / feats["url_length"])

    def test_suspicious_word_count(self):
        feats = extract_url_features_one("http://secure-login-verify.example.com")
        assert feats["num_suspicious_words"] == 3

    def test_empty_url_does_not_divide_by_zero(self):
        feats = extract_url_features_one("")
        assert feats["digit_ratio"] == 0.0
        assert feats["url_length"] == 0


class TestVectorisation:
    def test_batch_matches_single(self):
        urls = [
            "https://github.com/login",
            "http://user@192.168.0.1:8080/verify.php?id=1#top",
            "example.com",
            "",
        ]
        batch = extract_url_features(urls)
        for i, url in enumerate(urls):
            assert batch.iloc[i].to_dict() == extract_url_features_one(url)

    def test_accepts_dataframe_series_and_list(self):
        urls = ["https://a.com", "http://b.xyz/login"]
        from_list = extract_url_features(urls)
        from_series = extract_url_features(pd.Series(urls))
        from_frame = extract_url_features(pd.DataFrame({"url": urls}))
        pd.testing.assert_frame_equal(from_list, from_series)
        pd.testing.assert_frame_equal(from_list, from_frame)

    def test_nulls_are_treated_as_empty_not_dropped(self):
        features = extract_url_features(pd.Series(["https://a.com", None]))
        assert len(features) == 2
        assert features.iloc[1]["url_length"] == 0


class TestFeatureSetContract:
    def test_build_returns_declared_columns_in_order(self):
        features = URL_FEATURES.build(pd.DataFrame({"url": ["https://a.com"]}))
        assert list(features.columns) == URL_FEATURES.columns

    def test_no_nulls_in_output(self):
        urls = ["https://a.com/x", "", "not a url", "ftp://h/p", None]
        assert not URL_FEATURES.build(pd.DataFrame({"url": urls})).isna().any().any()
