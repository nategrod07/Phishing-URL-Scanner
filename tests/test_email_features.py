from phishing_ml.email_features import extract_email_features


def test_url_count_from_body():
    feats = extract_email_features(
        sender="a@example.com", subject="hi",
        body="click http://a.com or http://b.com now",
    )
    assert feats["num_urls"] == 2
    assert feats["has_urls"] == 1


def test_url_count_prefers_urls_arg():
    feats = extract_email_features(sender="a@example.com", subject="hi", body="no links here", urls=3)
    assert feats["num_urls"] == 3


def test_sender_domain_bucketing():
    assert extract_email_features("bob@gmail.com", "hi", "body")["sender_domain"] == "gmail.com"
    assert extract_email_features("bob@some-corp.io", "hi", "body")["sender_domain"] == "other"
    assert extract_email_features(None, "hi", "body")["sender_domain"] == "unknown"


def test_suspicious_word_count():
    feats = extract_email_features(
        "a@x.com", "URGENT: verify your account",
        "click here to confirm your password now",
    )
    assert feats["num_suspicious_words"] >= 3
