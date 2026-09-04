from phishing_ml.features import extract_features


def test_https_flag():
    assert extract_features("https://example.com")["has_https"] == 1
    assert extract_features("http://example.com")["has_https"] == 0


def test_ip_host_detected():
    feats = extract_features("http://192.168.1.10/login")
    assert feats["has_ip_address"] == 1


def test_tld_extracted():
    assert extract_features("https://example.com/path")["tld"] == "com"
    assert extract_features("http://phish.xyz")["tld"] == "xyz"


def test_suspicious_word_count():
    feats = extract_features("http://secure-login-verify.example.com")
    assert feats["num_suspicious_words"] >= 2
