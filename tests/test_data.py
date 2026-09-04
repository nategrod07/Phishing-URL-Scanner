import pandas as pd
import pytest

from phishing_ml.data import (
    IngestReport,
    generate_synthetic_dataset,
    iter_email_chunks,
    load_email_features,
    load_url_csv,
)


@pytest.fixture
def dataset_dir(tmp_path):
    """A miniature stand-in for the Kaggle corpus.

    CEAS_08 carries the full column set; Enron has only subject/body/label,
    matching the real dataset's shape.
    """
    pd.DataFrame(
        {
            "sender": ["a@x.com", "b@y.com", "c@z.com", "a@x.com"],
            "subject": ["hi", "Re: deal", "WIN NOW", "hi"],
            "body": ["hello", "see http://a.com", "click here", "hello"],
            "urls": [0, 1, 1, 0],
            "label": [0, 0, 1, 0],
        }
    ).to_csv(tmp_path / "CEAS_08.csv", index=False)

    pd.DataFrame(
        {
            "subject": ["meeting", "URGENT verify"],
            "body": ["agenda attached", "confirm your account"],
            "label": [0, 1],
        }
    ).to_csv(tmp_path / "Enron.csv", index=False)

    return str(tmp_path)


class TestIngestReport:
    def test_accounts_for_every_row(self):
        report = IngestReport(
            rows_read=100, dropped_missing_label=5, dropped_duplicates=10, dropped_sampling=25
        )
        assert report.rows_kept == 60
        assert report.retention == pytest.approx(0.6)

    def test_retention_of_empty_read_is_zero(self):
        assert IngestReport().retention == 0.0


class TestChunking:
    def test_reads_every_file(self, dataset_dir):
        seen = {name for name, _ in iter_email_chunks(dataset_dir)}
        assert seen == {"CEAS_08.csv", "Enron.csv"}

    def test_files_missing_optional_columns_still_yield_rows(self, dataset_dir):
        enron = pd.concat(
            [chunk for name, chunk in iter_email_chunks(dataset_dir) if name == "Enron.csv"]
        )
        assert len(enron) == 2
        assert "sender" not in enron.columns

    def test_chunksize_does_not_change_the_result(self, dataset_dir):
        whole, labels_whole, _ = load_email_features(dataset_dir, chunksize=1000)
        split, labels_split, _ = load_email_features(dataset_dir, chunksize=1)
        pd.testing.assert_frame_equal(whole, split)
        pd.testing.assert_series_equal(labels_whole, labels_split)

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No known dataset CSVs"):
            list(iter_email_chunks(str(tmp_path)))


class TestLoadEmailFeatures:
    def test_keeps_all_rows_by_default_except_duplicates(self, dataset_dir):
        features, labels, report = load_email_features(dataset_dir)
        assert report.rows_read == 6
        assert report.dropped_duplicates == 1  # the repeated a@x.com/hi/hello row
        assert report.dropped_sampling == 0
        assert len(features) == len(labels) == 5

    def test_duplicates_can_be_kept(self, dataset_dir):
        _, labels, report = load_email_features(dataset_dir, drop_duplicates=False)
        assert report.dropped_duplicates == 0
        assert len(labels) == 6

    def test_duplicates_are_detected_across_chunk_boundaries(self, dataset_dir):
        # The duplicate pair is rows 0 and 3 of CEAS_08, so a chunksize of 1
        # puts them in different chunks.
        _, _, report = load_email_features(dataset_dir, chunksize=1)
        assert report.dropped_duplicates == 1

    def test_unlabelled_rows_are_dropped_and_counted(self, tmp_path):
        pd.DataFrame(
            {"subject": ["a", "b"], "body": ["x", "y"], "label": [1, None]}
        ).to_csv(tmp_path / "Enron.csv", index=False)
        _, labels, report = load_email_features(str(tmp_path))
        assert report.dropped_missing_label == 1
        assert len(labels) == 1

    def test_labels_are_integers(self, dataset_dir):
        _, labels, _ = load_email_features(dataset_dir)
        assert labels.dtype.kind == "i"
        assert set(labels.unique()) <= {0, 1}

    def test_features_and_labels_are_aligned(self, dataset_dir):
        features, labels, _ = load_email_features(dataset_dir)
        assert list(features.index) == list(labels.index)

    def test_sampling_is_balanced_and_reported(self, dataset_dir):
        features, labels, report = load_email_features(dataset_dir, sample_size=2)
        assert len(features) == 2
        assert labels.value_counts().to_dict() == {0: 1, 1: 1}
        assert report.dropped_sampling == 3

    def test_sample_larger_than_corpus_is_a_no_op(self, dataset_dir):
        _, labels, report = load_email_features(dataset_dir, sample_size=10_000)
        assert report.dropped_sampling == 0
        assert len(labels) == 5

    def test_no_labelled_rows_raises(self, tmp_path):
        pd.DataFrame({"subject": ["a"], "body": ["x"], "label": [None]}).to_csv(
            tmp_path / "Ling.csv", index=False
        )
        with pytest.raises(ValueError, match="No labelled email rows"):
            load_email_features(str(tmp_path))


class TestUrlData:
    def test_synthetic_dataset_is_balanced_and_deterministic(self):
        first = generate_synthetic_dataset(n_per_class=50)
        second = generate_synthetic_dataset(n_per_class=50)
        assert first["label"].value_counts().to_dict() == {0: 50, 1: 50}
        pd.testing.assert_frame_equal(first, second)

    def test_load_url_csv_renames_columns(self, tmp_path):
        path = tmp_path / "urls.csv"
        pd.DataFrame({"link": ["http://a.com"], "is_phish": [1]}).to_csv(path, index=False)
        loaded = load_url_csv(str(path), url_col="link", label_col="is_phish")
        assert list(loaded.columns) == ["url", "label"]
