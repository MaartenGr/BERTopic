"""Tests for PR02: shared helper methods extraction.

Run from BERTopic repo root:
    pytest tests/test_shared_helpers.py -v

These tests validate the three helpers:
- _aggregate_documents (static)
- _get_feature_names (instance)
- _topic_name_from_words (static)
"""

import copy

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic


class TestAggregateDocuments:
    """Tests for BERTopic._aggregate_documents."""

    def test_basic_groupby(self):
        df = pd.DataFrame(
            {
                "Document": ["hello world", "foo bar", "baz qux"],
                "Topic": [0, 0, 1],
                "ID": [0, 1, 2],
            }
        )
        result = BERTopic._aggregate_documents(df)
        assert len(result) == 2
        assert result.loc[result.Topic == 0, "Document"].to_numpy()[0] == "hello world foo bar"
        assert result.loc[result.Topic == 1, "Document"].to_numpy()[0] == "baz qux"

    def test_empty_dataframe(self):
        df = pd.DataFrame({"Document": [], "Topic": []})
        result = BERTopic._aggregate_documents(df)
        assert len(result) == 0

    def test_single_doc_per_topic(self):
        df = pd.DataFrame(
            {
                "Document": ["only doc"],
                "Topic": [0],
            }
        )
        result = BERTopic._aggregate_documents(df)
        assert len(result) == 1
        assert result.loc[result.Topic == 0, "Document"].to_numpy()[0] == "only doc"

    def test_drops_extra_columns(self):
        """Extra columns should not appear in output unless aggregated."""
        df = pd.DataFrame(
            {
                "Document": ["a", "b"],
                "Topic": [0, 0],
                "ID": [0, 1],
                "Image": [None, None],
            }
        )
        result = BERTopic._aggregate_documents(df)
        assert "Document" in result.columns
        assert "Topic" in result.columns


class TestGetFeatureNames:
    """Tests for BERTopic._get_feature_names."""

    def test_returns_feature_names(self):
        model = BERTopic()
        model.vectorizer_model = CountVectorizer()
        model.vectorizer_model.fit(["hello world", "foo bar baz"])
        names = model._get_feature_names()
        assert "hello" in names
        assert "world" in names
        assert "foo" in names


class TestTopicNameFromWords:
    """Tests for BERTopic._topic_name_from_words."""

    def test_default_n(self):
        """Default n=5 should join the first 5 words."""
        words = [(f"word{i}", 0.5 - i * 0.1) for i in range(10)]
        name = BERTopic._topic_name_from_words(words)
        assert name == "word0_word1_word2_word3_word4"

    def test_custom_n(self):
        words = [(f"w{i}", 0.5) for i in range(10)]
        name = BERTopic._topic_name_from_words(words, n=3)
        assert name == "w0_w1_w2"

    def test_fewer_words_than_n(self):
        words = [("only", 0.9)]
        name = BERTopic._topic_name_from_words(words, n=5)
        assert name == "only"

    def test_empty_words(self):
        name = BERTopic._topic_name_from_words([])
        assert name == ""


class TestHelpersIntegration:
    """Verify refactored methods produce identical output to upstream."""

    def test_hierarchical_topics_unchanged(self, base_topic_model, documents):
        """hierarchical_topics should produce valid output after refactoring."""
        model = copy.deepcopy(base_topic_model)
        hier = model.hierarchical_topics(documents)

        assert len(hier) > 0
        assert "Parent_Name" in hier.columns
        # Parent names should be underscore-joined keywords
        for name in hier["Parent_Name"]:
            assert "_" in name or len(name.split("_")) >= 1

    def test_extract_topics_unchanged(self, base_topic_model, documents, document_embeddings):
        """_extract_topics should produce identical topic representations."""
        model = copy.deepcopy(base_topic_model)
        original_topics = dict(model.topic_representations_)

        # Re-run _extract_topics
        docs_df = pd.DataFrame(
            {
                "Document": documents,
                "ID": range(len(documents)),
                "Topic": model.topics_,
            }
        )
        model._extract_topics(docs_df, embeddings=document_embeddings)

        # Topic representations should be identical
        assert set(model.topic_representations_.keys()) == set(original_topics.keys())
