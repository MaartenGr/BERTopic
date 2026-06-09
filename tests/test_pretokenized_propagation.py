"""Tests for PR13b: tokenized_documents propagation to remaining pipeline methods.

Tests that partial_fit, topics_over_time, topics_per_class, merge_topics, and
reduce_topics correctly propagate the Tokenized_Document column through the pipeline.

Run from BERTopic repo root:
    pytest tests/test_pretokenized_propagation.py -v
"""

import copy

import pandas as pd
import pytest

from bertopic import BERTopic


def simple_tokenizer(text):
    """Simple whitespace tokenizer for testing."""
    return text.lower().split()


@pytest.fixture
def tokenized_documents(documents):
    """Pre-tokenize documents using simple whitespace tokenizer."""
    return [simple_tokenizer(doc) for doc in documents]


class TestPartialFitTokenizedDocuments:
    """Test that partial_fit accepts and propagates tokenized_documents."""

    def test_partial_fit_signature_accepts_tokenized_documents(self):
        """Verify partial_fit has tokenized_documents parameter."""
        import inspect

        sig = inspect.signature(BERTopic.partial_fit)
        assert "tokenized_documents" in sig.parameters

    def test_partial_fit_default_none(self):
        """Default tokenized_documents is None — backward compatible."""
        import inspect

        sig = inspect.signature(BERTopic.partial_fit)
        param = sig.parameters["tokenized_documents"]
        assert param.default is None


class TestTopicsOverTimeTokenizedDocuments:
    """Test that topics_over_time accepts and propagates tokenized_documents."""

    def test_signature_accepts_tokenized_documents(self):
        """Verify topics_over_time has tokenized_documents parameter."""
        import inspect

        sig = inspect.signature(BERTopic.topics_over_time)
        assert "tokenized_documents" in sig.parameters

    def test_topics_over_time_with_tokenized(self, base_topic_model, documents, tokenized_documents):
        """topics_over_time should accept tokenized_documents and produce results."""
        model = copy.deepcopy(base_topic_model)
        timestamps = [f"2024-0{(i % 3) + 1}" for i in range(len(documents))]

        result = model.topics_over_time(
            documents,
            timestamps,
            tokenized_documents=tokenized_documents,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestTopicsPerClassTokenizedDocuments:
    """Test that topics_per_class accepts and propagates tokenized_documents."""

    def test_signature_accepts_tokenized_documents(self):
        """Verify topics_per_class has tokenized_documents parameter."""
        import inspect

        sig = inspect.signature(BERTopic.topics_per_class)
        assert "tokenized_documents" in sig.parameters

    def test_topics_per_class_with_tokenized(self, base_topic_model, documents, tokenized_documents):
        """topics_per_class should accept tokenized_documents and produce results."""
        model = copy.deepcopy(base_topic_model)
        classes = [f"class_{i % 3}" for i in range(len(documents))]

        result = model.topics_per_class(
            documents,
            classes=classes,
            tokenized_documents=tokenized_documents,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestMergeTopicsTokenizedDocuments:
    """Test that merge_topics accepts and propagates tokenized_documents."""

    def test_signature_accepts_tokenized_documents(self):
        """Verify merge_topics has tokenized_documents parameter."""
        import inspect

        sig = inspect.signature(BERTopic.merge_topics)
        assert "tokenized_documents" in sig.parameters

    def test_merge_topics_with_tokenized(self, base_topic_model, documents, tokenized_documents, document_embeddings):
        """merge_topics should accept tokenized_documents."""
        model = copy.deepcopy(base_topic_model)
        unique_topics = [t for t in set(model.topics_) if t != -1]
        if len(unique_topics) >= 2:
            topics_to_merge = unique_topics[:2]
            model.merge_topics(
                documents,
                topics_to_merge,
                tokenized_documents=tokenized_documents,
            )
            assert hasattr(model, "topic_representations_")


class TestReduceTopicsTokenizedDocuments:
    """Test that reduce_topics accepts and propagates tokenized_documents."""

    def test_signature_accepts_tokenized_documents(self):
        """Verify reduce_topics has tokenized_documents parameter."""
        import inspect

        sig = inspect.signature(BERTopic.reduce_topics)
        assert "tokenized_documents" in sig.parameters

    def test_reduce_topics_with_tokenized(self, base_topic_model, documents, tokenized_documents):
        """reduce_topics should accept tokenized_documents."""
        model = copy.deepcopy(base_topic_model)
        n_topics = max(2, len(set(model.topics_)) - 1)
        model.reduce_topics(
            documents,
            nr_topics=n_topics,
            tokenized_documents=tokenized_documents,
        )
        assert len(set(model.topics_) - {-1}) <= n_topics


class TestTokenizedDocumentAggregation:
    """Test that Tokenized_Document column is correctly aggregated with 'sum'."""

    def test_tuple_sum_concatenates(self):
        """Verify that 'sum' aggregation on tuples concatenates them."""
        df = pd.DataFrame(
            {
                "Topic": [0, 0, 1, 1],
                "Document": ["a b", "c d", "e f", "g h"],
                "Tokenized_Document": [
                    ("a", "b"),
                    ("c", "d"),
                    ("e", "f"),
                    ("g", "h"),
                ],
            }
        )
        agg_dict = {"Document": " ".join, "Tokenized_Document": "sum"}
        result = df.groupby(["Topic"], as_index=False).agg(agg_dict)

        assert result.loc[result.Topic == 0, "Tokenized_Document"].iloc[0] == ("a", "b", "c", "d")
        assert result.loc[result.Topic == 1, "Tokenized_Document"].iloc[0] == ("e", "f", "g", "h")

    def test_conditional_agg_without_column(self):
        """When Tokenized_Document is not present, aggregation is unchanged."""
        df = pd.DataFrame(
            {
                "Topic": [0, 0, 1],
                "Document": ["a", "b", "c"],
            }
        )
        agg_dict = {"Document": " ".join}
        if "Tokenized_Document" in df.columns:
            agg_dict["Tokenized_Document"] = "sum"
        result = df.groupby(["Topic"], as_index=False).agg(agg_dict)

        assert "Tokenized_Document" not in result.columns
        assert result.loc[result.Topic == 0, "Document"].iloc[0] == "a b"

    def test_conditional_agg_with_timestamps(self):
        """Aggregation preserves Timestamps count alongside Tokenized_Document sum."""
        df = pd.DataFrame(
            {
                "Topic": [0, 0, 1],
                "Document": ["a", "b", "c"],
                "Timestamps": ["2024-01", "2024-01", "2024-01"],
                "Tokenized_Document": [("a",), ("b",), ("c",)],
            }
        )
        agg_dict = {"Document": " ".join, "Timestamps": "count"}
        if "Tokenized_Document" in df.columns:
            agg_dict["Tokenized_Document"] = "sum"
        result = df.groupby(["Topic"], as_index=False).agg(agg_dict)

        assert result.loc[result.Topic == 0, "Timestamps"].iloc[0] == 2
        assert result.loc[result.Topic == 0, "Tokenized_Document"].iloc[0] == ("a", "b")
