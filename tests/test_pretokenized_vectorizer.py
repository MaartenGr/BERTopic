"""Tests for PR13c: tokenized_documents for direct vectorizer callers.

Tests that hierarchical_topics, reduce_outliers, and _extract_representative_docs
correctly handle pre-tokenized documents.

Run from BERTopic repo root:
    pytest tests/test_pretokenized_vectorizer.py -v
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


class TestHierarchicalTopicsTokenizedDocuments:
    """Test that hierarchical_topics uses tokenized_documents."""

    def test_signature_accepts_tokenized_documents(self):
        """Verify hierarchical_topics has tokenized_documents parameter."""
        import inspect

        sig = inspect.signature(BERTopic.hierarchical_topics)
        assert "tokenized_documents" in sig.parameters

    def test_hierarchical_topics_with_tokenized(self, base_topic_model, documents, tokenized_documents):
        """hierarchical_topics should work with tokenized_documents."""
        model = copy.deepcopy(base_topic_model)
        hier = model.hierarchical_topics(
            documents,
            tokenized_documents=tokenized_documents,
        )
        assert isinstance(hier, pd.DataFrame)
        assert len(hier) > 0


class TestReduceOutliersTokenizedDocuments:
    """Test that reduce_outliers uses tokenized_documents in c-tf-idf and distributions."""

    def test_ctfidf_strategy_with_tokenized(self, base_topic_model, documents, tokenized_documents):
        """reduce_outliers c-tf-idf strategy should accept tokenized_documents."""
        model = copy.deepcopy(base_topic_model)
        topics = model.topics_

        if -1 in topics:
            new_topics = model.reduce_outliers(
                documents,
                topics,
                tokenized_documents=tokenized_documents,
                strategy="c-tf-idf",
                threshold=0.0,
            )
            assert len(new_topics) == len(documents)

    # test_distributions_strategy_with_tokenized removed: approximate_distribution
    # does not yet support tokenized_documents (PR13d scope)


class TestTokenizedDocumentAggregation:
    """Test that Tokenized_Document column is correctly aggregated with 'sum'."""

    def test_tuple_sum_concatenates(self):
        """Verify that 'sum' aggregation on tuples concatenates them."""
        df = pd.DataFrame(
            {
                "Document": ["a b", "c d", "e f"],
                "Topic": [0, 0, 1],
                "Tokenized_Document": [("a", "b"), ("c", "d"), ("e", "f")],
            }
        )
        agg_dict = {"Document": " ".join, "Tokenized_Document": "sum"}
        result = df.groupby(["Topic"], as_index=False).agg(agg_dict)

        assert "Tokenized_Document" in result.columns
        assert result.loc[result.Topic == 0, "Tokenized_Document"].iloc[0] == ("a", "b", "c", "d")

    def test_without_tokenized_column_unchanged(self):
        """When Tokenized_Document is not present, aggregation is unchanged."""
        df = pd.DataFrame(
            {
                "Document": ["a", "b", "c"],
                "Topic": [0, 0, 1],
            }
        )
        agg_dict = {"Document": " ".join}
        if "Tokenized_Document" in df.columns:
            agg_dict["Tokenized_Document"] = "sum"
        result = df.groupby(["Topic"], as_index=False).agg(agg_dict)

        assert "Tokenized_Document" not in result.columns
        assert result.loc[result.Topic == 0, "Document"].iloc[0] == "a b"
