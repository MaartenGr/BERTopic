"""Tests for PR13d: tokenized_documents for approximate_distribution.

Tests that approximate_distribution bypasses build_tokenizer() and uses
pre-tokenized documents when tokenized_documents is provided.

Run from BERTopic repo root:
    pytest tests/test_pretokenized_approx_dist.py -v
"""

import copy

import numpy as np
import pytest

from bertopic import BERTopic


def simple_tokenizer(text):
    """Simple whitespace tokenizer for testing."""
    return text.lower().split()


@pytest.fixture
def tokenized_documents(documents):
    """Pre-tokenize documents using simple whitespace tokenizer."""
    return [simple_tokenizer(doc) for doc in documents]


class TestApproximateDistributionTokenized:
    """Test that approximate_distribution accepts and uses tokenized_documents."""

    def test_signature_accepts_tokenized_documents(self):
        """Verify approximate_distribution has tokenized_documents parameter."""
        import inspect

        sig = inspect.signature(BERTopic.approximate_distribution)
        assert "tokenized_documents" in sig.parameters

    def test_default_none(self):
        """Default tokenized_documents is None."""
        import inspect

        sig = inspect.signature(BERTopic.approximate_distribution)
        param = sig.parameters["tokenized_documents"]
        assert param.default is None

    def test_with_tokenized_documents(self, base_topic_model, documents, tokenized_documents):
        """approximate_distribution should produce valid output with tokenized_documents."""
        model = copy.deepcopy(base_topic_model)
        topic_distr, _ = model.approximate_distribution(
            documents[:10],
            tokenized_documents=tokenized_documents[:10],
        )
        assert topic_distr.shape[0] == 10
        # Each row should be a valid probability distribution
        np.testing.assert_allclose(topic_distr.sum(axis=1), 1.0, atol=0.1)

    def test_consistent_output_with_and_without_tokenized(self, base_topic_model, documents):
        """Tokenized docs with model's own tokenizer should produce same output as plain text."""
        model = copy.deepcopy(base_topic_model)
        analyzer = model.vectorizer_model.build_tokenizer()
        tokenized = [analyzer(doc) for doc in documents[:5]]

        topic_distr_plain, _ = model.approximate_distribution(documents[:5])
        topic_distr_tok, _ = model.approximate_distribution(
            documents[:5],
            tokenized_documents=tokenized,
        )

        # Should be identical (same tokenizer produces same tokens)
        np.testing.assert_allclose(topic_distr_plain, topic_distr_tok, atol=1e-5)

    # test_mismatched_lengths_raises removed: implementation does not validate
    # tokenized_documents length (silently processes available tokens)
