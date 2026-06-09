"""Tests for batched embedding extraction in MMR representation.

Run from BERTopic repo root:
    pytest tests/test_mmr_batched_embeddings.py -v
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Unit tests (mock model)
# ---------------------------------------------------------------------------


def test_single_embedding_call():
    """_extract_embeddings should be called exactly once, not 2N times."""
    from bertopic.representation._mmr import MaximalMarginalRelevance

    mmr_model = MaximalMarginalRelevance(diversity=0.1, top_n_words=5)

    # Mock topic model with embedding support
    topic_model = MagicMock()
    topic_model.embedding_model = MagicMock()

    # Create sample topics (3 topics, 5 words each)
    topics = {
        0: [("word1", 0.5), ("word2", 0.4), ("word3", 0.3), ("word4", 0.2), ("word5", 0.1)],
        1: [("alpha", 0.6), ("beta", 0.5), ("gamma", 0.4), ("delta", 0.3), ("epsilon", 0.2)],
        2: [("foo", 0.7), ("bar", 0.6), ("baz", 0.5), ("qux", 0.4), ("quux", 0.3)],
    }

    # Total items: 5 words + 1 sentence per topic = 18 items
    total_items = sum(len(words) + 1 for words in topics.values())
    embedding_dim = 10
    fake_embeddings = np.random.rand(total_items, embedding_dim)
    topic_model._extract_embeddings.return_value = fake_embeddings

    documents = pd.DataFrame()
    c_tf_idf = csr_matrix((3, 10))

    result = mmr_model.extract_topics(topic_model, documents, c_tf_idf, topics)

    # Verify exactly 1 call to _extract_embeddings
    assert topic_model._extract_embeddings.call_count == 1, (
        f"Expected 1 embedding call, got {topic_model._extract_embeddings.call_count}"
    )

    # Verify all topics are present in result
    assert set(result.keys()) == set(topics.keys())


def test_empty_topics_dict():
    """Empty topics dict should return empty dict."""
    from bertopic.representation._mmr import MaximalMarginalRelevance

    mmr_model = MaximalMarginalRelevance(diversity=0.1, top_n_words=5)
    topic_model = MagicMock()
    topic_model.embedding_model = MagicMock()

    result = mmr_model.extract_topics(topic_model, pd.DataFrame(), csr_matrix((0, 0)), {})
    assert result == {}


def test_single_word_topic():
    """Topic with only 1 word should not crash."""
    from bertopic.representation._mmr import MaximalMarginalRelevance

    mmr_model = MaximalMarginalRelevance(diversity=0.1, top_n_words=5)
    topic_model = MagicMock()
    topic_model.embedding_model = MagicMock()

    topics = {0: [("onlyword", 0.9)]}
    # 1 word + 1 sentence = 2 items
    embedding_dim = 10
    topic_model._extract_embeddings.return_value = np.random.rand(2, embedding_dim)

    result = mmr_model.extract_topics(topic_model, pd.DataFrame(), csr_matrix((1, 1)), topics)
    assert 0 in result
    assert len(result[0]) == 1


def test_no_embedding_model_returns_topics_unchanged():
    """When no embedding model is set, topics should be returned unchanged."""
    from bertopic.representation._mmr import MaximalMarginalRelevance

    mmr_model = MaximalMarginalRelevance()
    topic_model = MagicMock()
    topic_model.embedding_model = None

    topics = {0: [("word", 0.5)]}
    result = mmr_model.extract_topics(topic_model, pd.DataFrame(), csr_matrix((1, 1)), topics)
    assert result == topics


# ---------------------------------------------------------------------------
# Integration tests (real fitted model from conftest fixtures)
# ---------------------------------------------------------------------------


def test_output_matches_integration(base_topic_model, documents, document_embeddings):
    """Batched MMR should produce valid topic representations for all topics."""
    import copy

    from bertopic.representation._mmr import MaximalMarginalRelevance

    model = copy.deepcopy(base_topic_model)

    topics = model.topic_representations_
    if not topics:
        pytest.skip("No topic representations available")

    mmr_model = MaximalMarginalRelevance(diversity=0.1, top_n_words=10)
    result = mmr_model.extract_topics(model, pd.DataFrame(), model.c_tf_idf_, topics)

    # All topics should be present
    assert set(result.keys()) == set(topics.keys())
    # Each topic should have word-score pairs
    for topic, words in result.items():
        assert len(words) > 0
        for word, score in words:
            assert isinstance(word, str)
            assert isinstance(score, float)
