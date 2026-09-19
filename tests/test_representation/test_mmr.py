import numpy as np
import pytest

from bertopic.representation._mmr import mmr


@pytest.fixture
def doc_embedding():
    rng = np.random.RandomState(0)
    return rng.randn(1, 8)


def test_mmr_top_n_exceeds_available_words(doc_embedding):
    """top_n greater than the number of candidate words must not crash.

    Regression test for https://github.com/MaartenGr/BERTopic/issues/2266:
    a topic can have fewer extracted candidate words than the configured
    top_n_words (e.g. after ClassTfidfTransformer tuning), and the MMR loop
    ran range(top_n - 1) iterations against a shrinking candidate list with
    no bound, calling np.argmax on an empty array once every candidate had
    already been selected.
    """
    rng = np.random.RandomState(0)
    words = ["a", "b", "c"]
    word_embeddings = rng.randn(3, 8)

    result = mmr(doc_embedding, word_embeddings, words, diversity=0.5, top_n=10)

    assert len(result) == len(words)
    assert set(result) == set(words)


def test_mmr_top_n_equals_available_words(doc_embedding):
    rng = np.random.RandomState(0)
    words = ["x", "y"]
    word_embeddings = rng.randn(2, 8)

    result = mmr(doc_embedding, word_embeddings, words, diversity=0.5, top_n=2)

    assert len(result) == 2
    assert set(result) == set(words)


def test_mmr_single_candidate_word(doc_embedding):
    rng = np.random.RandomState(0)
    result = mmr(doc_embedding, rng.randn(1, 8), ["solo"], diversity=0.5, top_n=10)

    assert result == ["solo"]


def test_mmr_top_n_fewer_than_available_words(doc_embedding):
    """Existing behavior (more candidates than top_n) must be unaffected."""
    rng = np.random.RandomState(0)
    words = [f"w{i}" for i in range(15)]
    word_embeddings = rng.randn(15, 8)

    result = mmr(doc_embedding, word_embeddings, words, diversity=0.5, top_n=10)

    assert len(result) == 10
    assert len(set(result)) == 10
    assert set(result).issubset(set(words))
