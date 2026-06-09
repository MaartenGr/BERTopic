"""Tests for OnlineCountVectorizer with pre-tokenized documents.

These tests verify that OnlineCountVectorizer.partial_fit() and .update_bow()
work correctly when the analyzer is a callable that accepts token lists,
enabling the pre-tokenized documents feature (PR5) to work with online learning.

Run from BERTopic repo root:
    pytest tests/test_vectorizers/test_online_cv_pretokenized.py -v

Note: These would be added to BERTopic's test_vectorizers/ directory alongside
the existing test_online_cv.py.
"""

import pytest

from bertopic.vectorizers import OnlineCountVectorizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def identity_analyzer():
    """An analyzer that passes through pre-tokenized documents."""

    def _analyzer(doc):
        return doc

    return _analyzer


@pytest.fixture
def pretokenized_vectorizer(identity_analyzer):
    """OnlineCountVectorizer configured for pre-tokenized input."""
    return OnlineCountVectorizer(analyzer=identity_analyzer, stop_words=None)


@pytest.fixture
def pretokenized_vectorizer_with_decay(identity_analyzer):
    """OnlineCountVectorizer with decay, configured for pre-tokenized input."""
    return OnlineCountVectorizer(analyzer=identity_analyzer, decay=0.1)


@pytest.fixture
def pretokenized_vectorizer_with_cleanup(identity_analyzer):
    """OnlineCountVectorizer with delete_min_df, configured for pre-tokenized input."""
    return OnlineCountVectorizer(analyzer=identity_analyzer, delete_min_df=2)


@pytest.fixture
def token_batches():
    """Three batches of pre-tokenized documents for online learning simulation."""
    return [
        [["topic", "modeling", "bert"], ["deep", "learning", "bert"]],
        [["topic", "analysis", "novel"], ["clustering", "bert", "method"]],
        [["topic", "summarization", "text"], ["novel", "method", "analysis"]],
    ]


@pytest.fixture
def tokenized_documents(documents):
    """Pre-tokenize the conftest documents fixture using simple whitespace split."""
    return [doc.lower().split() for doc in documents]


# ---------------------------------------------------------------------------
# partial_fit — vocabulary building with token lists
# ---------------------------------------------------------------------------
class TestPartialFitPreTokenized:
    def test_first_partial_fit_builds_vocabulary(self, pretokenized_vectorizer):
        """First partial_fit should build vocabulary from token lists."""
        docs = [["hello", "world"], ["foo", "bar"]]
        pretokenized_vectorizer.partial_fit(docs)
        assert set(pretokenized_vectorizer.vocabulary_.keys()) == {"hello", "world", "foo", "bar"}

    def test_subsequent_partial_fit_adds_oov(self, pretokenized_vectorizer):
        """Subsequent partial_fit should add only new tokens to vocabulary."""
        pretokenized_vectorizer.partial_fit([["hello", "world"]])
        initial_size = len(pretokenized_vectorizer.vocabulary_)

        pretokenized_vectorizer.partial_fit([["hello", "new_token"]])
        assert len(pretokenized_vectorizer.vocabulary_) == initial_size + 1
        assert "new_token" in pretokenized_vectorizer.vocabulary_

    def test_partial_fit_no_vocabulary_duplication(self, pretokenized_vectorizer):
        """Repeated tokens across batches should not duplicate vocabulary entries."""
        pretokenized_vectorizer.partial_fit([["hello", "world"]])
        pretokenized_vectorizer.partial_fit([["hello", "world", "new"]])
        assert len(pretokenized_vectorizer.vocabulary_) == 3

    def test_partial_fit_returns_self(self, pretokenized_vectorizer):
        """partial_fit should return self for method chaining."""
        result = pretokenized_vectorizer.partial_fit([["a", "b"]])
        assert result is pretokenized_vectorizer

    def test_partial_fit_empty_token_list(self, pretokenized_vectorizer):
        """Empty token lists should not modify vocabulary."""
        pretokenized_vectorizer.partial_fit([["hello"]])
        vocab_before = dict(pretokenized_vectorizer.vocabulary_)
        pretokenized_vectorizer.partial_fit([[]])
        assert pretokenized_vectorizer.vocabulary_ == vocab_before

    def test_partial_fit_single_token_documents(self, pretokenized_vectorizer):
        """Single-token documents should work correctly."""
        pretokenized_vectorizer.partial_fit([["alpha"], ["beta"], ["gamma"]])
        assert len(pretokenized_vectorizer.vocabulary_) == 3

    def test_partial_fit_with_tuples(self, pretokenized_vectorizer):
        """Tuples (used for DataFrame hashability) should work like lists."""
        pretokenized_vectorizer.partial_fit([("hello", "world"), ("foo",)])
        assert "hello" in pretokenized_vectorizer.vocabulary_
        assert "foo" in pretokenized_vectorizer.vocabulary_


# ---------------------------------------------------------------------------
# update_bow — bag-of-words with token lists
# ---------------------------------------------------------------------------
class TestUpdateBowPreTokenized:
    def test_update_bow_shape(self, pretokenized_vectorizer):
        """update_bow should produce correctly shaped sparse matrix."""
        docs = [["hello", "world"], ["foo", "bar"]]
        pretokenized_vectorizer.partial_fit(docs)
        X = pretokenized_vectorizer.update_bow(docs)
        assert X.shape == (2, 4)

    def test_update_bow_counts(self, pretokenized_vectorizer):
        """update_bow should correctly count token occurrences."""
        docs = [["a", "a", "b"], ["b", "c"]]
        pretokenized_vectorizer.partial_fit(docs)
        X = pretokenized_vectorizer.update_bow(docs)
        dense = X.toarray()
        a_idx = pretokenized_vectorizer.vocabulary_["a"]
        b_idx = pretokenized_vectorizer.vocabulary_["b"]
        assert dense[0, a_idx] == 2
        assert dense[0, b_idx] == 1
        assert dense[1, b_idx] == 1

    def test_update_bow_accumulates(self, pretokenized_vectorizer):
        """Successive update_bow calls should accumulate the BoW matrix."""
        pretokenized_vectorizer.partial_fit([["hello", "world"]])
        pretokenized_vectorizer.update_bow([["hello", "world"]])

        pretokenized_vectorizer.partial_fit([["hello", "new"]])
        pretokenized_vectorizer.update_bow([["hello", "new"]])

        assert pretokenized_vectorizer.X_.shape[1] == 3  # hello, world, new

    def test_update_bow_empty_document(self, pretokenized_vectorizer):
        """Empty token lists should produce zero rows in BoW matrix."""
        docs = [["hello"], []]
        pretokenized_vectorizer.partial_fit(docs)
        X = pretokenized_vectorizer.update_bow(docs)
        assert X.shape[0] == 2
        assert X.toarray()[1].sum() == 0

    def test_update_bow_with_tuples(self, pretokenized_vectorizer):
        """Tuples should work as input to update_bow."""
        docs = [("a", "b"), ("c",)]
        pretokenized_vectorizer.partial_fit(docs)
        X = pretokenized_vectorizer.update_bow(docs)
        assert X.shape == (2, 3)


# ---------------------------------------------------------------------------
# Decay with token lists
# ---------------------------------------------------------------------------
class TestDecayPreTokenized:
    def test_decay_reduces_previous_counts(self, pretokenized_vectorizer_with_decay):
        """Decay should reduce counts from previous batches."""
        v = pretokenized_vectorizer_with_decay
        v.partial_fit([["hello", "world"]])
        v.update_bow([["hello", "world"]])

        v.partial_fit([["hello", "new"]])
        v.update_bow([["hello", "new"]])

        world_idx = v.vocabulary_["world"]
        # world was 1.0, then decayed by 0.1 → 0.9, not incremented in batch 2
        assert v.X_.toarray()[0, world_idx] == pytest.approx(0.9, abs=0.01)


# ---------------------------------------------------------------------------
# delete_min_df with token lists
# ---------------------------------------------------------------------------
class TestDeleteMinDfPreTokenized:
    def test_delete_min_df_removes_rare_tokens(self, pretokenized_vectorizer_with_cleanup):
        """Tokens below delete_min_df threshold should be removed from vocabulary."""
        v = pretokenized_vectorizer_with_cleanup
        # "rare" appears once, "common" appears twice
        docs = [["common", "rare"], ["common", "also_common"]]
        v.partial_fit(docs)
        v.update_bow(docs)
        assert "rare" not in v.vocabulary_
        assert "common" in v.vocabulary_


# ---------------------------------------------------------------------------
# Full online learning cycle with token lists
# ---------------------------------------------------------------------------
class TestOnlineLearningCyclePreTokenized:
    def test_multi_batch_cycle(self, pretokenized_vectorizer_with_decay, token_batches):
        """Simulate multiple batches of online learning with pre-tokenized input."""
        v = pretokenized_vectorizer_with_decay
        for batch in token_batches:
            v.partial_fit(batch)
            X = v.update_bow(batch)
            assert X.shape[0] == len(batch)

        all_tokens = {token for batch in token_batches for doc in batch for token in doc}
        for token in all_tokens:
            assert token in v.vocabulary_, f"Missing token: {token}"
        assert v.X_.shape[1] == len(all_tokens)

    def test_vocabulary_grows_monotonically(self, pretokenized_vectorizer, token_batches):
        """Vocabulary size should only grow or stay the same across batches."""
        v = pretokenized_vectorizer
        prev_size = 0
        for batch in token_batches:
            v.partial_fit(batch)
            v.update_bow(batch)
            assert len(v.vocabulary_) >= prev_size
            prev_size = len(v.vocabulary_)
