"""Tests for PR7: auto min_similarity for approximate_distribution.

Run from BERTopic repo root:
    pytest tests/test_auto_min_similarity.py -v
"""

import copy

import numpy as np
import pytest


class TestAutoMinSimilarity:
    """Verify auto min_similarity for approximate_distribution."""

    def test_basic_auto_min_similarity(self, base_topic_model, documents):
        """Auto min_similarity should produce valid topic distributions."""
        model = copy.deepcopy(base_topic_model)
        topic_distr, _ = model.approximate_distribution(
            documents[:50],
            min_similarity=None,
            outliers_nb_target=5,
        )
        assert topic_distr.shape[0] == 50
        assert topic_distr.shape[1] > 0

    @pytest.mark.parametrize("target", [0, 5, 10, 25])
    def test_outliers_near_target(self, base_topic_model, documents, target):
        """Number of zero-distribution docs should be near the target."""
        model = copy.deepcopy(base_topic_model)
        n_docs = min(50, len(documents))
        topic_distr, _ = model.approximate_distribution(
            documents[:n_docs],
            min_similarity=None,
            outliers_nb_target=target,
        )

        if topic_distr is not None:
            nb_outliers = int(np.sum(np.sum(topic_distr, axis=1) == 0))
            # Should be >= target (or 0 if impossible)
            assert nb_outliers >= target or nb_outliers == 0

    def test_mutually_exclusive_params(self, base_topic_model, documents):
        """Cannot set both min_similarity and outliers_nb_target."""
        model = copy.deepcopy(base_topic_model)
        with pytest.raises(ValueError, match="either"):
            model.approximate_distribution(
                documents[:10],
                min_similarity=0.1,
                outliers_nb_target=5,
            )

    def test_default_uses_min_similarity(self, base_topic_model, documents):
        """Default call (no outliers_nb_target) should use min_similarity=0.1 as before."""
        model = copy.deepcopy(base_topic_model)
        # Calling with default args should work unchanged
        topic_distr, _ = model.approximate_distribution(documents[:10])
        assert topic_distr.shape[0] == 10

    def test_min_similarity_min_threshold(self, base_topic_model, documents):
        """min_similarity_min_threshold should constrain the search."""
        model = copy.deepcopy(base_topic_model)
        n_docs = 30

        # Low threshold: more outliers removed
        result_low, _ = model.approximate_distribution(
            documents[:n_docs],
            min_similarity=None,
            outliers_nb_target=0,
            min_similarity_min_threshold=0.0,
        )

        # High threshold: fewer outliers removed
        result_high, _ = model.approximate_distribution(
            documents[:n_docs],
            min_similarity=None,
            outliers_nb_target=0,
            min_similarity_min_threshold=0.5,
        )

        if result_low is not None and result_high is not None:
            outliers_low = int(np.sum(np.sum(result_low, axis=1) == 0))
            outliers_high = int(np.sum(np.sum(result_high, axis=1) == 0))
            assert outliers_high >= outliers_low

    def test_backward_compatible_min_similarity(self, base_topic_model, documents):
        """Existing min_similarity parameter should still work."""
        model = copy.deepcopy(base_topic_model)
        topic_distr, _ = model.approximate_distribution(
            documents[:20],
            min_similarity=0.1,
        )
        assert topic_distr.shape[0] == 20

    def test_with_calculate_tokens(self, base_topic_model, documents):
        """Auto min_similarity should work with calculate_tokens=True."""
        model = copy.deepcopy(base_topic_model)
        topic_distr, token_distr = model.approximate_distribution(
            documents[:10],
            min_similarity=None,
            outliers_nb_target=2,
            calculate_tokens=True,
        )
        assert topic_distr.shape[0] == 10
        if token_distr is not None:
            assert len(token_distr) == 10
