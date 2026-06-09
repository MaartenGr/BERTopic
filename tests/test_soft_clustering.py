"""Tests for PR11: soft clustering via temperature-scaled probabilities.

Run from BERTopic repo root:
    pytest tests/test_soft_clustering.py -v
"""

import copy

import numpy as np


class TestSoftClusteringMath:
    """Unit tests for the temperature-scaled softmax math."""

    def test_probability_matrix_shape_and_sums(self):
        """Softmax over distances should produce valid probability distributions."""
        from scipy.special import softmax

        n_docs, n_topics, dim = 5, 3, 10
        np.random.seed(42)
        embeddings = np.random.rand(n_docs, dim)
        topic_embeddings = np.random.rand(n_topics, dim)

        distances = np.linalg.norm(embeddings[:, np.newaxis, :] - topic_embeddings[np.newaxis, :, :], axis=2)
        probs = softmax(-distances / 0.5, axis=1)

        assert probs.shape == (n_docs, n_topics)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_low_temperature_is_sharper(self):
        """Lower temperature should produce sharper (more peaked) distributions."""
        from scipy.special import softmax

        n_docs, n_topics, dim = 10, 3, 10
        np.random.seed(42)
        embeddings = np.random.rand(n_docs, dim)
        topic_embeddings = np.random.rand(n_topics, dim)

        distances = np.linalg.norm(embeddings[:, np.newaxis, :] - topic_embeddings[np.newaxis, :, :], axis=2)

        probs_low = softmax(-distances / 0.1, axis=1)
        probs_high = softmax(-distances / 10.0, axis=1)

        entropy_low = -np.sum(probs_low * np.log(probs_low + 1e-10), axis=1).mean()
        entropy_high = -np.sum(probs_high * np.log(probs_high + 1e-10), axis=1).mean()
        assert entropy_low < entropy_high

    def test_high_temperature_approaches_uniform(self):
        """Very high temperature should produce near-uniform distributions."""
        from scipy.special import softmax

        n_docs, n_topics, dim = 5, 4, 10
        np.random.seed(42)
        embeddings = np.random.rand(n_docs, dim)
        topic_embeddings = np.random.rand(n_topics, dim)

        distances = np.linalg.norm(embeddings[:, np.newaxis, :] - topic_embeddings[np.newaxis, :, :], axis=2)

        probs = softmax(-distances / 1000.0, axis=1)
        expected_uniform = 1.0 / n_topics
        np.testing.assert_allclose(probs, expected_uniform, atol=0.01)


class TestSoftClusteringIntegration:
    """Integration tests calling model.transform() with soft_clustering_temp."""

    def test_transform_with_soft_clustering(self, base_topic_model, documents, document_embeddings):
        """Transform with soft_clustering_temp should return 2D probability matrix."""
        model = copy.deepcopy(base_topic_model)

        topics, probs = model.transform(
            documents[:20],
            embeddings=document_embeddings[:20],
            soft_clustering_temp=0.5,
        )

        assert len(topics) == 20
        # Probabilities should be a 2D matrix
        assert probs.ndim == 2
        assert probs.shape[0] == 20
        # Each row should sum to ~1
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_hard_predictions_unchanged_with_soft_clustering(self, base_topic_model, documents, document_embeddings):
        """Hard topic predictions should not change when soft_clustering_temp is set."""
        model = copy.deepcopy(base_topic_model)

        topics_hard, _ = model.transform(
            documents[:20],
            embeddings=document_embeddings[:20],
        )
        topics_soft, _ = model.transform(
            documents[:20],
            embeddings=document_embeddings[:20],
            soft_clustering_temp=0.5,
        )

        assert topics_hard == topics_soft

    def test_none_temp_returns_default_probabilities(self, base_topic_model, documents, document_embeddings):
        """soft_clustering_temp=None should not change default behavior."""
        model = copy.deepcopy(base_topic_model)

        _, probs_default = model.transform(
            documents[:10],
            embeddings=document_embeddings[:10],
        )
        _, probs_none = model.transform(
            documents[:10],
            embeddings=document_embeddings[:10],
            soft_clustering_temp=None,
        )

        np.testing.assert_array_equal(probs_default, probs_none)

    def test_different_temperatures_produce_different_probs(self, base_topic_model, documents, document_embeddings):
        """Different temperatures should produce different probability distributions."""
        model = copy.deepcopy(base_topic_model)

        _, probs_low = model.transform(
            documents[:10],
            embeddings=document_embeddings[:10],
            soft_clustering_temp=0.1,
        )
        _, probs_high = model.transform(
            documents[:10],
            embeddings=document_embeddings[:10],
            soft_clustering_temp=10.0,
        )

        # Should not be identical
        assert not np.allclose(probs_low, probs_high)
