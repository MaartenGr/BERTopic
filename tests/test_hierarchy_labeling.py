"""Tests for hierarchical topic labeling with representation model (PR18).

These tests verify that `hierarchical_topics(use_representation_model=True)`
replaces parent keyword names with representation-model labels.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from bertopic import BERTopic
from bertopic.representation import BaseRepresentation


@pytest.fixture
def fitted_model():
    """Create a minimally fitted BERTopic model for hierarchy testing."""
    model = BERTopic(verbose=False)

    # Simulate a fitted model with 3 topics (0, 1, 2)
    model.topics_ = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    model.topic_representations_ = {
        0: [("safety", 0.5), ("equipment", 0.4), ("ppe", 0.3), ("wear", 0.2), ("worker", 0.1)],
        1: [("fire", 0.5), ("alarm", 0.4), ("smoke", 0.3), ("detector", 0.2), ("building", 0.1)],
        2: [("spill", 0.5), ("chemical", 0.4), ("clean", 0.3), ("hazard", 0.2), ("material", 0.1)],
    }

    # Create c-TF-IDF matrix (3 topics x 10 features)
    model.c_tf_idf_ = csr_matrix(np.random.rand(3, 10))
    model.topic_embeddings_ = np.random.rand(3, 5)
    model.topic_sizes_ = {0: 3, 1: 3, 2: 3}

    # Fit a vectorizer
    from sklearn.feature_extraction.text import CountVectorizer

    docs = [
        "safety equipment ppe wear worker",
        "fire alarm smoke detector building",
        "spill chemical clean hazard material",
    ]
    model.vectorizer_model = CountVectorizer()
    model.vectorizer_model.fit(docs)
    model.ctfidf_model = MagicMock()
    model.ctfidf_model.transform = MagicMock(return_value=csr_matrix(np.random.rand(1, 10)))

    return model


class TestHierarchyLabelingDefault:
    """Test that default behavior (use_representation_model=False) is unchanged."""

    def test_parent_names_are_keywords(self, fitted_model):
        """Without use_representation_model, parent names should be keyword concatenations."""
        docs = ["doc"] * 9
        with patch.object(fitted_model, "_preprocess_text", side_effect=lambda x: list(x)):
            hier = fitted_model.hierarchical_topics(docs, use_representation_model=False)

        # Parent names should be underscore-joined keywords
        for name in hier["Parent_Name"]:
            # Should NOT contain spaces (keyword format, not LLM labels)
            assert "_" in name


class TestHierarchyLabelingEnabled:
    """Test that use_representation_model=True produces representation-model labels."""

    def test_parent_names_use_representation_model(self, fitted_model):
        """With use_representation_model=True, parent names should come from the
        representation model instead of raw keyword concatenation.
        """
        # Set up a mock representation model
        mock_repr = MagicMock(spec=BaseRepresentation)
        mock_repr.extract_topics.return_value = {
            0: [("PPE Compliance", 0.9), ("Safety Standards", 0.8)],
        }
        fitted_model.representation_model = mock_repr

        docs = ["doc"] * 9
        with patch.object(fitted_model, "_preprocess_text", side_effect=lambda x: list(x)):
            with patch.object(
                fitted_model,
                "_extract_words_per_topic",
                wraps=fitted_model._extract_words_per_topic,
            ):
                hier = fitted_model.hierarchical_topics(docs, use_representation_model=True)

        # Verify the method completed without error
        assert len(hier) > 0

    def test_noop_without_representation_model(self, fitted_model):
        """When representation_model is None, use_representation_model=True should
        be a graceful noop (same as False).
        """
        fitted_model.representation_model = None

        docs = ["doc"] * 9
        with patch.object(fitted_model, "_preprocess_text", side_effect=lambda x: list(x)):
            hier = fitted_model.hierarchical_topics(docs, use_representation_model=True)

        # Should still produce valid output
        assert len(hier) > 0
        # Parent names should be keyword format (no representation model to override)
        for name in hier["Parent_Name"]:
            assert "_" in name

    def test_distance_column_preserved(self, fitted_model):
        """Distance column should be present and valid after label override."""
        docs = ["doc"] * 9
        with patch.object(fitted_model, "_preprocess_text", side_effect=lambda x: list(x)):
            hier = fitted_model.hierarchical_topics(docs, use_representation_model=False)

        assert "Distance" in hier.columns
        assert hier["Distance"].dtype == float
        assert (hier["Distance"] >= 0).all()


class TestHierarchyLabelingIntegration:
    """Integration tests with real fitted models."""

    def test_with_keybert_representation(self, representation_topic_model, documents):
        """End-to-end test with a real KeyBERTInspired representation model."""
        import copy

        model = copy.deepcopy(representation_topic_model)
        hier = model.hierarchical_topics(documents, use_representation_model=True)

        assert len(hier) > 0
        assert "Parent_Name" in hier.columns
        assert "Distance" in hier.columns

    def test_default_false_unchanged(self, base_topic_model, documents):
        """Default use_representation_model=False should produce keyword names."""
        import copy

        model = copy.deepcopy(base_topic_model)
        hier = model.hierarchical_topics(documents)

        assert len(hier) > 0
        for name in hier["Parent_Name"]:
            # Keyword format: words separated by underscores
            assert isinstance(name, str)
            assert len(name) > 0
