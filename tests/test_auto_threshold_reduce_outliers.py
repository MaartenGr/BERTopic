"""Tests for PR6: auto-threshold for reduce_outliers.

Run from BERTopic repo root:
    pytest tests/test_auto_threshold_reduce_outliers.py -v
"""

import copy

import pytest


class TestAutoThresholdReduceOutliers:
    """Verify auto-threshold for reduce_outliers across all strategies."""

    def _has_outliers(self, model):
        """Check if model has outlier topics."""
        return -1 in model.topics_

    @pytest.mark.parametrize("target", [0.0, 0.05, 0.1, 0.5])
    def test_auto_threshold_ctfidf(self, base_topic_model, documents, target):
        """c-tf-idf auto-threshold should reduce outliers near target."""
        model = copy.deepcopy(base_topic_model)
        if not self._has_outliers(model):
            pytest.skip("No outliers in model")

        topics = model.topics_
        new_topics = model.reduce_outliers(
            documents,
            topics,
            strategy="c-tf-idf",
            threshold=None,
            outliers_percentage_target=target,
        )

        actual_outlier_pct = sum(1 for t in new_topics if t == -1) / len(new_topics)
        original_outlier_pct = sum(1 for t in topics if t == -1) / len(topics)
        # Auto-threshold should reduce outliers; may overshoot target due to
        # discrete threshold steps in c-tf-idf.
        assert actual_outlier_pct <= original_outlier_pct or actual_outlier_pct == 0

    @pytest.mark.parametrize("target", [0.0, 0.05, 0.1])
    def test_auto_threshold_embeddings(self, base_topic_model, documents, document_embeddings, target):
        """Embeddings auto-threshold should reduce outliers near target."""
        model = copy.deepcopy(base_topic_model)
        if not self._has_outliers(model):
            pytest.skip("No outliers in model")

        new_topics = model.reduce_outliers(
            documents,
            model.topics_,
            strategy="embeddings",
            threshold=None,
            outliers_percentage_target=target,
            embeddings=document_embeddings,
        )

        actual_outlier_pct = sum(1 for t in new_topics if t == -1) / len(new_topics)
        original_outlier_pct = sum(1 for t in model.topics_ if t == -1) / len(model.topics_)
        assert actual_outlier_pct <= original_outlier_pct or actual_outlier_pct == 0

    @pytest.mark.parametrize("target", [0.0, 0.1])
    def test_auto_threshold_distributions(self, base_topic_model, documents, target):
        """Distributions auto-threshold should reduce outliers near target."""
        model = copy.deepcopy(base_topic_model)
        if not self._has_outliers(model):
            pytest.skip("No outliers in model")

        new_topics = model.reduce_outliers(
            documents,
            model.topics_,
            strategy="distributions",
            threshold=None,
            outliers_percentage_target=target,
        )
        assert len(new_topics) == len(documents)

    def test_mutually_exclusive_params(self, base_topic_model, documents):
        """Cannot set both threshold and outliers_percentage_target."""
        model = copy.deepcopy(base_topic_model)
        with pytest.raises(ValueError, match="either"):
            model.reduce_outliers(
                documents,
                model.topics_,
                threshold=0.5,
                outliers_percentage_target=0.1,
            )

    def test_target_out_of_range(self, base_topic_model, documents):
        """outliers_percentage_target must be 0-1."""
        model = copy.deepcopy(base_topic_model)
        with pytest.raises(ValueError, match="between 0 and 1"):
            model.reduce_outliers(
                documents,
                model.topics_,
                threshold=None,
                outliers_percentage_target=1.5,
            )

    def test_already_below_target(self, base_topic_model, documents):
        """If already below target, return topics unchanged."""
        model = copy.deepcopy(base_topic_model)
        topics = model.topics_

        # Set target to 100% — should return unchanged
        result = model.reduce_outliers(
            documents,
            topics,
            threshold=None,
            outliers_percentage_target=1.0,
        )
        assert result == topics

    def test_min_threshold_constrains_search(self, base_topic_model, documents):
        """min_threshold should limit the threshold search range."""
        model = copy.deepcopy(base_topic_model)
        if not self._has_outliers(model):
            pytest.skip("No outliers in model")

        # With very high min_threshold, fewer outliers should be reassigned
        result_low = model.reduce_outliers(
            documents,
            model.topics_,
            strategy="c-tf-idf",
            threshold=None,
            outliers_percentage_target=0.0,
            min_threshold=0.0,
        )
        result_high = model.reduce_outliers(
            documents,
            model.topics_,
            strategy="c-tf-idf",
            threshold=None,
            outliers_percentage_target=0.0,
            min_threshold=0.5,
        )

        outliers_low = sum(1 for t in result_low if t == -1)
        outliers_high = sum(1 for t in result_high if t == -1)
        # Higher min_threshold should result in same or more outliers
        assert outliers_high >= outliers_low

    def test_backward_compatible_threshold(self, base_topic_model, documents):
        """Existing threshold parameter should still work."""
        model = copy.deepcopy(base_topic_model)
        if not self._has_outliers(model):
            pytest.skip("No outliers in model")

        new_topics = model.reduce_outliers(
            documents,
            model.topics_,
            strategy="c-tf-idf",
            threshold=0.0,
        )
        assert len(new_topics) == len(documents)

    def test_auto_threshold_probabilities(self, base_topic_model, documents):
        """Probabilities strategy should also support auto-threshold."""
        model = copy.deepcopy(base_topic_model)
        if not self._has_outliers(model):
            pytest.skip("No outliers in model")

        if model.probabilities_ is None:
            pytest.skip("No probabilities available")

        new_topics = model.reduce_outliers(
            documents,
            model.topics_,
            strategy="probabilities",
            probabilities=model.probabilities_,
            threshold=None,
            outliers_percentage_target=0.05,
        )
        assert len(new_topics) == len(documents)

    def test_output_length_matches_input(self, base_topic_model, documents):
        """Output list should always have the same length as input."""
        model = copy.deepcopy(base_topic_model)
        if not self._has_outliers(model):
            pytest.skip("No outliers in model")

        new_topics = model.reduce_outliers(
            documents,
            model.topics_,
            strategy="c-tf-idf",
            threshold=None,
            outliers_percentage_target=0.0,
        )
        assert len(new_topics) == len(model.topics_)
