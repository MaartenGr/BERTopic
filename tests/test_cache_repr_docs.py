"""Tests for PR05: cache representative_docs_ to avoid redundant recomputation.

Run from BERTopic repo root:
    pytest tests/test_cache_repr_docs.py -v
"""

import copy
from unittest.mock import patch

import pandas as pd
import pytest


class TestCacheReprDocs:
    """Verify that _save_representative_docs uses caching correctly."""

    def test_cache_flag_initialized_false(self):
        """_repr_docs_valid should be False after __init__."""
        from bertopic import BERTopic

        model = BERTopic()
        assert hasattr(model, "_repr_docs_valid")
        assert model._repr_docs_valid is False

    def test_cache_set_after_first_save(self, base_topic_model):
        """After first _save_representative_docs, cache flag should be True."""
        model = copy.deepcopy(base_topic_model)
        # After fit, _save_representative_docs was called
        assert model._repr_docs_valid is True
        assert model.representative_docs_

    def test_second_call_skips_recomputation(self, base_topic_model):
        """Second call to _save_representative_docs should skip recomputation."""
        model = copy.deepcopy(base_topic_model)

        # Build a documents DataFrame
        docs = ["doc"] * len(model.topics_)
        documents = pd.DataFrame({"Document": docs, "ID": range(len(docs)), "Topic": model.topics_})

        # Save the original repr docs
        original_repr_docs = dict(model.representative_docs_)

        # Patch _extract_representative_docs to track calls
        with patch.object(model, "_extract_representative_docs") as mock_extract:
            model._save_representative_docs(documents)
            mock_extract.assert_not_called()

        # repr_docs should be unchanged
        assert model.representative_docs_ == original_repr_docs

    def test_cache_invalidated_by_update_topics(self, base_topic_model, documents):
        """update_topics should invalidate the cache."""
        model = copy.deepcopy(base_topic_model)
        assert model._repr_docs_valid is True

        model.update_topics(documents)
        assert model._repr_docs_valid is False

    def test_cache_invalidated_by_merge_topics(self, base_topic_model, documents):
        """merge_topics should invalidate and recompute."""
        model = copy.deepcopy(base_topic_model)
        assert model._repr_docs_valid is True

        # Find two valid topics to merge
        valid_topics = [t for t in set(model.topics_) if t != -1]
        if len(valid_topics) >= 2:
            model.merge_topics(documents, [valid_topics[0], valid_topics[1]])
            # After merge, _save_representative_docs is called, so cache is valid again
            assert model._repr_docs_valid is True

    def test_cache_invalidated_by_reduce_topics(self, base_topic_model, documents):
        """reduce_topics should invalidate and recompute."""
        model = copy.deepcopy(base_topic_model)
        assert model._repr_docs_valid is True

        nr_topics = max(2, len(set(model.topics_)) - 2)
        model.reduce_topics(documents, nr_topics=nr_topics)
        # After reduce, _save_representative_docs is called, so cache is valid again
        assert model._repr_docs_valid is True

    def test_forced_recomputation_when_cache_invalid(self, base_topic_model):
        """When cache is invalid, _save_representative_docs should call _extract_representative_docs."""
        model = copy.deepcopy(base_topic_model)

        docs = ["doc"] * len(model.topics_)
        documents = pd.DataFrame({"Document": docs, "ID": range(len(docs)), "Topic": model.topics_})

        # Manually invalidate cache
        model._repr_docs_valid = False

        with patch.object(
            model,
            "_extract_representative_docs",
            return_value=({}, [], [], []),
        ) as mock_extract:
            model._save_representative_docs(documents)
            mock_extract.assert_called_once()

        assert model._repr_docs_valid is True

    def test_cache_invalidated_by_reduce_outliers(self, base_topic_model, documents):
        """reduce_outliers should invalidate the repr_docs cache."""
        model = copy.deepcopy(base_topic_model)
        assert model._repr_docs_valid is True

        if -1 not in model.topics_:
            pytest.skip("No outliers in model")

        new_topics = model.reduce_outliers(documents, model.topics_, threshold=0.0)
        model.update_topics(documents, topics=new_topics)
        # After update_topics, cache should be invalidated
        assert model._repr_docs_valid is False
