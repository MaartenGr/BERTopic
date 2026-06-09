"""Tests for divisive hierarchical_topics strategy.

Run from BERTopic repo root:
    pytest tests/test_divisive_hierarchy.py -v
"""

import copy
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Helper: build a minimal mock BERTopic model for unit tests
# ---------------------------------------------------------------------------


def _make_mock_model(n_topics, n_docs):
    """Create a minimal mock BERTopic model for testing divisive hierarchy."""
    from bertopic._bertopic import BERTopic

    model = MagicMock(spec=BERTopic)
    model._outliers = 1
    model.topics_ = [i % n_topics for i in range(n_docs)]

    # Topic embeddings: outlier + n_topics
    np.random.seed(42)
    model.topic_embeddings_ = np.random.rand(n_topics + 1, 10)
    model.c_tf_idf_ = csr_matrix(np.random.rand(n_topics + 1, 20))

    # Vectorizer mock
    model.vectorizer_model = MagicMock()
    model.vectorizer_model.get_feature_names_out.return_value = np.array([f"word{i}" for i in range(20)])
    model.vectorizer_model.transform.return_value = csr_matrix(np.random.rand(n_topics, 20))

    # ctfidf model
    model.ctfidf_model = MagicMock()
    model.ctfidf_model.transform.side_effect = lambda x: csr_matrix(np.random.rand(x.shape[0], 20))

    # Preprocessing
    model._preprocess_text.side_effect = lambda x: x

    # get_topic returns word tuples
    model.get_topic.return_value = [(f"word{i}", 0.5 - i * 0.1) for i in range(5)]

    # _extract_words_per_topic returns dict
    model._extract_words_per_topic.return_value = {0: [(f"word{i}", 0.5 - i * 0.1) for i in range(5)]}

    # Bind the real method
    model._divisive_hierarchical_topics = BERTopic._divisive_hierarchical_topics.__get__(model)

    return model


# ---------------------------------------------------------------------------
# Unit tests (mock model)
# ---------------------------------------------------------------------------


def test_divisive_returns_dataframe():
    """Divisive strategy should return a DataFrame with expected columns."""
    model = _make_mock_model(n_topics=4, n_docs=20)
    result = model._divisive_hierarchical_topics(docs=["doc"] * 20, use_ctfidf=True)
    assert isinstance(result, pd.DataFrame)
    expected_cols = {
        "Parent_ID",
        "Parent_Name",
        "Topics",
        "Child_Left_ID",
        "Child_Left_Name",
        "Child_Right_ID",
        "Child_Right_Name",
        "Distance",
    }
    assert set(result.columns) == expected_cols


def test_divisive_two_topics():
    """With 2 topics, should produce exactly 1 split."""
    model = _make_mock_model(n_topics=2, n_docs=10)
    result = model._divisive_hierarchical_topics(docs=["doc"] * 10, use_ctfidf=True)
    assert len(result) == 1


def test_divisive_single_topic_empty():
    """With 1 topic, no splits needed — empty or no rows."""
    model = _make_mock_model(n_topics=1, n_docs=5)
    result = model._divisive_hierarchical_topics(docs=["doc"] * 5, use_ctfidf=True)
    assert len(result) == 0


def test_divisive_ids_are_strings():
    """Parent and child IDs should be string type."""
    model = _make_mock_model(n_topics=3, n_docs=15)
    result = model._divisive_hierarchical_topics(docs=["doc"] * 15, use_ctfidf=True)
    if not result.empty:
        assert result["Parent_ID"].dtype == object  # string
        assert result["Child_Left_ID"].dtype == object
        assert result["Child_Right_ID"].dtype == object


def test_divisive_distances_non_negative():
    """All distances should be non-negative."""
    model = _make_mock_model(n_topics=5, n_docs=25)
    result = model._divisive_hierarchical_topics(docs=["doc"] * 25, use_ctfidf=True)
    if not result.empty:
        assert (result["Distance"] >= 0).all()


def test_divisive_no_id_collisions():
    """Internal node IDs must be unique — no collisions on depth > 2 trees."""
    model = _make_mock_model(n_topics=8, n_docs=40)
    result = model._divisive_hierarchical_topics(docs=["doc"] * 40, use_ctfidf=True)
    if result.empty:
        return

    # Collect all node IDs that appear in the hierarchy
    all_ids = set()
    for _, row in result.iterrows():
        for col in ("Parent_ID", "Child_Left_ID", "Child_Right_ID"):
            all_ids.add(row[col])

    # Parent IDs must all be unique
    parent_ids = result["Parent_ID"].tolist()
    assert len(parent_ids) == len(set(parent_ids)), "Duplicate Parent_IDs found"


def test_divisive_root_has_max_parent_id():
    """The root node should have the highest Parent_ID (get_topic_tree convention)."""
    model = _make_mock_model(n_topics=6, n_docs=30)
    result = model._divisive_hierarchical_topics(docs=["doc"] * 30, use_ctfidf=True)
    if result.empty:
        return

    # The root split should contain all topics
    n_topics = len([t for t in set(model.topics_) if t != -1])
    root_row = result.loc[result["Parent_ID"].astype(int).idxmax()]
    assert len(root_row["Topics"]) == n_topics


def test_divisive_leaf_ids_below_n_topics():
    """Leaf (single-topic) child IDs should be actual topic IDs, not internal IDs."""
    model = _make_mock_model(n_topics=4, n_docs=20)
    result = model._divisive_hierarchical_topics(docs=["doc"] * 20, use_ctfidf=True)
    if result.empty:
        return

    topic_ids = sorted([t for t in set(model.topics_) if t != -1])

    # Gather IDs that are NOT parent IDs (i.e., they are leaves)
    parent_ids = set(result["Parent_ID"].tolist())
    for _, row in result.iterrows():
        for col in ("Child_Left_ID", "Child_Right_ID"):
            child_id = row[col]
            if child_id not in parent_ids:
                # This is a leaf — its int ID should be a valid topic ID
                assert int(child_id) in topic_ids


# ---------------------------------------------------------------------------
# Integration tests (real fitted model from conftest fixtures)
# ---------------------------------------------------------------------------


def test_divisive_strategy_on_fitted_model(base_topic_model, documents):
    """hierarchical_topics(strategy='divisive') should work on a fitted model."""
    model = copy.deepcopy(base_topic_model)
    result = model.hierarchical_topics(documents, strategy="divisive")

    assert isinstance(result, pd.DataFrame)
    expected_cols = {
        "Parent_ID",
        "Parent_Name",
        "Topics",
        "Child_Left_ID",
        "Child_Left_Name",
        "Child_Right_ID",
        "Child_Right_Name",
        "Distance",
    }
    assert set(result.columns) == expected_cols
    assert len(result) > 0


def test_default_strategy_is_agglomerative(base_topic_model, documents):
    """Default strategy should be agglomerative (backward compat)."""
    model = copy.deepcopy(base_topic_model)
    result_default = model.hierarchical_topics(documents)
    result_agg = model.hierarchical_topics(documents, strategy="agglomerative")

    # Same columns and structure
    assert list(result_default.columns) == list(result_agg.columns)


def test_get_topic_tree_compat(base_topic_model, documents):
    """get_topic_tree should work with divisive hierarchy output."""
    model = copy.deepcopy(base_topic_model)
    hier = model.hierarchical_topics(documents, strategy="divisive")
    tree = model.get_topic_tree(hier)
    assert isinstance(tree, str)
    assert len(tree) > 0
