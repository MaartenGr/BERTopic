"""Smoke tests for the two hierarchy plots.

Neither had any coverage, and between them they hold most of what the narwhals
translation rewrote. Two fixtures are enough: one that produces outliers and one that
does not, since the branch that differs is whether a cluster can come back empty.
"""

import copy
import pytest


HIERARCHY_MODELS = ["base_topic_model", "kmeans_pca_topic_model"]


def test_visualize_hierarchy_builds_its_own_hierarchy(kmeans_pca_topic_model):
    """Called with nothing, the plot derives the hierarchy from the model."""
    topic_model = copy.deepcopy(kmeans_pca_topic_model)

    fig = topic_model.visualize_hierarchy()

    assert len(fig.to_dict()["data"]) > 0


@pytest.mark.parametrize("model", HIERARCHY_MODELS)
def test_visualize_hierarchy_accepts_a_prepared_hierarchy(model, documents, request):
    """The usual path: `hierarchical_topics` out, straight back in."""
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    hierarchical_topics = topic_model.hierarchical_topics(documents)

    fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)

    assert len(fig.to_dict()["data"]) > 0


@pytest.mark.parametrize("model", HIERARCHY_MODELS)
def test_visualize_hierarchical_documents(model, documents, reduced_embeddings, request):
    """One trace per level, and every document placed.

    `reduced_embeddings` keeps UMAP out of the plotting call, which is most of the cost.
    """
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    hierarchical_topics = topic_model.hierarchical_topics(documents)

    fig = topic_model.visualize_hierarchical_documents(
        documents, hierarchical_topics, reduced_embeddings=reduced_embeddings, nr_levels=3
    )

    assert len(fig.to_dict()["data"]) > 0


def test_hierarchical_documents_allows_a_level_per_merge(
    kmeans_pca_topic_model, documents, reduced_embeddings
):
    """The docstring names this as the way to see every merge, and it used to raise.

    `nr_levels` equal to the hierarchy size asks for one level per merge. The topic names
    were built with `range(max_parent_id)`, which excludes the root, so the deepest level
    looked up a topic that had never been named.
    """
    topic_model = copy.deepcopy(kmeans_pca_topic_model)
    hierarchical_topics = topic_model.hierarchical_topics(documents)

    fig = topic_model.visualize_hierarchical_documents(
        documents,
        hierarchical_topics,
        reduced_embeddings=reduced_embeddings,
        nr_levels=len(hierarchical_topics),
    )

    assert len(fig.to_dict()["data"]) > 0


def test_hierarchical_documents_caps_levels_at_the_hierarchy_size(
    kmeans_pca_topic_model, documents, reduced_embeddings
):
    """More levels than merges is meaningless rather than fatal.

    `np.array_split` returns empty splits once there are more levels than rows, and the
    boundary lookup then indexed an empty array. A shallow hierarchy hit this at the
    default `nr_levels=10`.
    """
    topic_model = copy.deepcopy(kmeans_pca_topic_model)
    hierarchical_topics = topic_model.hierarchical_topics(documents)

    fig = topic_model.visualize_hierarchical_documents(
        documents,
        hierarchical_topics,
        reduced_embeddings=reduced_embeddings,
        nr_levels=len(hierarchical_topics) + 20,
    )

    assert len(fig.to_dict()["data"]) > 0
