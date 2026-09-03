import copy
import importlib.util

import pytest


@pytest.mark.parametrize(
    "model",
    [
        ("kmeans_pca_topic_model"),
        ("base_topic_model"),
        ("custom_topic_model"),
        ("merged_topic_model"),
        ("reduced_topic_model"),
    ],
)
def test_documents(model, reduced_embeddings, documents, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    topics = set(topic_model.topics_)
    if -1 in topics:
        topics.remove(-1)
    fig = topic_model.visualize_documents(documents, embeddings=reduced_embeddings, hide_document_hover=True)
    fig_topics = [int(data["name"].split("_")[0]) for data in fig.to_dict()["data"][1:]]
    assert set(fig_topics) == topics


@pytest.mark.skipif(
    importlib.util.find_spec("datamapplot") is None, reason="datamapplot is only in the `datamap` extra"
)
def test_document_datamap(base_topic_model, documents, reduced_embeddings):
    """The datamap plot, which had no coverage when its pandas usage was rewritten."""
    topic_model = copy.deepcopy(base_topic_model)

    figure = topic_model.visualize_document_datamap(documents, reduced_embeddings=reduced_embeddings)

    assert figure is not None
