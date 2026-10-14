import copy
import pytest


@pytest.mark.parametrize(
    "model",
    [
        ("kmeans_pca_topic_model"),
        ("base_topic_model"),
        ("custom_topic_model"),
        ("merged_topic_model"),
        ("reduced_topic_model"),
        ("online_topic_model"),
    ],
)
def test_dynamic(model, documents, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    timestamps = [i % 10 for i in range(len(documents))]
    topics_over_time = topic_model.topics_over_time(documents, timestamps)
    fig = topic_model.visualize_topics_over_time(topics_over_time)

    assert len(fig.to_dict()["data"]) == len(set(topic_model.topics_)) - topic_model._outliers


def test_dynamic_with_an_aspect_as_labels(representation_topic_model, documents):
    timestamps = [index % 10 for index in range(len(documents))]
    topics_over_time = representation_topic_model.topics_over_time(documents, timestamps)
    fig = representation_topic_model.visualize_topics_over_time(
        topics_over_time, topics=[0, 1, 2], custom_labels="MMR"
    )

    assert [trace["name"].split("_")[0] for trace in fig.to_dict()["data"]] == ["0", "1", "2"]
