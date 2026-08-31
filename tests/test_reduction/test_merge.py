import copy
import pytest

from bertopic import BERTopic


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
def test_merge(model, documents, request):
    topic_model: BERTopic = copy.deepcopy(request.getfixturevalue(model))
    nr_topics = len(set(topic_model.topics_))

    topics_to_merge = [1, 2]
    topic_model.merge_topics(documents, topics_to_merge)
    mappings = topic_model._topics.get_mappings()

    assert nr_topics == len(set(topic_model.topics_)) + 1
    assert sum(topic_model._topics.frequencies().values()) == len(documents)
    assert set(mappings.values()) <= set(topic_model._topics.topic_ids())

    topics_to_merge = [1, 2]
    topic_model.merge_topics(documents, topics_to_merge)
    mappings = topic_model._topics.get_mappings()

    assert nr_topics == len(set(topic_model.topics_)) + 2
    assert sum(topic_model._topics.frequencies().values()) == len(documents)
    assert set(mappings.values()) <= set(topic_model._topics.topic_ids())
