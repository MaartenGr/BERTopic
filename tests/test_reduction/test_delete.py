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
def test_delete(model, request):
    topic_model: BERTopic = copy.deepcopy(request.getfixturevalue(model))
    nr_topics = len(set(topic_model.topics_))
    length_documents = len(topic_model.topics_)

    # First deletion
    topics_to_delete = [1, 2]
    topic_model.delete_topics(topics_to_delete)
    mappings = topic_model._topics.get_mappings(from_original=True)
    original_predictions = topic_model._topics._original_predictions.tolist()
    mapped_labels = [mappings[label] for label in original_predictions]

    if model == "online_topic_model" or model == "kmeans_pca_topic_model":
        assert nr_topics == len(set(topic_model.topics_)) + 1
        assert sum(topic_model._topics.frequencies().values()) == length_documents
    else:
        assert nr_topics == len(set(topic_model.topics_)) + 2
        assert sum(topic_model._topics.frequencies().values()) == length_documents

    assert mapped_labels == topic_model.topics_

    # Find two existing topics for second deletion
    remaining_topics = sorted(list(set(topic_model.topics_)))
    remaining_topics = [t for t in remaining_topics if t != -1]  # Exclude outlier topic
    topics_to_delete = remaining_topics[:2]  # Take first two remaining topics

    # Second deletion
    topic_model.delete_topics(topics_to_delete)
    mappings = topic_model._topics.get_mappings(from_original=True)
    original_predictions = topic_model._topics._original_predictions.tolist()
    mapped_labels = [mappings[label] for label in original_predictions]

    if model == "online_topic_model" or model == "kmeans_pca_topic_model":
        assert nr_topics == len(set(topic_model.topics_)) + 3
        assert sum(topic_model._topics.frequencies().values()) == length_documents
    else:
        assert nr_topics == len(set(topic_model.topics_)) + 4
        assert sum(topic_model._topics.frequencies().values()) == length_documents

    assert mapped_labels == topic_model.topics_
