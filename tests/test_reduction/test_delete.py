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
def test_delete(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    nr_topics = len(set(topic_model.topics_))
    length_documents = len(topic_model.topics_)

    # First deletion
    topics_to_delete = [1, 2]
    topic_model.delete_topics(topics_to_delete)
    mappings = topic_model.topic_mapper_.get_mappings(list(topic_model.hdbscan_model.labels_))
    mapped_labels = [mappings[label] for label in topic_model.hdbscan_model.labels_]

    if model == "online_topic_model" or model == "kmeans_pca_topic_model":
        assert nr_topics == len(set(topic_model.topics_)) + 1
        assert topic_model.get_topic_info().Count.sum() == length_documents
    else:
        assert nr_topics == len(set(topic_model.topics_)) + 2
        assert topic_model.get_topic_info().Count.sum() == length_documents

    if model == "online_topic_model":
        assert mapped_labels == topic_model.topics_[950:]
    else:
        assert mapped_labels == topic_model.topics_

    # Find two existing topics for second deletion
    remaining_topics = sorted(list(set(topic_model.topics_)))
    remaining_topics = [t for t in remaining_topics if t != -1]  # Exclude outlier topic
    topics_to_delete = remaining_topics[:2]  # Take first two remaining topics

    # Second deletion
    topic_model.delete_topics(topics_to_delete)
    mappings = topic_model.topic_mapper_.get_mappings(list(topic_model.hdbscan_model.labels_))
    mapped_labels = [mappings[label] for label in topic_model.hdbscan_model.labels_]

    if model == "online_topic_model" or model == "kmeans_pca_topic_model":
        assert nr_topics == len(set(topic_model.topics_)) + 3
        assert topic_model.get_topic_info().Count.sum() == length_documents
    else:
        assert nr_topics == len(set(topic_model.topics_)) + 4
        assert topic_model.get_topic_info().Count.sum() == length_documents

    if model == "online_topic_model":
        assert mapped_labels == topic_model.topics_[950:]
    else:
        assert mapped_labels == topic_model.topics_


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
def test_delete_with_custom_labels(model, request):
    """Custom labels are a list, so deleting topics must remap them positionally."""
    topic_model = copy.deepcopy(request.getfixturevalue(model))

    # Label every topic after itself so misalignment is detectable
    original_topics = sorted(set(topic_model.topics_))
    topic_model.set_topic_labels([f"label of topic {topic}" for topic in original_topics])
    labels_before = dict(zip(original_topics, topic_model.custom_labels_))

    topics_to_delete = [topic for topic in original_topics if topic != -1][:2]
    topic_model.delete_topics(topics_to_delete)

    remaining_topics = sorted(set(topic_model.topics_))
    assert isinstance(topic_model.custom_labels_, list)
    assert len(topic_model.custom_labels_) == len(remaining_topics)

    # Every surviving topic keeps its own label, even though topics are renumbered
    labels_after = dict(zip(remaining_topics, topic_model.custom_labels_))
    mappings = topic_model.topic_mapper_.get_mappings(original_topics=False)
    for topic in original_topics:
        if topic in topics_to_delete or topic == -1:
            continue
        assert labels_after[mappings[topic]] == labels_before[topic]

    # A newly created outlier topic gets an empty label rather than stealing one
    if -1 not in original_topics:
        assert labels_after[-1] == ""
