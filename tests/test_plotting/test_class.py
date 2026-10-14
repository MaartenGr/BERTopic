import copy


def test_visualize_topics_per_class(base_topic_model, documents):
    """One trace per selected topic, which is the last plot without any coverage."""
    topic_model = copy.deepcopy(base_topic_model)
    classes = [index % 4 for index in range(len(documents))]
    topics_per_class = topic_model.topics_per_class(documents, classes)

    fig = topic_model.visualize_topics_per_class(topics_per_class, top_n_topics=5)

    assert len(fig.to_dict()["data"]) == 5


def test_visualize_topics_per_class_with_an_aspect_as_labels(representation_topic_model, documents):
    classes = [index % 4 for index in range(len(documents))]
    topics_per_class = representation_topic_model.topics_per_class(documents, classes)
    fig = representation_topic_model.visualize_topics_per_class(
        topics_per_class, topics=[0, 1, 2], custom_labels="MMR"
    )

    assert [trace["name"].split("_")[0] for trace in fig.to_dict()["data"]] == ["0", "1", "2"]
