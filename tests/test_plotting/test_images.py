import copy

import numpy as np
import pytest


def add_image_aspect(topic_model):
    """Add an image per topic, mimicking the output of `VisualRepresentation`."""
    topics = set(topic_model.topics_)
    images = {topic: np.random.randint(0, 255, size=(60, 90, 3), dtype=np.uint8) for topic in topics}
    topic_model.topic_aspects_["Visual_Aspect"] = images


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
def test_representative_images(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    add_image_aspect(topic_model)
    topics = set(topic_model.topics_)
    if -1 in topics:
        topics.remove(-1)

    fig = topic_model.visualize_representative_images()
    fig_dict = fig.to_dict()

    assert len(fig_dict["data"]) == len(topics)
    assert [trace["visible"] for trace in fig_dict["data"]].count(True) == 1
    assert fig_dict["data"][0]["visible"]

    for slider in fig_dict["layout"]["sliders"]:
        assert len(slider["steps"]) == len(topics)
        for step in slider["steps"]:
            assert int(step["label"].split(" ")[-1]) != -1


def test_representative_images_subset(base_topic_model):
    topic_model = copy.deepcopy(base_topic_model)
    add_image_aspect(topic_model)
    selected = sorted(set(topic_model.topics_) - {-1})[:2]

    fig = topic_model.visualize_representative_images(topics=selected)
    fig_dict = fig.to_dict()

    assert len(fig_dict["data"]) == len(selected)
    for slider in fig_dict["layout"]["sliders"]:
        assert [step["label"] for step in slider["steps"]] == [f"Topic {topic}" for topic in selected]


def test_representative_images_missing_aspect(base_topic_model):
    topic_model = copy.deepcopy(base_topic_model)

    with pytest.raises(ValueError, match="Visual_Aspect"):
        topic_model.visualize_representative_images()
