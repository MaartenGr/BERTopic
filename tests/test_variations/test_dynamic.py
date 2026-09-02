import copy
import pytest


@pytest.mark.parametrize(
    "model",
    [
        ("kmeans_pca_topic_model"),
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

    assert topics_over_time["Frequency"].sum() == len(documents)
    assert set(topics_over_time["Topic"].unique().to_list()) == set(topic_model.topics_)
    assert len(topics_over_time["Timestamp"].unique()) == len(set(timestamps))


# Timestamps reach numpy's `datetime64`, which parses ISO 8601 only, so anything else has to
# be parsed from an explicit format first. The test above uses ints and never exercises this.
@pytest.mark.parametrize(
    "timestamp,datetime_format",
    [
        ("2024-01-{day:02d}", None),
        ("{day:02d}/01/2024", "%d/%m/%Y"),
        ("Jan{day:02d}", "%b%d"),
    ],
)
def test_dynamic_string_timestamps(timestamp, datetime_format, base_topic_model, documents):
    timestamps = [timestamp.format(day=(index % 27) + 1) for index in range(len(documents))]

    topics_over_time = base_topic_model.topics_over_time(
        documents, timestamps, datetime_format=datetime_format
    )

    assert topics_over_time["Frequency"].sum() == len(documents)
    assert len(topics_over_time["Timestamp"].unique()) == len(set(timestamps))
