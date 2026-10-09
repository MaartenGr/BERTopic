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
    assert set(topics_over_time["Topic"].unique()) == set(topic_model.topics_)
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


def test_evolution_tuning_changes_representations(kmeans_pca_topic_model, documents):
    timestamps = [index % 10 for index in range(len(documents))]
    tuned = kmeans_pca_topic_model.topics_over_time(documents, timestamps, evolution_tuning=True)
    untuned = kmeans_pca_topic_model.topics_over_time(documents, timestamps, evolution_tuning=False)
    assert tuned["Words"].to_list() != untuned["Words"].to_list()


def test_global_tuning_uses_each_topics_own_row(kmeans_pca_topic_model, documents):
    words = [word for word, _ in kmeans_pca_topic_model.get_topic(5)[:5]]
    assert words != [word for word, _ in kmeans_pca_topic_model.get_topic(0)[:5]]

    # A timestamp holding all of topic 5's documents and nothing else keeps topic 5's own words
    timestamps = [0 if topic == 5 else 1 for topic in kmeans_pca_topic_model.topics_]
    over_time = kmeans_pca_topic_model.topics_over_time(documents, timestamps, evolution_tuning=False)
    assert over_time[over_time["Timestamp"] == 0]["Words"].iloc[0] == ", ".join(words)


def test_topics_over_time_follows_verbose(kmeans_pca_topic_model, documents, capfd):
    kmeans_pca_topic_model.topics_over_time(documents, [index % 10 for index in range(len(documents))])
    assert "it/s" not in capfd.readouterr().err
