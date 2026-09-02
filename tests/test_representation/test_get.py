import copy
import pytest
import polars as pl

from bertopic._topics import Label, StructuredJSON


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
def test_get_topic(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    topics = [topic_model.get_topic(topic) for topic in set(topic_model.topics_)]
    unknown_topic = topic_model.get_topic(500)

    for topic in topics:
        assert topic is not False

    assert len(topics) == len(topic_model.get_topic_info())
    assert not unknown_topic


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
def test_get_topics(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    topics = topic_model.get_topics()

    assert topics == topic_model.topic_representations_
    assert len(topics.keys()) == len(set(topic_model.topics_))


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
def test_get_topic_freq(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    for topic in set(topic_model.topics_):
        assert not isinstance(topic_model.get_topic_freq(topic), pl.DataFrame)

    topic_freq = topic_model.get_topic_freq()
    unique_topics = set(topic_model.topics_)
    topics_in_model = set(topic_model._topics.topic_ids())

    assert isinstance(topic_freq, pl.DataFrame)

    assert len(topic_freq) == len(set(topic_model.topics_))
    assert len(topics_in_model.difference(unique_topics)) == 0
    assert len(unique_topics.difference(topics_in_model)) == 0


@pytest.mark.parametrize(
    "model",
    [
        ("base_topic_model"),
        ("custom_topic_model"),
        ("merged_topic_model"),
        ("reduced_topic_model"),
    ],
)
def test_get_representative_docs(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    all_docs = topic_model.get_representative_docs()
    unique_topics = set(topic_model.topics_)
    topics_in_model = set(topic_model._topics.topic_ids())

    assert len(all_docs) == len(topic_model.topic_sizes_.keys())
    assert len(all_docs) == len(topics_in_model)
    assert len(all_docs) == topic_model.c_tf_idf_.shape[0]
    assert len(all_docs) == len(topic_model.topic_labels_)
    assert all([True if len(docs) == 3 else False for docs in all_docs.values()])

    topics = set(list(all_docs.keys()))

    assert len(topics.difference(unique_topics)) == 0
    assert len(topics.difference(topics_in_model)) == 0


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
def test_get_topic_info(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    info = topic_model.get_topic_info()

    if topic_model._outliers:
        assert info.row(0, named=True)["Topic"] == -1
    else:
        assert info.row(0, named=True)["Topic"] == 0

    for topic in set(topic_model.topics_):
        assert len(topic_model.get_topic_info(topic)) == 1

    assert len(topic_model.get_topic_info(200)) == 0


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
def test_get_topic_always_returns_words_with_scores(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))

    for topic in set(topic_model.topics_):
        representation = topic_model.get_topic(topic)
        assert all(isinstance(word, str) and isinstance(score, float) for word, score in representation)


def test_get_topic_flattens_a_label_representation(base_topic_model):
    """The regression this unit exists for: a label used to come back as a bare string.

    `get_topic` returned `'label_0'`, so `visualize_barchart` raised on `for word, _ in ...`
    and `_create_topic_vectors` would have embedded individual characters.
    """
    topic_model = copy.deepcopy(base_topic_model)
    topic_ids = topic_model._topics.topic_ids()
    topic_model._topics.set_data(
        representations={"Main": {topic_id: Label(data=f"label_{topic_id}") for topic_id in topic_ids}}
    )
    first = topic_ids[0]

    assert topic_model.get_topic(first) == [(f"label_{first}", 1.0)]
    assert topic_model.visualize_barchart() is not None


def test_get_representation_keeps_the_structure_get_topic_flattens(base_topic_model):
    """Flattening is lossy, so the representation itself stays reachable."""
    topic_model = copy.deepcopy(base_topic_model)
    first = topic_model._topics.topic_ids()[0]
    topic_model._topics.set_data(representations={"Main": {first: StructuredJSON(data={"topic": "cats"})}})

    representation = topic_model.get_representation(first)

    assert isinstance(representation, StructuredJSON)
    assert representation.data == {"topic": "cats"}
    assert topic_model.get_topic(first) == [("cats", 1.0)]


def test_topic_aspects_excludes_the_main_representation(representation_topic_model):
    """Aspects are the additional representations; Main is reached through `get_topic`."""
    aspects = representation_topic_model.topic_aspects_

    assert "Main" not in aspects
    assert "MMR" in aspects
