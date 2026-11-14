"""Sequences of operations on fitted models, with the topics checked after every step.

The invariants in `tests/test_invariants.py` hold for freshly fitted models. Several bugs
appeared only after more than one step, such as fitting twice, merging and then reducing,
or reducing outliers and then reducing topics. Each test here runs one such sequence on a
copy of a fitted model and calls `Topics.check()` after every step.
"""

import copy
import tempfile

import pytest
from sklearn.metrics.pairwise import cosine_similarity

from bertopic import BERTopic


def fit_again(topic_model, documents, embeddings):
    return topic_model.fit(documents, embeddings)


def learn_another_batch(topic_model, documents, embeddings):
    topic_model.partial_fit(documents[:100], embeddings[:100])

    # The batch leaves out at least one topic
    assert set(topic_model.topic_sizes_) - set(topic_model.topics_[-100:])
    return topic_model


def merge_two_topics(topic_model, documents, embeddings):
    topic_model.merge_topics(documents, [0, 1])
    return topic_model


def reduce_to_ten_topics(topic_model, documents, embeddings):
    topic_model.reduce_topics(documents, nr_topics=10)
    return topic_model


def delete_a_topic(topic_model, documents, embeddings):
    topic_model.delete_topics([1])
    return topic_model


def set_labels(topic_model, documents, embeddings):
    topic_model.set_topic_labels({0: "first", 1: "second"})
    return topic_model


def reduce_outliers(topic_model, documents, embeddings):
    topics = topic_model.reduce_outliers(
        documents, topic_model.topics_, strategy="embeddings", embeddings=embeddings
    )

    # Each outlier moves to its nearest topic, found by topic ID, or stays one when no topic is similar
    outliers = [index for index, topic in enumerate(topic_model.topics_) if topic == -1]
    topic_ids = topic_model._topics.topic_ids(outliers=False)
    topic_embeddings = [topic_model._topics[topic_id].embedding for topic_id in topic_ids]
    similarity = cosine_similarity(embeddings[outliers], topic_embeddings)
    nearest = [topic_ids[row.argmax()] if row.max() > 0 else -1 for row in similarity]
    assert [topics[index] for index in outliers] == nearest

    topic_model.update_topics(documents, topics=topics, embeddings=embeddings)
    return topic_model


def leave_a_gap(topic_model, documents, embeddings):
    # Topic 1's documents move to topic 0, so the topic IDs skip 1
    topics = [0 if topic == 1 else topic for topic in topic_model.topics_]
    topic_model.update_topics(documents, topics=topics, embeddings=embeddings)
    return topic_model


def save_and_load(topic_model, documents, embeddings):
    with tempfile.TemporaryDirectory() as directory:
        topic_model.save(directory, serialization="safetensors", save_ctfidf=True, save_embedding_model=False)
        return BERTopic.load(directory)


def merge_with_a_copy(topic_model, documents, embeddings):
    return BERTopic.merge_models([topic_model, copy.deepcopy(topic_model)])


@pytest.mark.parametrize(
    ("model", "steps"),
    [
        pytest.param("kmeans_pca_topic_model", [fit_again], id="fit twice"),
        pytest.param("base_topic_model", [merge_two_topics, reduce_to_ten_topics], id="merge, then reduce"),
        pytest.param(
            "kmeans_pca_topic_model",
            [merge_two_topics, reduce_to_ten_topics],
            id="merge, then reduce without outliers",
        ),
        pytest.param(
            "base_topic_model",
            [reduce_outliers, reduce_to_ten_topics],
            id="reduce outliers, then reduce topics",
        ),
        pytest.param(
            "base_topic_model", [delete_a_topic, merge_two_topics, delete_a_topic], id="delete, merge, delete"
        ),
        pytest.param(
            "kmeans_pca_topic_model",
            [delete_a_topic, merge_two_topics, delete_a_topic],
            id="delete, merge, delete without outliers",
        ),
        pytest.param("base_topic_model", [set_labels, reduce_to_ten_topics], id="set labels, then reduce"),
        pytest.param(
            "base_topic_model",
            [leave_a_gap, reduce_outliers],
            id="gap, then reduce outliers",
            marks=pytest.mark.xfail(
                strict=True, raises=AssertionError, reason="reduce_outliers reads a column as a topic ID"
            ),
        ),
        pytest.param(
            "base_topic_model",
            [leave_a_gap, save_and_load],
            id="gap, then save and load",
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="transform by similarity reads a column as a topic ID",
            ),
        ),
        pytest.param(
            "kmeans_pca_topic_model",
            [save_and_load, merge_two_topics, reduce_to_ten_topics],
            id="save and load, then merge and reduce",
        ),
        pytest.param(
            "base_topic_model",
            [merge_with_a_copy, delete_a_topic],
            id="merge models, then delete",
        ),
        pytest.param(
            "online_topic_model", [learn_another_batch, delete_a_topic], id="learn another batch, then delete"
        ),
        pytest.param(
            "zeroshot_topic_model",
            [reduce_to_ten_topics, delete_a_topic],
            id="zero-shot, reduce, then delete",
        ),
    ],
)
def test_a_sequence_of_operations_keeps_every_rule(model, steps, documents, document_embeddings, request):
    """After every step the topics keep their rules, and new documents go to topics that exist."""
    topic_model = copy.deepcopy(request.getfixturevalue(model))

    for step in steps:
        topic_model = step(topic_model, documents, document_embeddings)
        topic_model._topics.check()

        # A cluster model still finds outliers once every fitted outlier has a topic
        predictions, _ = topic_model.transform(documents, document_embeddings)
        assert set(predictions) <= {*topic_model.topic_sizes_, -1}
