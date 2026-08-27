import copy
import pytest
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer

from bertopic._corpus import Corpus


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
def test_update_topics(model, documents, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    old_ctfidf = topic_model.c_tf_idf_
    old_topics = topic_model.topics_

    topic_model.update_topics(documents, n_gram_range=(1, 3))

    assert old_ctfidf.shape[1] < topic_model.c_tf_idf_.shape[1]
    assert old_topics == topic_model.topics_

    updated_topics = [topic if topic != 1 else 0 for topic in old_topics]
    topic_model.update_topics(documents, topics=updated_topics, n_gram_range=(1, 3))

    assert len(set(old_topics)) - 1 == len(set(topic_model.topics_))

    old_topics = topic_model.topics_
    updated_topics = [topic if topic != 2 else 0 for topic in old_topics]
    topic_model.update_topics(documents, topics=updated_topics, n_gram_range=(1, 3))

    assert len(set(old_topics)) - 1 == len(set(topic_model.topics_))


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
def test_extract_representations(model, documents, document_embeddings, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    corpus = Corpus(documents=documents, topics=np.array(topic_model.topics_), embeddings=document_embeddings)

    topic_model._extract_representations(corpus)

    assert topic_model.c_tf_idf_.shape[0] == len(set(topic_model.topics_))
    assert topic_model.c_tf_idf_.shape[1] > 100

    freq = topic_model.get_topic_freq()
    assert isinstance(freq, pl.DataFrame)
    assert len(freq["Topic"].unique()) == len(set(topic_model.topics_))
    assert len(freq["Topic"].unique()) == len(freq)


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
def test_extract_representations_custom_cv(model, documents, document_embeddings, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    corpus = Corpus(documents=documents, topics=np.array(topic_model.topics_), embeddings=document_embeddings)

    cv = CountVectorizer(ngram_range=(1, 2))
    topic_model.vectorizer_model = cv
    topic_model._extract_representations(corpus)

    assert topic_model.c_tf_idf_.shape[0] == len(set(topic_model.topics_))
    assert topic_model.c_tf_idf_.shape[1] > 100

    freq = topic_model.get_topic_freq()
    assert isinstance(freq, pl.DataFrame)
    assert len(freq["Topic"].unique()) == len(set(topic_model.topics_))
    assert len(freq["Topic"].unique()) == len(freq)


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
@pytest.mark.parametrize("reduced_topics", [2, 4, 10])
def test_topic_reduction(model, reduced_topics, documents, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    old_topics = copy.deepcopy(topic_model.topics_)
    old_freq = topic_model.get_topic_freq()

    topic_model.reduce_topics(documents, nr_topics=reduced_topics)

    new_freq = topic_model.get_topic_freq()

    if model != "online_topic_model":
        assert old_freq["Count"].sum() == new_freq["Count"].sum()
    assert len(old_freq["Topic"].unique()) == len(old_freq)
    assert len(new_freq["Topic"].unique()) == len(new_freq)
    assert len(topic_model.topics_) == len(old_topics)
    assert topic_model.topics_ != old_topics


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
def test_topic_reduction_edge_cases(model, documents, document_embeddings, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    nr_topics_before = len(set(topic_model.topics_))

    # Set nr_topics higher than existing topics — reduction should be a no-op
    topic_model.nr_topics = nr_topics_before + 100
    corpus = Corpus(documents=documents, topics=np.array(topic_model.topics_), embeddings=document_embeddings)

    corpus = topic_model._reduce_topics(corpus)

    nr_topics_after = len(set(topic_model.topics_))
    assert nr_topics_before == nr_topics_after


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
def test_find_topics(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    similar_topics, similarity = topic_model.find_topics("car")

    assert np.mean(similarity) > 0.1
    assert len(similar_topics) > 0
