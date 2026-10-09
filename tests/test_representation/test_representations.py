import copy
import pytest
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from bertopic._corpus import Corpus
from bertopic.representation._mmr import mmr


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


def test_update_topics_moves_topic_embeddings(kmeans_pca_topic_model, documents):
    topic_model = copy.deepcopy(kmeans_pca_topic_model)
    embeddings, sizes = topic_model.topic_embeddings_, topic_model.topic_sizes_

    # Merge topic 1 into topic 0
    topics = [0 if topic == 1 else topic for topic in topic_model.topics_]
    topic_model.update_topics(documents, topics=topics)

    # Topic 0's embedding is the size-weighted mean of both, as merge_topics gives
    expected = np.average(embeddings[:2], axis=0, weights=[sizes[0], sizes[1]])
    assert topic_model.topic_embeddings_.shape == (len(embeddings) - 1, embeddings.shape[1])
    assert np.allclose(topic_model.topic_embeddings_[0], expected)


def test_update_topics_with_embeddings(kmeans_pca_topic_model, documents, document_embeddings):
    topic_model = copy.deepcopy(kmeans_pca_topic_model)
    topics = [0 if topic == 1 else topic for topic in topic_model.topics_]
    topic_model.update_topics(documents, topics=topics, embeddings=document_embeddings)

    # With the documents' own embeddings, topic 0's embedding is their centroid
    expected = document_embeddings[np.array(topics) == 0].mean(axis=0)
    assert np.allclose(topic_model.topic_embeddings_[0], expected)


def test_update_topics_moves_documents_between_topics(kmeans_pca_topic_model, documents):
    topic_model = copy.deepcopy(kmeans_pca_topic_model)

    # Move five documents from topic 0 to topic 1, keeping the same set of topics
    moved = [index for index, topic in enumerate(topic_model.topics_) if topic == 0][:5]
    topics = [1 if index in moved else topic for index, topic in enumerate(topic_model.topics_)]
    topic_model.update_topics(documents, topics=topics)

    assert topic_model.topics_ == topics
    assert topic_model.topic_sizes_ == {topic: topics.count(topic) for topic in set(topics)}


def test_update_topics_with_more_documents(base_topic_model, documents):
    # As when the rest of a corpus was predicted with transform after fitting on a sample
    topic_model = copy.deepcopy(base_topic_model)
    topics = topic_model.topics_ + topic_model.topics_[:20]
    topic_model.update_topics(documents + documents[:20], topics=topics)

    assert topic_model.topics_ == topics
    assert len(topic_model.get_document_info(documents + documents[:20])) == len(topics)


def test_update_topics_keeps_the_models_own(representation_topic_model, documents):
    topic_model = copy.deepcopy(representation_topic_model)
    before = (topic_model.vectorizer_model, topic_model.ctfidf_model, topic_model.representation_model)
    topic_model.update_topics(documents)
    after = (topic_model.vectorizer_model, topic_model.ctfidf_model, topic_model.representation_model)

    assert after == before


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
    assert isinstance(freq, pd.DataFrame)
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
    assert isinstance(freq, pd.DataFrame)
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


def test_find_topics_with_a_list(base_topic_model):
    # One term in a list finds what the term finds, and several terms are averaged into one query
    assert base_topic_model.find_topics(["car"]) == base_topic_model.find_topics("car")
    topics, _ = base_topic_model.find_topics(["car", "engine"])
    assert len(topics) == 5


def test_mmr_with_fewer_words_than_top_n():
    words = ["space", "nasa", "orbit"]
    selected = mmr(np.ones((1, 3)), np.eye(3), words, top_n=10)
    assert sorted(selected) == sorted(words)
