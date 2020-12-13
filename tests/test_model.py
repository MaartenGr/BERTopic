import pytest
import numpy as np
import pandas as pd
from unittest import mock
from sklearn.datasets import fetch_20newsgroups, make_blobs
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

from bertopic import BERTopic

newsgroup_docs = fetch_20newsgroups(subset='all')['data'][:500]
embedding_model = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")


def create_embeddings(docs):
    """ For mocking the _extract_embeddings function """
    if len(docs) > 1:
        blobs, _ = make_blobs(n_samples=len(docs), centers=5, n_features=768, random_state=42)
    else:
        blobs, _ = make_blobs(n_samples=len(docs), centers=1, n_features=768, random_state=42)
    return blobs


def test_full():
    model = BERTopic(language="english", verbose=True, n_neighbors=5, min_topic_size=5)

    # Test fit
    topics, probs = model.fit_transform(newsgroup_docs)

    for topic in set(topics):
        words = model.get_topic(topic)[:10]
        assert len(words) == 10

    for topic in model.get_topics_freq().Topic:
        words = model.get_topic(topic)[:10]
        assert len(words) == 10

    assert len(model.get_topics_freq()) > 2
    assert probs.shape == (500, len(model.get_topics_freq())-1)
    assert len(model.get_topics()) == len(model.get_topics_freq())

    # Test transform
    doc = "This is a new document to predict."
    topics_test, probs_test = model.transform([doc])

    assert len(probs_test) == len(model.get_topics_freq())-1
    assert len(topics_test) == 1


def test_load_model():
    """ Check if the model is correctly saved """
    model = BERTopic(language="Dutch", embedding_model=None, n_components=12)
    model.save("test")
    loaded_model = BERTopic.load("test")
    assert type(model) == type(loaded_model)
    assert model.language == loaded_model.language
    assert model.embedding_model == loaded_model.embedding_model
    assert model.top_n_words == loaded_model.top_n_words
    assert model.n_neighbors == loaded_model.n_neighbors
    assert model.n_components == loaded_model.n_components


def test_extract_incorrect_embeddings():
    """ Test if errors are raised when loading incorrect model """
    with pytest.raises(ValueError):
        model = BERTopic(language=None, embedding_model='not_a_model')
        model._extract_embeddings(["Some document"])

    with pytest.raises(ValueError):
        model = BERTopic(language="Unknown language")
        model._extract_embeddings(["Some document"])


def test_extract_embeddings():
    """ Test if correct model is loaded and embeddings match the sentence-transformers version """
    docs = ["some document"]
    model = BERTopic(language=None, embedding_model="distilbert-base-nli-stsb-mean-tokens")
    bertopic_embeddings = model._extract_embeddings(docs)

    assert isinstance(bertopic_embeddings, np.ndarray)
    assert bertopic_embeddings.shape == (1, 768)

    sentence_embeddings = embedding_model.encode(docs, show_progress_bar=False)
    assert np.array_equal(bertopic_embeddings, sentence_embeddings)


@pytest.mark.parametrize("embeddings,shape", [(np.random.rand(100, 68), 100),
                                              (np.random.rand(10, 768), 10),
                                              (np.random.rand(1000, 5), 1000)])
def test_reduce_dimensionality(embeddings, shape):
    """ Testing whether the dimensionality is reduced to the correct shape """
    model = BERTopic()
    umap_embeddings = model._reduce_dimensionality(embeddings)
    assert umap_embeddings.shape == (shape, 5)


@pytest.mark.parametrize("samples,features,centers",
                         [(200, 500, 1),
                          (500, 200, 1),
                          (200, 500, 2),
                          (500, 200, 2),
                          (200, 500, 4),
                          (500, 200, 4)])
def test_cluster_embeddings(samples, features, centers):
    """ Testing whether the clusters are correctly created and if the old and new dataframes
    are the exact same aside from the Topic column """
    embeddings, _ = make_blobs(n_samples=samples, centers=centers, n_features=features, random_state=42)
    documents = [str(i + 1) for i in range(embeddings.shape[0])]
    old_df = pd.DataFrame({"Document": documents,
                           "ID": range(len(documents)),
                           "Topic": None})
    model = BERTopic()
    new_df, _ = model._cluster_embeddings(embeddings, old_df)

    assert len(new_df.Topic.unique()) == centers
    assert "Topic" in new_df.columns
    pd.testing.assert_frame_equal(old_df.drop("Topic", 1), new_df.drop("Topic", 1))


def test_extract_topics():
    """ Test whether the topics are correctly extracted using c-TF-IDF """
    nr_topics = 5
    documents = pd.DataFrame({"Document": newsgroup_docs,
                              "ID": range(len(newsgroup_docs)),
                              "Topic": np.random.randint(-1, nr_topics-1, len(newsgroup_docs))})
    model = BERTopic()
    model._update_topic_size(documents)
    c_tf_idf = model._extract_topics(documents)
    freq = model.get_topics_freq()

    assert c_tf_idf.shape[0] == 5
    assert c_tf_idf.shape[1] > 100
    assert isinstance(freq, pd.DataFrame)
    assert nr_topics == len(freq.Topic.unique())
    assert freq.Count.sum() == len(documents)
    assert len(freq.Topic.unique()) == len(freq)


def test_extract_topics_custom_cv():
    """ Test whether the topics are correctly extracted using c-TF-IDF
    with custom CountVectorizer
    """
    nr_topics = 5
    documents = pd.DataFrame({"Document": newsgroup_docs,
                              "ID": range(len(newsgroup_docs)),
                              "Topic": np.random.randint(-1, nr_topics-1, len(newsgroup_docs))})

    cv = CountVectorizer(ngram_range=(1, 2))
    model = BERTopic(vectorizer=cv)
    model._update_topic_size(documents)
    c_tf_idf = model._extract_topics(documents)
    freq = model.get_topics_freq()

    assert c_tf_idf.shape[0] == 5
    assert c_tf_idf.shape[1] > 100
    assert isinstance(freq, pd.DataFrame)
    assert nr_topics == len(freq.Topic.unique())
    assert freq.Count.sum() == len(documents)
    assert len(freq.Topic.unique()) == len(freq)


@pytest.mark.parametrize("reduced_topics", [5, 10, 20, 40])
def test_topic_reduction(reduced_topics):
    """ Test whether the topics are correctly reduced """
    model = BERTopic()
    nr_topics = reduced_topics + 2
    model.nr_topics = reduced_topics
    old_documents = pd.DataFrame({"Document": newsgroup_docs,
                                  "ID": range(len(newsgroup_docs)),
                                  "Topic": np.random.randint(-1, nr_topics-1, len(newsgroup_docs))})
    model._update_topic_size(old_documents)
    c_tf_idf = model._extract_topics(old_documents.copy())
    old_freq = model.get_topics_freq()

    new_documents = model._reduce_topics(old_documents.copy(), c_tf_idf)
    new_freq = model.get_topics_freq()

    assert old_freq.Count.sum() == new_freq.Count.sum()
    assert len(old_freq.Topic.unique()) == len(old_freq)
    assert len(new_freq.Topic.unique()) == len(new_freq)
    assert isinstance(model.mapped_topics, dict)
    assert not set(model.get_topics_freq().Topic).difference(set(new_documents.Topic))
    assert model.mapped_topics


def test_topic_reduction_edge_cases():
    """ Test whether the topics are not reduced if the reduced number
    of topics exceeds the actual number of topics found """
    model = BERTopic()
    nr_topics = 5
    model.nr_topics = 100
    old_documents = pd.DataFrame({"Document": newsgroup_docs,
                                  "ID": range(len(newsgroup_docs)),
                                  "Topic": np.random.randint(-1, nr_topics-1, len(newsgroup_docs))})
    model._update_topic_size(old_documents)
    c_tf_idf = model._extract_topics(old_documents)
    old_freq = model.get_topics_freq()

    new_documents = model._reduce_topics(old_documents, c_tf_idf)
    new_freq = model.get_topics_freq()

    assert not set(old_documents.Topic).difference(set(new_documents.Topic))
    pd.testing.assert_frame_equal(old_documents, new_documents)
    pd.testing.assert_frame_equal(old_freq, new_freq)


@mock.patch("bertopic.model.BERTopic._extract_embeddings")
def test_fit_transform(embeddings):
    """ Test whether predictions are correctly made """
    blobs, _ = make_blobs(n_samples=len(newsgroup_docs), centers=5, n_features=768, random_state=42)
    embeddings.return_value = blobs
    model = BERTopic()
    predictions, probabilities = model.fit_transform(newsgroup_docs)

    assert isinstance(predictions, list)
    assert len(predictions) == len(newsgroup_docs)
    assert not set(predictions).difference(set(model.get_topics().keys()))
    assert probabilities.shape[0] == len(newsgroup_docs)
