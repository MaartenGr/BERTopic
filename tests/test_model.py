import pytest
import numpy as np
import pandas as pd
from unittest import mock

from sklearn.datasets import fetch_20newsgroups, make_blobs
from bertopic import BERTopic

newsgroup_docs = fetch_20newsgroups(subset='all')['data'][:1000]


def create_embeddings(docs):
    """ For mocking the _extract_embeddings function """
    if len(docs) > 1:
        blobs, _ = make_blobs(n_samples=len(docs), centers=5, n_features=768, random_state=42)
    else:
        blobs, _ = make_blobs(n_samples=len(docs), centers=1, n_features=768, random_state=42)
    return blobs


def test_extract_embeddings():
    """ Test if only correct models are loaded """
    with pytest.raises(OSError):
        model = BERTopic(bert_model='not_a_model')
        model._extract_embeddings(["Some document"])

    # model = BERTopic(bert_model='distilbert-base-nli-mean-tokens')
    # embeddings = model._extract_embeddings(["Some document"])
    #
    # assert isinstance(embeddings, np.ndarray)
    # assert embeddings.shape == (1, 768)


@pytest.mark.parametrize("embeddings,shape", [(np.random.rand(100, 68), 100),
                                              (np.random.rand(10, 768), 10),
                                              (np.random.rand(1000, 5), 1000)])
def test_reduce_dimensionality(base_bertopic, embeddings, shape):
    """ Testing whether the dimensionality is reduced to the correct shape """
    umap_embeddings = base_bertopic._reduce_dimensionality(embeddings)
    assert umap_embeddings.shape == (shape, 5)


@pytest.mark.parametrize("samples,features,centers",
                         [(200, 500, 1),
                          (500, 200, 1),
                          (200, 500, 2),
                          (500, 200, 2),
                          (200, 500, 4),
                          (500, 200, 4)])
def test_cluster_embeddings(base_bertopic, samples, features, centers):
    """ Testing whether the clusters are correctly created and if the old and new dataframes
    are the exact same aside from the Topic column """
    embeddings, _ = make_blobs(n_samples=samples, centers=centers, n_features=features, random_state=42)
    documents = [str(i + 1) for i in range(embeddings.shape[0])]
    old_df = pd.DataFrame({"Document": documents,
                           "ID": range(len(documents)),
                           "Topic": None})
    new_df, _ = base_bertopic._cluster_embeddings(embeddings, old_df)

    assert len(new_df.Topic.unique()) == centers
    assert "Topic" in new_df.columns
    pd.testing.assert_frame_equal(old_df.drop("Topic", 1), new_df.drop("Topic", 1))


def test_extract_topics(base_bertopic):
    """ Test whether the topics are correctly extracted using c-TF-IDF """
    nr_topics = 5
    documents = pd.DataFrame({"Document": newsgroup_docs,
                              "ID": range(len(newsgroup_docs)),
                              "Topic": np.random.randint(-1, nr_topics-1, len(newsgroup_docs))})
    base_bertopic._update_topic_size(documents)
    c_tf_idf = base_bertopic._extract_topics(documents, topic_reduction=False)
    freq = base_bertopic.get_topics_freq()

    assert c_tf_idf.shape[0] == 5
    assert c_tf_idf.shape[1] > 100
    assert isinstance(freq, pd.DataFrame)
    assert nr_topics == len(freq.Topic.unique())
    assert freq.Count.sum() == len(documents)
    assert len(freq.Topic.unique()) == len(freq)


@pytest.mark.parametrize("reduced_topics", [5, 10, 20, 40])
def test_topic_reduction(reduced_topics):
    """ Test whether the topics are correctly reduced """
    base_bertopic = BERTopic(bert_model='distilbert-base-nli-mean-tokens', verbose=False)
    nr_topics = reduced_topics + 2
    base_bertopic.nr_topics = reduced_topics
    old_documents = pd.DataFrame({"Document": newsgroup_docs,
                                  "ID": range(len(newsgroup_docs)),
                                  "Topic": np.random.randint(-1, nr_topics-1, len(newsgroup_docs))})
    base_bertopic._update_topic_size(old_documents)
    c_tf_idf = base_bertopic._extract_topics(old_documents.copy(), topic_reduction=True)
    old_freq = base_bertopic.get_topics_freq()

    new_documents = base_bertopic._reduce_topics(old_documents.copy(), c_tf_idf)
    new_freq = base_bertopic.get_topics_freq()

    assert old_freq.Count.sum() == new_freq.Count.sum()
    assert len(old_freq.Topic.unique()) == len(old_freq)
    assert len(new_freq.Topic.unique()) == len(new_freq)
    assert isinstance(base_bertopic.mapped_topics, dict)
    assert not set(base_bertopic.get_topics_freq().Topic).difference(set(new_documents.Topic))
    assert base_bertopic.mapped_topics


def test_topic_reduction_edge_cases(base_bertopic):
    """ Test whether the topics are not reduced if the reduced number
    of topics exceeds the actual number of topics found """

    nr_topics = 5
    base_bertopic.nr_topics = 100
    old_documents = pd.DataFrame({"Document": newsgroup_docs,
                                  "ID": range(len(newsgroup_docs)),
                                  "Topic": np.random.randint(-1, nr_topics-1, len(newsgroup_docs))})
    base_bertopic._update_topic_size(old_documents)
    c_tf_idf = base_bertopic._extract_topics(old_documents, topic_reduction=True)
    old_freq = base_bertopic.get_topics_freq()

    new_documents = base_bertopic._reduce_topics(old_documents, c_tf_idf)
    new_freq = base_bertopic.get_topics_freq()

    assert not set(old_documents.Topic).difference(set(new_documents.Topic))
    pd.testing.assert_frame_equal(old_documents, new_documents)
    pd.testing.assert_frame_equal(old_freq, new_freq)


def test_fit(base_bertopic):
    """ Test whether the fit method works as intended """
    with mock.patch("bertopic.model.BERTopic._extract_embeddings", wraps=create_embeddings) as mock_bar:
        base_bertopic.fit(newsgroup_docs)

        all_topics = base_bertopic.get_topics()
        topic_zero = base_bertopic.get_topic(0)

        prediction, probabilities = base_bertopic.transform(["This is a new document to predict"])

        assert isinstance(topic_zero, list)
        assert len(topic_zero) > 0
        assert isinstance(topic_zero[0], tuple)
        assert isinstance(topic_zero[0][0], str)
        assert isinstance(topic_zero[0][1], float)

        assert all_topics
        assert isinstance(all_topics, dict)
        assert all_topics.get(0)
        assert len(all_topics[0]) == base_bertopic.top_n_words

        assert isinstance(prediction, np.ndarray)
        assert len(prediction) == 1
        assert len(probabilities) == len(all_topics)


@mock.patch("bertopic.model.BERTopic._extract_embeddings")
def test_fit_transform(embeddings, base_bertopic):
    """ Test whether predictions are correctly made """
    blobs, _ = make_blobs(n_samples=len(newsgroup_docs), centers=5, n_features=768, random_state=42)
    embeddings.return_value = blobs
    predictions, probabilities = base_bertopic.fit_transform(newsgroup_docs)

    assert isinstance(predictions, list)
    assert len(predictions) == len(newsgroup_docs)
    assert not set(predictions).difference(set(base_bertopic.get_topics().keys()))
    assert probabilities.shape[0] == len(newsgroup_docs)


def test_load_model(base_bertopic):
    """ Check if the model is correctly saved
    TODO: Should check whether the class variables are equal
    """
    base_bertopic.save("test")
    loaded_bertopic = BERTopic.load("test")
    assert type(base_bertopic) == type(loaded_bertopic)
