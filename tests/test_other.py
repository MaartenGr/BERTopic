"""
Unit Tests of uncategorized functions/features

These tests are those that could not easily be categorized
into one of the other test_XXX.py files.


"""

from unittest import mock
from sklearn.datasets import fetch_20newsgroups, make_blobs
from sklearn.feature_extraction.text import TfidfVectorizer
from bertopic import BERTopic

newsgroup_docs = fetch_20newsgroups(subset='all')['data'][:1000]


def test_load_save_model():
    """ Check if the model is correctly saved """
    model = BERTopic(language="Dutch", embedding_model=None)
    model.save("test")
    loaded_model = BERTopic.load("test")
    assert type(model) == type(loaded_model)
    assert model.language == loaded_model.language
    assert model.embedding_model == loaded_model.embedding_model
    assert model.top_n_words == loaded_model.top_n_words
    assert model.n_neighbors == loaded_model.n_neighbors


@mock.patch("bertopic._bertopic.BERTopic._extract_embeddings")
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
