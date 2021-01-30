"""
Unit Tests of uncategorized functions/features

These tests are those that could not easily be categorized
into one of the other test_XXX.py files.


"""

from sklearn.datasets import fetch_20newsgroups
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


def test_get_params():
    """ Test if parameters could be extracted """
    model = BERTopic()
    params = model.get_params()
    assert not params["embedding_model"]
    assert not params["low_memory"]
    assert not params["nr_topics"]
    assert not params["stop_words"]
    assert params["n_neighbors"] == 15
    assert params["n_gram_range"] == (1, 1)
    assert params["min_topic_size"] == 10
    assert params["language"] == 'english'
