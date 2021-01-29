"""
Unit Tests of uncategorized functions/features

These tests are those that could not easily be categorized
into one of the other test_XXX.py files.


"""

from sklearn.datasets import fetch_20newsgroups
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
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
    assert model.get_params() == {'allow_st_model': True,
                                  'embedding_model': None,
                                  'hdbscan_model': HDBSCAN(min_cluster_size=10, prediction_data=True),
                                  'language': 'english',
                                  'low_memory': False,
                                  'min_topic_size': 10,
                                  'n_gram_range': (1, 1),
                                  'n_neighbors': 15,
                                  'nr_topics': None,
                                  'stop_words': None,
                                  'top_n_words': 10,
                                  'umap_model': UMAP(metric='cosine', min_dist=0.0, n_components=5),
                                  'vectorizer_model': CountVectorizer(),
                                  'verbose': False}
