from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
import pytest


@pytest.fixture(scope="module")
def base_bertopic():
    model = BERTopic(bert_model='distilbert-base-nli-mean-tokens',
                     top_n_words=20,
                     nr_topics=None,
                     n_gram_range=(1, 1),
                     min_topic_size=30,
                     n_neighbors=15,
                     n_components=5,
                     verbose=False)
    return model


@pytest.fixture(scope="module")
def base_bertopic_custom_cv():
    cv = CountVectorizer(ngram_range=(1, 2))
    model = BERTopic(bert_model='distilbert-base-nli-mean-tokens',
                     top_n_words=20,
                     nr_topics=None,
                     min_topic_size=30,
                     n_neighbors=15,
                     n_components=5,
                     verbose=False,
                     vectorizer=cv)
    return model
