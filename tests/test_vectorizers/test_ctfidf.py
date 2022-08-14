
import copy
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import CTfidfTransformer

docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data'][:500]


@pytest.mark.parametrize('model', [('kmeans_pca_topic_model'),
                                   ('base_topic_model'),
                                   ('custom_topic_model'),
                                   ('merged_topic_model'),
                                   ('reduced_topic_model'),
                                   ('online_topic_model')])
def test_ctfidf(model, request):
    """ Test c-TF-IDF

    Test whether the c-TF-IDF matrix is correctly calculated.
    This includes the general shape of the matrix as well as the
    possible values that could occupy the matrix.
    """
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    topics = topic_model.topics_
    documents = pd.DataFrame({"Document": docs,
                                "ID": range(len(docs)),
                                "Topic": topics})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    documents = topic_model._preprocess_text(documents_per_topic.Document.values)
    count = topic_model.vectorizer_model.fit(documents)

    words = count.get_feature_names()
    X = count.transform(documents)
    transformer = CTfidfTransformer().fit(X)
    c_tf_idf = transformer.transform(X)

    assert len(words) > 1000
    assert all([isinstance(x, str) for x in words])

    assert isinstance(X, csr_matrix)
    assert isinstance(c_tf_idf, csr_matrix)

    assert X.shape[0] == len(set(topics))
    assert X.shape[1] == len(words)

    assert c_tf_idf.shape[0] == len(set(topics))
    assert c_tf_idf.shape[1] == len(words)

    assert np.min(X) == 0


@pytest.mark.parametrize('model', [('kmeans_pca_topic_model'),
                                   ('base_topic_model'),
                                   ('custom_topic_model'),
                                   ('merged_topic_model'),
                                   ('reduced_topic_model'),
                                   ('online_topic_model')])
def test_ctfidf_custom_cv(model, request):
    """ Test c-TF-IDF with custom CountVectorizer

    Test whether the c-TF-IDF matrix is correctly calculated
    with a custom countvectorizer. By increasing the ngram_range, a larger
    matrix should be generated.
    """
    cv = CountVectorizer(ngram_range=(1, 3), stop_words="english")
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    topic_model.vectorizer_model = cv
    topics = topic_model.topics_
    documents = pd.DataFrame({"Document": docs,
                                "ID": range(len(docs)),
                                "Topic": topics})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    documents = topic_model._preprocess_text(documents_per_topic.Document.values)
    count = topic_model.vectorizer_model.fit(documents)
    words = count.get_feature_names()
    X = count.transform(documents)
    transformer = CTfidfTransformer().fit(X)
    c_tf_idf = transformer.transform(X)

    assert len(words) > 1000
    assert all([isinstance(x, str) for x in words])

    assert isinstance(X, csr_matrix)
    assert isinstance(c_tf_idf, csr_matrix)

    assert X.shape[0] == len(set(topics))
    assert X.shape[1] == len(words)

    assert c_tf_idf.shape[0] == len(set(topics))
    assert c_tf_idf.shape[1] == len(words)

    assert np.min(X) == 0
