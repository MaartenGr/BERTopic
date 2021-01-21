"""
Unit tests for the models underpinning BERTopic:
    * SentenceTransformers
    * UMAP
    * HDBSCAN
    * class-based TF-IDF
    * MMR

For each model, several common cases are tested to check whether they, isolated,
work as intended. This does not include whether the values in itself are correct.
For example, if embeddings are extracted from SentenceTransformer, we assume that
the embeddings themselves are of quality. However, sanity checks will be executed.
"""


import pytest
import numpy as np
import pandas as pd

from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups, make_blobs
from sentence_transformers import SentenceTransformer

from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic._mmr import mmr
from bertopic._ctfidf import ClassTFIDF


newsgroup_docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data'][:1000]
embedding_model = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")


def test_extract_embeddings(base_bertopic):
    """ Test SentenceTransformer

    Check whether the embeddings are correctly generated
    for both a single string or a list of strings. This means that
    the correct shape should be outputted. The embeddings by itself
    should not exceed certain values as a sanity check.
    """
    single_embedding = base_bertopic._extract_embeddings("a document")
    multiple_embeddings = base_bertopic._extract_embeddings(["a document", "another document"])

    assert single_embedding.shape[0] == 768
    assert np.min(single_embedding) > -5
    assert np.max(single_embedding) < 5

    assert multiple_embeddings.shape[0] == 2
    assert multiple_embeddings.shape[1] == 768
    assert np.min(multiple_embeddings) > -5
    assert np.max(multiple_embeddings) < 5


def test_extract_embeddings_compare():
    """ Test SentenceTransformer with BERTopic

    Test if the correct embedding model is loaded in BERTopic and
    whether BERTopic embeddings match the sentence-transformers embeddings.
    """
    docs = ["some document"]
    model = BERTopic(embedding_model="distilbert-base-nli-stsb-mean-tokens")
    bertopic_embeddings = model._extract_embeddings(docs)

    assert isinstance(bertopic_embeddings, np.ndarray)
    assert bertopic_embeddings.shape == (1, 768)

    sentence_embeddings = embedding_model.encode(docs, show_progress_bar=False)
    assert np.array_equal(bertopic_embeddings, sentence_embeddings)


def test_extract_incorrect_embeddings():
    """ Test if errors are raised when loading incorrect model """
    with pytest.raises(ValueError):
        model = BERTopic(language="Unknown language")
        model._extract_embeddings(["Some document"])


@pytest.mark.parametrize("embeddings,shape", [(np.random.rand(100, 68), 100),
                                              (np.random.rand(10, 768), 10),
                                              (np.random.rand(1000, 5), 1000)])
def test_umap_reduce_dimensionality(embeddings, shape):
    """ Test UMAP

    Testing whether the dimensionality across different shapes is
    reduced to the correct shape. For now, testing the shape is sufficient
    as the main goal here is to reduce the dimensionality, the quality is
    tested in the full pipeline.
    """
    model = BERTopic()
    umap_embeddings = model._reduce_dimensionality(embeddings)
    assert umap_embeddings.shape == (shape, 5)


@pytest.mark.parametrize("embeddings,shape,n_components", [(np.random.rand(100, 68), 100, 2),
                                                           (np.random.rand(10, 768), 10, 5),
                                                           (np.random.rand(1000, 5), 1000, 10)])
def test_custom_umap_reduce_dimensionality(embeddings, shape, n_components):
    """ Test Custom UMAP

    Testing whether the dimensionality is reduced to the correct shape with
    a custom UMAP model. The custom UMAP model differs in the resulting
    dimensionality and is tested across different embeddings.
    """
    model = BERTopic(umap_model=UMAP(n_components=n_components))
    umap_embeddings = model._reduce_dimensionality(embeddings)
    assert umap_embeddings.shape == (shape, n_components)


@pytest.mark.parametrize("samples,features,centers",
                         [(200, 500, 1),
                          (500, 200, 1),
                          (200, 500, 2),
                          (500, 200, 2),
                          (200, 500, 4),
                          (500, 200, 4)])
def test_hdbscan_cluster_embeddings(samples, features, centers):
    """ Test HDBSCAN

    Testing whether the clusters are correctly created and if the old and new dataframes
    are the exact same aside from the Topic column.
    """
    embeddings, _ = make_blobs(n_samples=samples, centers=centers, n_features=features, random_state=42)
    documents = [str(i + 1) for i in range(embeddings.shape[0])]
    old_df = pd.DataFrame({"Document": documents, "ID": range(len(documents)), "Topic": None})
    model = BERTopic()
    new_df, _ = model._cluster_embeddings(embeddings, old_df)

    assert len(new_df.Topic.unique()) == centers
    assert "Topic" in new_df.columns
    pd.testing.assert_frame_equal(old_df.drop("Topic", 1), new_df.drop("Topic", 1))


@pytest.mark.parametrize("samples,features,centers",
                         [(200, 500, 1),
                          (500, 200, 1),
                          (200, 500, 2),
                          (500, 200, 2),
                          (200, 500, 4),
                          (500, 200, 4)])
def test_custom_hdbscan_cluster_embeddings(samples, features, centers):
    """ Test Custom HDBSCAN

    Testing whether the clusters are correctly created using a custom HDBSCAN instance
    and if the old and new dataframes are the exact same aside from the Topic column.
    """
    embeddings, _ = make_blobs(n_samples=samples, centers=centers, n_features=features, random_state=42)
    documents = [str(i + 1) for i in range(embeddings.shape[0])]
    old_df = pd.DataFrame({"Document": documents, "ID": range(len(documents)), "Topic": None})
    hdbscan_model = HDBSCAN(min_cluster_size=10, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
    model = BERTopic(hdbscan_model=hdbscan_model)
    new_df, _ = model._cluster_embeddings(embeddings, old_df)

    assert len(new_df.Topic.unique()) == centers
    assert "Topic" in new_df.columns
    pd.testing.assert_frame_equal(old_df.drop("Topic", 1), new_df.drop("Topic", 1))
    assert model.hdbscan.metric == "euclidean"


def test_ctfidf(base_bertopic):
    """ Test c-TF-IDF

    Test whether the c-TF-IDF matrix is correctly calculated.
    This includes the general shape of the matrix as well as the
    possible values that could occupy the matrix.
    """
    nr_topics = 10
    documents = pd.DataFrame({"Document": newsgroup_docs,
                              "ID": range(len(newsgroup_docs)),
                              "Topic": np.random.randint(-1, nr_topics, len(newsgroup_docs))})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    documents = base_bertopic._preprocess_text(documents_per_topic.Document.values)
    count = base_bertopic.vectorizer.fit(documents)
    words = count.get_feature_names()
    X = count.transform(documents)
    transformer = ClassTFIDF().fit(X, n_samples=len(newsgroup_docs))
    c_tf_idf = transformer.transform(X)

    assert len(words) > 1000
    assert all([isinstance(x, str) for x in words])

    assert isinstance(X, csr_matrix)
    assert isinstance(c_tf_idf, csr_matrix)

    assert X.shape[0] == nr_topics + 1
    assert X.shape[1] == len(words)

    assert c_tf_idf.shape[0] == nr_topics + 1
    assert c_tf_idf.shape[1] == len(words)

    assert np.min(c_tf_idf) > -1
    assert np.max(c_tf_idf) < 1

    assert np.min(X) == 0


def test_ctfidf_custom_cv():
    """ Test c-TF-IDF with custom CountVectorizer

    Test whether the c-TF-IDF matrix is correctly calculated
    with a custom countvectorizer. By increasing the ngram_range, a larger
    matrix should be generated.
    """
    cv = CountVectorizer(ngram_range=(1, 3), stop_words="english")
    model = BERTopic(vectorizer_model=cv)

    nr_topics = 10
    documents = pd.DataFrame({"Document": newsgroup_docs,
                              "ID": range(len(newsgroup_docs)),
                              "Topic": np.random.randint(-1, nr_topics, len(newsgroup_docs))})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    documents = model._preprocess_text(documents_per_topic.Document.values)
    count = model.vectorizer.fit(documents)
    words = count.get_feature_names()
    X = count.transform(documents)
    transformer = ClassTFIDF().fit(X, n_samples=len(newsgroup_docs))
    c_tf_idf = transformer.transform(X)

    assert len(words) > 1000
    assert all([isinstance(x, str) for x in words])

    assert isinstance(X, csr_matrix)
    assert isinstance(c_tf_idf, csr_matrix)

    assert X.shape[0] == nr_topics + 1
    assert X.shape[1] == len(words)

    assert c_tf_idf.shape[0] == nr_topics + 1
    assert c_tf_idf.shape[1] == len(words)

    assert np.min(c_tf_idf) > -1
    assert np.max(c_tf_idf) < 1

    assert np.min(X) == 0


@pytest.mark.parametrize("words,diversity",
                         [(['stars', 'star', 'starry', 'astronaut', 'astronauts'], 0),
                          (['stars', 'spaceship', 'nasa', 'skies', 'sky'], 1)])
def test_mmr(words, diversity):
    """ Test MMR

    Testing both low and high diversity when selecing candidates.
    In the parameters, you can see that low diversity leads to very
    similar words/vectors to be selected, whereas a high diversity
    leads to a selection of candidates that, albeit similar to the input
    document, are less similar to each other.
    """
    candidates = mmr(doc_embedding=np.array([5, 5, 5, 5]).reshape(1, -1),
                     word_embeddings=np.array([[1, 1, 2, 2],
                                               [1, 2, 4, 7],
                                               [4, 4, 4, 4],
                                               [4, 4, 4, 4],
                                               [4, 4, 4, 4],
                                               [1, 1, 9, 3],
                                               [5, 3, 5, 8],
                                               [6, 6, 6, 6],
                                               [6, 6, 6, 6],
                                               [5, 8, 7, 2]]),
                     words=['space', 'nasa', 'stars', 'star', 'starry', 'spaceship',
                            'sky', 'astronaut', 'astronauts', 'skies'],
                     diversity=diversity)
    assert candidates == words
