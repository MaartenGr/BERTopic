import numpy as np
import pytest

from bertopic import BERTopic
from bertopic.vectorizers import OnlineCountVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA


CHUNK_SIZE = 100


@pytest.fixture
def chunks(documents):
    """Three small batches, mirroring the loop in the online topic modeling docs."""
    return [documents[index : index + CHUNK_SIZE] for index in range(0, 3 * CHUNK_SIZE, CHUNK_SIZE)]


def online_model(**kwargs):
    """A model whose sub-models all support `partial_fit`, as online learning requires."""
    return BERTopic(
        umap_model=IncrementalPCA(n_components=5),
        hdbscan_model=MiniBatchKMeans(n_clusters=5, random_state=0),
        vectorizer_model=OnlineCountVectorizer(stop_words="english", decay=0.01),
        **kwargs,
    )


def test_partial_fit_selects_a_default_embedding_backend(chunks):
    """The documented example passes no `embedding_model` and used to raise on the first batch."""
    topic_model = online_model()

    for chunk in chunks:
        topic_model.partial_fit(chunk)

    assert topic_model.embedding_model is not None
    assert len(topic_model.topics_) == 3 * CHUNK_SIZE


def test_partial_fit_accumulates_topics_across_batches(chunks, embedding_model):
    """Every batch is kept, not just the most recent one.

    Older versions tracked only the last batch, which is why the docs advised collecting
    `topics_` by hand and assigning them back. That assignment is no longer needed, and
    `topics_` is read-only, so this is now the only path.
    """
    topic_model = online_model(embedding_model=embedding_model)

    for index, chunk in enumerate(chunks, start=1):
        topic_model.partial_fit(chunk)
        assert len(topic_model.topics_) == index * CHUNK_SIZE

    assert sum(topic_model.topic_sizes_.values()) == 3 * CHUNK_SIZE
    assert set(topic_model.topics_) == set(topic_model.topic_sizes_)


def test_hierarchical_topics_works_after_partial_fit(chunks, embedding_model):
    """The variation the docs warned would break without the manual assignment."""
    topic_model = online_model(embedding_model=embedding_model)
    seen = [document for chunk in chunks for document in chunk]

    for chunk in chunks:
        topic_model.partial_fit(chunk)

    assert len(topic_model.hierarchical_topics(seen)) > 0


class PresetClusters:
    """Labels each batch with the topics it is handed, so a test decides which topics a batch holds."""

    def __init__(self, batches):
        self.batches = iter(batches)

    def partial_fit(self, embeddings):
        self.labels_ = np.array(next(self.batches))
        return self


def test_partial_fit_with_a_topic_missing_from_a_batch():
    words = ["apple", "banana", "cherry", "grape"]
    batches = [[0, 1, 2] * 4, [0, 2, 3] * 4]
    topic_model = BERTopic(
        umap_model=IncrementalPCA(n_components=2),
        hdbscan_model=PresetClusters(batches),
        vectorizer_model=OnlineCountVectorizer(),
    )
    rng = np.random.default_rng(42)
    for labels in batches:
        topic_model.partial_fit([words[label] for label in labels], rng.normal(size=(len(labels), 5)))

    # The second batch holds no documents of topic 1, which must not shift the topics after it, and
    # adds topic 3, which joins as a plain integer like the others
    assert [topic_model.get_topic(topic)[0][0] for topic in range(4)] == words
    assert all(type(topic_id) is int for topic_id in topic_model.topic_labels_)
