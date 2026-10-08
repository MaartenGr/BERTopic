import numpy as np

from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.cluster import KMeans


class RecordingReduction(BaseDimensionalityReduction):
    """Skips dimensionality reduction but remembers the labels it was fitted with."""

    def fit(self, X, y=None):
        self.y = y
        return self


def test_guided_seeds_reach_dimensionality_reduction(documents, document_embeddings, embedding_model):
    embeddings = document_embeddings.copy()
    reduction = RecordingReduction()
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=reduction,
        hdbscan_model=KMeans(n_clusters=5, random_state=42),
        seed_topic_list=[["space", "nasa", "orbit"], ["god", "jesus", "church"]],
    )
    topic_model.fit(documents, embeddings)

    # Documents near a seed topic carry its index as a label, the rest -1
    assert set(reduction.y) == {-1, 0, 1}

    # Pulling documents towards their seed topic must not change the caller's embeddings
    assert np.array_equal(embeddings, document_embeddings)
