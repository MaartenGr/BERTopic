from bertopic import BERTopic
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def zeroshot_model(embedding_model, zeroshot_topic_list, zeroshot_min_similarity):
    """A zero-shot model with fast sub-models, since clustering is not what these tests check."""
    return BERTopic(
        embedding_model=embedding_model,
        umap_model=PCA(n_components=5, random_state=42),
        hdbscan_model=KMeans(n_clusters=5, random_state=42),
        zeroshot_topic_list=zeroshot_topic_list,
        zeroshot_min_similarity=zeroshot_min_similarity,
    )


def test_every_document_matching_a_zeroshot_topic(documents, document_embeddings, embedding_model):
    zeroshot_topic_list = ["religion", "cars", "electronics"]
    topic_model = zeroshot_model(embedding_model, zeroshot_topic_list, zeroshot_min_similarity=-1)
    topic_model.fit(documents, document_embeddings)

    assert set(topic_model.topic_labels_.values()) == set(zeroshot_topic_list)
    assert sum(topic_model.topic_sizes_.values()) == len(documents)


def test_unmatched_zeroshot_topic_keeps_the_others_named(documents, document_embeddings, embedding_model):
    zeroshot_topic_list = ["space exploration and nasa", "xyzzy quantum frobnicator glorp", "ice hockey"]
    topic_model = zeroshot_model(embedding_model, zeroshot_topic_list, zeroshot_min_similarity=0.35)
    topic_model.fit(documents, document_embeddings)

    # The middle label matches no document, which is what used to drop the label after it
    labels = set(topic_model.topic_labels_.values())
    assert "xyzzy quantum frobnicator glorp" not in labels
    assert {"space exploration and nasa", "ice hockey"} <= labels
    assert "" not in labels
