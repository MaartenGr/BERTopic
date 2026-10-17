import pytest
from bertopic import BERTopic
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def zeroshot_model(embedding_model, zeroshot_topic_list, zeroshot_min_similarity):
    """A zero-shot model with fast sub-models, since clustering is not what these tests check."""
    return BERTopic(
        embedding_model=embedding_model,
        umap_model=PCA(n_components=5, random_state=42),
        hdbscan_model=KMeans(n_clusters=10, random_state=42),
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


@pytest.mark.parametrize("nr_topics", [4, "auto"])
def test_reduction_keeps_zeroshot_topics(nr_topics, documents, document_embeddings, embedding_model):
    zeroshot_topic_list = ["religion", "cars", "electronics"]
    topic_model = zeroshot_model(embedding_model, zeroshot_topic_list, zeroshot_min_similarity=0.3)
    topic_model.fit(documents, document_embeddings)
    nr_topics_before = len(topic_model.topic_sizes_)
    topic_model.reduce_topics(documents, nr_topics=nr_topics)

    # Only the clustered topics are merged, so the zero-shot topics keep their names
    assert len(topic_model.topic_sizes_) < nr_topics_before
    assert set(zeroshot_topic_list) <= set(topic_model.topic_labels_.values())


def test_reduction_keeps_zeroshot_topics_of_a_merged_model(
    kmeans_pca_topic_model, documents, document_embeddings, embedding_model
):
    zeroshot_topic_list = ["religion", "cars", "electronics"]
    topic_model = zeroshot_model(embedding_model, zeroshot_topic_list, zeroshot_min_similarity=0.3)
    topic_model.fit(documents, document_embeddings)

    # Merging adds the zero-shot topics after the first model's topics, so they no longer come first
    merged_model = BERTopic.merge_models([kmeans_pca_topic_model, topic_model], min_similarity=0.95)
    zeroshot_names = set(zeroshot_topic_list) & set(merged_model.topic_labels_.values())
    assert zeroshot_names, "the merge must add zero-shot topics"

    merged_model.reduce_topics(documents + documents, nr_topics=8)
    assert zeroshot_names <= set(merged_model.topic_labels_.values())


def test_reducing_to_the_zeroshot_topics_raises(documents, document_embeddings, embedding_model):
    topic_model = zeroshot_model(
        embedding_model, ["religion", "cars", "electronics"], zeroshot_min_similarity=0.3
    )
    topic_model.fit(documents, document_embeddings)
    with pytest.raises(ValueError, match="never merges"):
        topic_model.reduce_topics(documents, nr_topics=3)


def test_update_topics_keeps_zeroshot_names(documents, document_embeddings, embedding_model):
    zeroshot_topic_list = ["religion", "cars", "electronics"]
    topic_model = zeroshot_model(embedding_model, zeroshot_topic_list, zeroshot_min_similarity=0.3)
    topic_model.fit(documents, document_embeddings)

    # Custom topics that change the set of topics, here by making the first document an outlier
    topic_model.update_topics(documents, topics=[-1, *topic_model.topics_[1:]])
    assert set(zeroshot_topic_list) <= set(topic_model.topic_labels_.values())
