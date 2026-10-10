import copy
import pytest
from bertopic import BERTopic
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd  # noqa: F401

from tests.conftest import ALL_MODEL_FIXTURES


@pytest.mark.parametrize("model", ALL_MODEL_FIXTURES)
def test_full_model(model, documents, request):
    """Tests the entire pipeline in one go. This serves as a sanity check to see if the default
    settings result in a good separation of topics.

    NOTE: This does not cover all cases but merely combines it all together
    """
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    if model == "base_topic_model":
        topic_model.save(
            "model_dir",
            serialization="pytorch",
            save_ctfidf=True,
            save_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        topic_model = BERTopic.load("model_dir")

    if model == "cuml_base_topic_model":
        assert "cuml" in str(type(topic_model.umap_model)).lower()
        assert "cuml" in str(type(topic_model.hdbscan_model)).lower()

    topics = topic_model.topics_

    for topic in set(topics):
        words = topic_model.get_topic(topic)[:10]
        assert len(words) == 10

    for topic in topic_model.get_topic_freq()["Topic"]:
        words = topic_model.get_topic(topic)[:10]
        assert len(words) == 10

    assert len(topic_model.get_topic_freq()) > 2
    assert len(topic_model.get_topics()) == len(topic_model.get_topic_freq())

    # Test extraction of document info
    document_info = topic_model.get_document_info(documents)
    assert len(document_info) == len(documents)

    # Test transform
    doc = "This is a new document to predict."
    topics_test, _probs_test = topic_model.transform([doc, doc])

    assert len(topics_test) == 2

    # Test zero-shot topic modeling
    if topic_model._is_zeroshot():
        if topic_model._outliers:
            assert set(topic_model.topic_labels_.keys()) == set(range(-1, len(topic_model.topic_labels_) - 1))
        else:
            assert set(topic_model.topic_labels_.keys()) == set(range(len(topic_model.topic_labels_)))

    # Test topics over time
    timestamps = [i % 10 for i in range(len(documents))]
    topics_over_time = topic_model.topics_over_time(documents, timestamps)

    assert topics_over_time["Frequency"].sum() == len(documents)
    assert topics_over_time["Topic"].nunique() == len(set(topics))

    # Test hierarchical topics
    topic_model.hierarchical_topics(documents)

    assert len(topic_model.hierarchy_) > 0
    assert topic_model.hierarchy_["Parent_ID"].astype(int).min() > max(topics)

    # Test creation of topic tree
    tree = topic_model.get_topic_tree(tight_layout=False)
    assert isinstance(tree, str)
    assert len(tree) > 10

    # Test find topic
    similar_topics, similarity = topic_model.find_topics("query", top_n=2)
    assert len(similar_topics) == 2
    assert len(similarity) == 2
    assert max(similarity) <= 1

    # Test topic reduction
    nr_topics = len(set(topics))
    nr_topics = 2 if nr_topics < 2 else nr_topics
    topic_model.reduce_topics(documents, nr_topics=nr_topics)

    assert len(topic_model.get_topic_freq()) == nr_topics
    assert len(topic_model.topics_) == len(topics)

    # Test update topics
    topic = topic_model.get_topic(1)[:10]
    vectorizer_model = topic_model.vectorizer_model
    topic_model.update_topics(documents, n_gram_range=(2, 2))

    updated_topic = topic_model.get_topic(1)[:10]

    topic_model.update_topics(documents, vectorizer_model=vectorizer_model)
    original_topic = topic_model.get_topic(1)[:10]

    assert topic != updated_topic
    assert topic == original_topic

    # Test updating topic labels
    topic_labels = topic_model.generate_topic_labels(
        nr_words=3, topic_prefix=False, word_length=10, separator=", "
    )
    assert len(topic_labels) == len(set(topic_model.topics_))

    # Test setting topic labels
    topic_model.set_topic_labels(topic_labels)
    assert topic_model.custom_labels_ == topic_labels

    # Test merging topics
    freq = topic_model.get_topic_freq(0)
    topics_to_merge = [0, 1]
    topic_model.merge_topics(documents, topics_to_merge)
    assert freq < topic_model.get_topic_freq(0)

    # Test reduction of outliers
    if -1 in topics:
        new_topics = topic_model.reduce_outliers(documents, topics, threshold=0.0)
        nr_outliers_topic_model = sum([1 for topic in topic_model.topics_ if topic == -1])
        nr_outliers_new_topics = sum([1 for topic in new_topics if topic == -1])

        if topic_model._outliers == 1:
            assert nr_outliers_topic_model > nr_outliers_new_topics

    # Combine models
    topic_model1 = BERTopic.load("model_dir")
    merged_model = BERTopic.merge_models([topic_model, topic_model1])

    assert len(merged_model.get_topic_info()) > len(topic_model.get_topic_info())


def test_transform_one_document_with_a_1d_embedding(kmeans_pca_topic_model, documents, document_embeddings):
    topics, _ = kmeans_pca_topic_model.transform(documents[0], document_embeddings[0])
    assert len(topics) == 1


def test_load_with_an_embedding_model_skips_the_saved_one(base_topic_model, tmp_path, monkeypatch):
    base_topic_model.save(
        tmp_path, serialization="safetensors", save_embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Record any attempt to build the saved model, raising as an unreachable Hub would
    attempts = []

    def unreachable_hub(name, *args, **kwargs):
        attempts.append(name)
        raise OSError("no connection")

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", unreachable_hub)
    loaded_model = BERTopic.load(tmp_path, embedding_model=base_topic_model.embedding_model)

    assert attempts == []
    assert loaded_model.embedding_model is base_topic_model.embedding_model


# Each method reads the documents' words, which a merged model has no fitted vectorizer for
NEEDS_A_FITTED_VECTORIZER = {
    "hierarchy": lambda model, docs: model.hierarchical_topics(docs),
    "over time": lambda model, docs: model.topics_over_time(docs, [index % 10 for index in range(len(docs))]),
    "per class": lambda model, docs: model.topics_per_class(docs, [index % 3 for index in range(len(docs))]),
    "distribution": lambda model, docs: model.approximate_distribution(docs),
    "c-tf-idf outliers": lambda model, docs: model.reduce_outliers(docs, model.topics_, strategy="c-tf-idf"),
}


@pytest.mark.parametrize("method", NEEDS_A_FITTED_VECTORIZER.values(), ids=NEEDS_A_FITTED_VECTORIZER.keys())
def test_merged_model_asks_for_update_topics(method, kmeans_pca_topic_model, custom_topic_model, documents):
    merged_model = BERTopic.merge_models([kmeans_pca_topic_model, custom_topic_model])
    with pytest.raises(ValueError, match="update_topics"):
        method(merged_model, documents + documents)


def test_merged_model_works_after_update_topics(kmeans_pca_topic_model, custom_topic_model, documents):
    merged_model = BERTopic.merge_models([kmeans_pca_topic_model, custom_topic_model])
    merged_model.update_topics(documents + documents)
    assert len(merged_model.hierarchical_topics(documents + documents)) > 0


def test_merged_model_works_with_topic_embeddings(documents, document_embeddings, embedding_model):
    # Models fitted on different documents count words over different vocabularies
    models = [
        BERTopic(
            embedding_model=embedding_model,
            umap_model=PCA(n_components=5, random_state=42),
            hdbscan_model=KMeans(n_clusters=6, random_state=42),
        ).fit(documents[part], document_embeddings[part])
        for part in (slice(0, 500), slice(500, None))
    ]
    merged_model = BERTopic.merge_models(models, min_similarity=0.9)
    assert len(merged_model.topic_sizes_) > len(models[0].topic_sizes_), "the second model must add topics"

    merged_model.visualize_topics()
    distribution, _ = merged_model.approximate_distribution(documents[:5], use_embedding_model=True)
    assert distribution.shape[0] == 5
