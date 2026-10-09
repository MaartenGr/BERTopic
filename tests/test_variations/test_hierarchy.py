import copy
import pytest
from scipy.cluster import hierarchy as sch
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


@pytest.mark.parametrize(
    "model",
    [
        ("kmeans_pca_topic_model"),
        ("custom_topic_model"),
        ("merged_topic_model"),
        ("reduced_topic_model"),
        ("online_topic_model"),
    ],
)
def test_hierarchy(model, documents, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    hierarchical_topics = topic_model.hierarchical_topics(documents)

    merged_topics = set([v for vals in hierarchical_topics["Topics"] for v in vals])

    assert len(hierarchical_topics) > 0
    assert merged_topics == set(topic_model.topics_).difference({-1})


@pytest.mark.parametrize(
    "model",
    [
        ("kmeans_pca_topic_model"),
        ("custom_topic_model"),
        ("merged_topic_model"),
        ("reduced_topic_model"),
        ("online_topic_model"),
    ],
)
def test_linkage(model, documents, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    linkage_function = lambda x: sch.linkage(x, "single", optimal_ordering=True)
    hierarchical_topics = topic_model.hierarchical_topics(documents, linkage_function=linkage_function)
    merged_topics = set([v for vals in hierarchical_topics["Topics"] for v in vals])
    tree = topic_model.get_topic_tree()

    assert len(hierarchical_topics) > 0
    assert len(tree) > 50
    assert len(tree.split("\n")) <= 2 * len(set(topic_model.topics_))
    assert merged_topics == set(topic_model.topics_).difference({-1})


@pytest.mark.parametrize(
    "model",
    [
        ("kmeans_pca_topic_model"),
        ("custom_topic_model"),
        ("merged_topic_model"),
        ("reduced_topic_model"),
        ("online_topic_model"),
    ],
)
def test_tree(model, documents, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    linkage_function = lambda x: sch.linkage(x, "single", optimal_ordering=True)
    hierarchical_topics = topic_model.hierarchical_topics(documents, linkage_function=linkage_function)
    merged_topics = set([v for vals in hierarchical_topics["Topics"] for v in vals])
    tree = topic_model.get_topic_tree()

    assert len(hierarchical_topics) > 0
    assert len(tree) > 50
    assert len(tree.split("\n")) <= 2 * len(set(topic_model.topics_))
    assert merged_topics == set(topic_model.topics_).difference({-1})


def test_hierarchy_with_identical_topics(kmeans_pca_topic_model, documents):
    topic_model = copy.deepcopy(kmeans_pca_topic_model)

    # Two topics with the same c-TF-IDF row, which 1 - cosine similarity puts a hair below zero apart
    row = csr_matrix(([0.1, 1.0], ([0, 0], [0, 1])), shape=(1, topic_model.c_tf_idf_.shape[1]))
    topic_model._topics[0].c_tf_idf = topic_model._topics[1].c_tf_idf = row
    assert 1 - cosine_similarity(topic_model.c_tf_idf_[:2])[0, 1] < 0

    hierarchical_topics = topic_model.hierarchical_topics(documents)
    assert len(hierarchical_topics) == len(topic_model.topic_sizes_) - 1
    topic_model.visualize_hierarchy()


def test_hierarchy_follows_verbose(kmeans_pca_topic_model, documents, capfd):
    kmeans_pca_topic_model.hierarchical_topics(documents)
    assert "it/s" not in capfd.readouterr().err
