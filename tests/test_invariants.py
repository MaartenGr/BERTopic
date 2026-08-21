"""Structural invariants that must hold for every BERTopic pipeline variant.

These are deliberately not snapshots. Snapshotting the pipeline variant matrix is
impractical, and UMAP is not bit-reproducible across platforms and versions, so such
snapshots would be flaky and expensive to maintain. Invariants sidestep both problems:
they survive intentional changes, and a failure names the rule that broke rather than
just reporting that something moved.

Every test runs against all nine model fixtures, so a change that holds for HDBSCAN but
breaks KMeans, zero-shot, online, or supervised modelling is caught here.
"""

import copy

import numpy as np
import pytest

from bertopic import BERTopic
from tests.conftest import ALL_MODEL_FIXTURES


@pytest.mark.parametrize("model", ALL_MODEL_FIXTURES)
def test_every_document_has_exactly_one_topic(model, documents, request):
    """There is one topic assignment per document, no more and no fewer."""
    topic_model = request.getfixturevalue(model)

    assert len(topic_model.topics_) == len(documents)


@pytest.mark.parametrize("model", ALL_MODEL_FIXTURES)
def test_topic_sizes_account_for_every_document(model, documents, request):
    """Document counts across all topics add up to the size of the corpus."""
    topic_model = request.getfixturevalue(model)

    assert sum(topic_model.topic_sizes_.values()) == len(documents)


@pytest.mark.parametrize("model", ALL_MODEL_FIXTURES)
def test_assigned_topics_and_known_topics_agree(model, request):
    """Every topic that documents point at exists, and every topic holds documents."""
    topic_model = request.getfixturevalue(model)

    assert set(topic_model.topics_) == set(topic_model.topic_sizes_)


@pytest.mark.parametrize("model", ALL_MODEL_FIXTURES)
def test_every_prediction_refers_to_a_known_topic(model, request):
    """No document points at a topic that was deleted or merged away."""
    topic_model = request.getfixturevalue(model)
    known_topics = set(topic_model.topic_sizes_)

    assert all(prediction in known_topics for prediction in topic_model.topics_)


@pytest.mark.parametrize("model", ALL_MODEL_FIXTURES)
def test_topic_ids_are_contiguous(model, request):
    """Topic IDs run from 0 upwards with no gaps, preceded by -1 when outliers exist."""
    topic_model = request.getfixturevalue(model)
    topic_ids = topic_model._topics.topic_ids()

    expected_start = -1 if -1 in topic_ids else 0
    assert topic_ids == list(range(expected_start, len(topic_ids) + expected_start))


@pytest.mark.parametrize("model", ALL_MODEL_FIXTURES)
def test_topic_matrices_have_one_row_per_topic(model, request):
    """c-TF-IDF and topic embeddings stay aligned with the set of topics."""
    topic_model = request.getfixturevalue(model)
    nr_topics = len(topic_model.topic_sizes_)

    assert topic_model.c_tf_idf_.shape[0] == nr_topics
    assert topic_model.topic_embeddings_.shape[0] == nr_topics


@pytest.mark.parametrize("model", ALL_MODEL_FIXTURES)
def test_every_topic_has_a_representation(model, request):
    """No topic is left without a Main representation to describe it."""
    topic_model = request.getfixturevalue(model)

    for topic in topic_model._topics:
        assert "Main" in topic.representations


@pytest.mark.parametrize("model", ALL_MODEL_FIXTURES)
def test_probability_matrix_has_one_column_per_topic(model, request):
    """The probability matrix carries a column for every topic, outlier included.

    The width is already correct everywhere, because the mapping sizes the matrix from
    the number of distinct target topics. What is wrong today is the *content*: columns
    are shifted by one and the outlier's mass is dropped. That is specified separately
    in `tests/test_topics.py`, which is where the fix will be verified.
    """
    topic_model = request.getfixturevalue(model)
    probabilities = topic_model.probabilities_

    if probabilities is None or probabilities.ndim == 1:
        pytest.skip("This model does not produce a full probability distribution")

    assert probabilities.shape[1] == len(topic_model._topics.topic_ids())


@pytest.mark.parametrize("model", ALL_MODEL_FIXTURES)
def test_probability_rows_match_the_document_count(model, documents, request):
    """However wide the probability matrix is, it has one row per document."""
    topic_model = request.getfixturevalue(model)
    probabilities = topic_model.probabilities_

    if probabilities is None:
        pytest.skip("This model does not produce probabilities")

    assert probabilities.shape[0] == len(documents)


@pytest.mark.parametrize("model", ALL_MODEL_FIXTURES)
def test_save_and_load_preserves_the_invariants(model, tmp_path, request):
    """A round-trip through disk changes nothing structural.

    Pickle is used rather than the safetensors default because it is the format that
    is meant to be lossless; what safetensors deliberately omits is a separate concern.
    """
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    path = tmp_path / "model"

    topic_model.save(str(path), serialization="pickle")
    loaded = BERTopic.load(str(path))

    assert loaded.topics_ == topic_model.topics_
    assert loaded.topic_sizes_ == topic_model.topic_sizes_
    assert loaded._topics.topic_ids() == topic_model._topics.topic_ids()
    assert loaded.topic_labels_ == topic_model.topic_labels_
    assert np.array_equal(loaded.c_tf_idf_.toarray(), topic_model.c_tf_idf_.toarray())


# The models that `test_representation/test_representations.py::test_topic_reduction`
# reduces down to 10 topics
REDUCIBLE_MODEL_FIXTURES = [
    "base_topic_model",
    "kmeans_pca_topic_model",
    "custom_topic_model",
    "merged_topic_model",
    "reduced_topic_model",
    "online_topic_model",
]


@pytest.mark.parametrize("model", REDUCIBLE_MODEL_FIXTURES)
def test_reduction_fixtures_have_more_topics_than_they_are_reduced_to(model, request):
    """Fixtures must hold more topics than the reduction tests reduce them to.

    `test_topic_reduction` reduces to 10 and then asserts the assignments changed. If a
    fixture already holds 10 topics or fewer, `reduce_topics` takes its no-op branch and
    that test fails for a reason unrelated to the code under test. Asserting it here
    reports the actual topic count, which turns a confusing downstream failure into a
    direct statement about the fixture.
    """
    topic_model = request.getfixturevalue(model)
    nr_topics = len(topic_model.topic_sizes_)

    assert nr_topics > 10, (
        f"{model} produced only {nr_topics} topics; the reduction tests reduce to 10 "
        "and need more than that to exercise a real reduction"
    )
