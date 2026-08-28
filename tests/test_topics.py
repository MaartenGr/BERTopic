"""Specification for `bertopic._topics`: topic ID and probability mapping.

These tests deliberately avoid the BERTopic pipeline. They build `Topics` objects
directly so the mapping layer can be pinned down exactly and run in milliseconds,
which makes them the fast inner loop while the prediction store is rewritten.

The convention being specified is:

    Column `j` of a probability matrix always corresponds to topic `topic_ids()[j]`.

So the matrix is always `(nr_documents, len(topic_ids()))` and the outlier column
exists exactly when the outlier topic does. Producers normalise at their own
boundary: HDBSCAN derives its outlier column as `1 - sum(row)`, while a cluster
model that represents outliers natively passes its column straight through.
Nothing downstream needs to know which of the two happened.

Tests marked `xfail(strict=True)` describe behaviour that is not correct yet. The
strict marker means the suite fails if they start passing silently, so the unit
that fixes them has to remove the marker deliberately.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from bertopic._corpus import Corpus
from bertopic._topics import Images, Keywords, Topic, TopicHierarchy, TopicMapping, Topics, TopicType


def build_topics(counts: dict[int, int], probabilities: np.ndarray | None = None) -> Topics:
    """Build a Topics collection with known IDs, document counts, and data matrices.

    Each topic is given an embedding and c-TF-IDF row filled with `topic_id + 10`, so
    that after a remapping it is obvious which original topic a given topic came from.
    The offset keeps every value non-zero, which matters because sparse rows that are
    entirely zero are not round-tripped faithfully.

    Arguments:
        counts: How many documents to assign to each topic ID.
        probabilities: Optional matrix whose column `j` corresponds to `sorted(counts)[j]`.

    Returns:
        A Topics collection ready to be remapped, merged, or deleted from.
    """
    predictions = []
    for topic_id, count in counts.items():
        predictions.extend([topic_id] * count)

    topics = Topics().initialize(predictions)
    sorted_ids = sorted(topics.topic_ids())

    topics.set_data(
        embeddings=np.array([[topic_id + 10.0] * 3 for topic_id in sorted_ids]),
        c_tf_idf=csr_matrix(np.array([[topic_id + 10.0] * 4 for topic_id in sorted_ids])),
    )

    if probabilities is not None:
        topics._original_probabilities = probabilities

    return topics


def make_probabilities(mass_per_topic: dict[int, float], nr_documents: int) -> np.ndarray:
    """Build a probability matrix following the column-per-topic convention.

    Every document is given the same distribution so that a remapping can be verified
    by reading a single row.

    Arguments:
        mass_per_topic: Probability mass per topic ID. Column `j` holds `sorted(keys)[j]`.
        nr_documents: The number of identical rows to create.
    """
    columns = [mass_per_topic[topic_id] for topic_id in sorted(mass_per_topic)]
    return np.array([columns] * nr_documents)


# --------------------------------------------------------------------------------------
# Reordering by frequency
# --------------------------------------------------------------------------------------


def test_sort_by_frequency_orders_topics_by_document_count():
    """The most frequent topic becomes topic 0, the next becomes topic 1, and so on."""
    topics = build_topics({-1: 5, 0: 2, 1: 8, 2: 4})
    topics.sort_by_frequency()

    assert topics.frequencies() == {-1: 5, 0: 8, 1: 4, 2: 2}


def test_sort_by_frequency_keeps_the_outlier_at_minus_one():
    """The outlier topic keeps ID -1 regardless of how many documents it holds."""
    topics = build_topics({-1: 99, 0: 2, 1: 8})
    topics.sort_by_frequency()

    assert topics.topic_ids() == [-1, 0, 1]
    assert topics[-1].nr_documents == 99
    assert topics[-1].topic_type == TopicType.OUTLIER


def test_sort_by_frequency_works_without_an_outlier():
    """Models that never produce outliers are numbered from 0 with no gap."""
    topics = build_topics({0: 2, 1: 8, 2: 4})
    topics.sort_by_frequency()

    assert topics.topic_ids() == [0, 1, 2]
    assert topics.frequencies() == {0: 8, 1: 4, 2: 2}


def test_sort_by_frequency_moves_topic_data_with_the_topic():
    """Embeddings and c-TF-IDF rows follow their topic to its new ID."""
    topics = build_topics({-1: 5, 0: 2, 1: 8, 2: 4})
    topics.sort_by_frequency()

    # Original topic 1 was the largest, so it becomes topic 0 and brings its data along
    assert list(topics[0].embedding) == [11.0, 11.0, 11.0]
    assert topics[0].c_tf_idf.toarray().tolist() == [[11.0] * 4]

    # Original topic 0 was the smallest, so it lands last
    assert list(topics[2].embedding) == [10.0, 10.0, 10.0]


def test_sort_by_frequency_remaps_predictions():
    """Document assignments are expressed in the new topic IDs."""
    topics = build_topics({-1: 2, 0: 1, 1: 3})
    topics.sort_by_frequency()

    # Original topic 1 had the most documents so it becomes 0, and original 0 becomes 1
    assert topics.predictions == [-1, -1, 1, 0, 0, 0]


def test_sort_by_frequency_records_the_cumulative_mapping():
    """The mapping records where each original topic ended up."""
    topics = build_topics({-1: 5, 0: 2, 1: 8, 2: 4})
    topics.sort_by_frequency()

    assert topics.get_mappings(from_original=True) == {-1: -1, 1: 0, 2: 1, 0: 2}


@pytest.mark.xfail(
    strict=True,
    reason="TopicMapping.map_probabilities ignores the outlier column; fixed in unit 5",
)
def test_reordering_permutes_probability_columns():
    """Reordering topics permutes the columns without losing or duplicating mass."""
    probabilities = make_probabilities({-1: 0.1, 0: 0.2, 1: 0.6, 2: 0.1}, nr_documents=19)
    topics = build_topics({-1: 5, 0: 2, 1: 8, 2: 4}, probabilities)
    topics.sort_by_frequency()

    # Original topic 1 becomes 0, original 2 becomes 1, original 0 becomes 2
    assert topics.probabilities[0].tolist() == pytest.approx([0.1, 0.6, 0.1, 0.2])


@pytest.mark.xfail(
    strict=True,
    reason="TopicMapping.map_probabilities drops the outlier column; fixed in unit 5",
)
def test_reordering_preserves_total_probability_mass():
    """A permutation cannot change how much mass a document carries."""
    probabilities = make_probabilities({-1: 0.1, 0: 0.2, 1: 0.6, 2: 0.1}, nr_documents=19)
    topics = build_topics({-1: 5, 0: 2, 1: 8, 2: 4}, probabilities)
    topics.sort_by_frequency()

    assert topics.probabilities[0].sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# Probability matrix shape
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "counts",
    [
        {-1: 5, 0: 2, 1: 8, 2: 4},
        {0: 2, 1: 8, 2: 4},
        {-1: 5, 0: 2},
    ],
    ids=["with_outlier", "without_outlier", "single_topic"],
)
def test_probability_matrix_has_one_column_per_topic(counts):
    """The matrix carries a column for every topic, outlier included when present."""
    probabilities = make_probabilities(
        {topic_id: 0.25 for topic_id in counts}, nr_documents=sum(counts.values())
    )
    topics = build_topics(counts, probabilities)
    topics.sort_by_frequency()

    assert topics.probabilities.shape[1] == len(topics.topic_ids())


def test_one_dimensional_probabilities_are_returned_unchanged():
    """With `calculate_probabilities=False` there is one value per document, not a matrix."""
    topics = build_topics({-1: 2, 0: 3}, np.array([0.4, 0.5, 0.6, 0.7, 0.8]))
    topics.sort_by_frequency()

    assert topics.probabilities.tolist() == pytest.approx([0.4, 0.5, 0.6, 0.7, 0.8])


# --------------------------------------------------------------------------------------
# Zero-shot ordering
# --------------------------------------------------------------------------------------


def test_zeroshot_topics_are_placed_before_clustered_topics():
    """Zero-shot topics take IDs 0..n in their original order, clustered topics follow."""
    predictions = [0] * 1 + [1] * 5 + [2] * 2 + [3] * 9 + [4] * 5
    topics = Topics().initialize(predictions, zeroshot_labels=["alpha", "beta"])
    topics.sort_by_frequency()

    # Zero-shot topics keep their order even though topic 0 holds the fewest documents,
    # while the clustered topics 2, 3 and 4 are sorted by frequency behind them
    assert topics.get_mappings(from_original=True) == {0: 0, 1: 1, 3: 2, 4: 3, 2: 4}
    assert topics.labels[0] == "alpha"
    assert topics.labels[1] == "beta"


def test_zeroshot_only_model_keeps_every_topic_in_label_order():
    """With no clustered topics the zero-shot order is the final order."""
    topics = Topics().initialize([0] * 1 + [1] * 7, zeroshot_labels=["alpha", "beta"])
    topics.sort_by_frequency()

    assert topics.get_mappings(from_original=True) == {0: 0, 1: 1}
    assert [topics[topic_id].topic_type for topic_id in topics.topic_ids()] == [
        TopicType.ZERO_SHOT,
        TopicType.ZERO_SHOT,
    ]


def test_clustered_only_model_is_sorted_purely_by_frequency():
    """Without zero-shot labels every topic is ordered by document count."""
    topics = Topics().initialize([0] * 1 + [1] * 7 + [2] * 3)
    topics.sort_by_frequency()

    assert topics.get_mappings(from_original=True) == {1: 0, 2: 1, 0: 2}


def test_zeroshot_topics_survive_alongside_an_outlier():
    """The outlier stays at -1 and does not consume a zero-shot slot."""
    predictions = [-1] * 4 + [0] * 2 + [1] * 6 + [2] * 3
    topics = Topics().initialize(predictions, zeroshot_labels=["alpha", "beta"])
    topics.sort_by_frequency()

    assert topics.topic_ids() == [-1, 0, 1, 2]
    assert topics[-1].topic_type == TopicType.OUTLIER
    assert topics.get_mappings(from_original=True) == {-1: -1, 0: 0, 1: 1, 2: 2}


# --------------------------------------------------------------------------------------
# Merging
# --------------------------------------------------------------------------------------


def test_merge_sums_document_counts():
    """A merged topic holds every document of the topics it absorbed."""
    topics = build_topics({-1: 5, 0: 8, 1: 4, 2: 2})
    topics.merge({-1: -1, 0: 0, 1: 0, 2: 1})

    assert topics.topic_ids() == [-1, 0, 1]
    assert topics.frequencies() == {-1: 5, 0: 12, 1: 2}


def test_merge_averages_embeddings_weighted_by_document_count():
    """The merged embedding is a document-count weighted average of its parts."""
    topics = Topics().initialize([0] * 8 + [1] * 2)
    topics.set_data(embeddings=np.array([[10.0, 0.0], [0.0, 20.0]]))
    topics.merge({0: 0, 1: 0})

    # 8/10 of topic 0 plus 2/10 of topic 1
    assert topics[0].embedding.tolist() == pytest.approx([8.0, 4.0])


def test_merge_keeps_topics_that_hold_no_documents():
    """A complete mapping carries zero-document topics through a merge.

    This is what `_reduce_to_n_topics` and `_auto_reduce_topics` now guarantee by
    building their mapping from `topic_ids()` rather than from document assignments.
    """
    topics = build_topics({0: 8, 1: 4, 2: 2})
    topics[2].nr_documents = 0
    topics.merge({0: 0, 1: 1, 2: 1})

    assert topics.topic_ids() == [0, 1]
    assert topics.frequencies() == {0: 8, 1: 4}


def test_merge_composes_with_an_earlier_reordering():
    """The cumulative mapping tracks original topics through both operations."""
    topics = build_topics({-1: 5, 0: 2, 1: 8, 2: 4})
    topics.sort_by_frequency()
    topics.merge({-1: -1, 0: 0, 1: 0, 2: 1})

    # Original 1 and 2 were sorted to 0 and 1, then merged together into 0
    assert topics.get_mappings(from_original=True) == {-1: -1, 1: 0, 2: 0, 0: 1}


@pytest.mark.xfail(
    strict=True,
    reason="Corpus.map_probabilities sums but TopicMapping overwrites; fixed in unit 5",
)
def test_merge_sums_probability_columns():
    """Merging topics adds their probability mass together rather than discarding it."""
    probabilities = make_probabilities({-1: 0.1, 0: 0.2, 1: 0.6, 2: 0.1}, nr_documents=19)
    topics = build_topics({-1: 5, 0: 8, 1: 4, 2: 2}, probabilities)
    topics.merge({-1: -1, 0: 0, 1: 0, 2: 1})

    # Topics 0 and 1 merge, so their 0.2 and 0.6 combine into a single 0.8 column
    assert topics.probabilities[0].tolist() == pytest.approx([0.1, 0.8, 0.1])


def test_merge_handles_topics_with_no_documents():
    """Merging topics that hold no documents falls back to equal weighting."""
    topics = Topics().initialize([0, 1])
    topics.set_data(embeddings=np.array([[10.0, 0.0], [0.0, 20.0]]))
    topics[0].nr_documents = 0
    topics[1].nr_documents = 0
    topics.merge({0: 0, 1: 0})

    assert topics[0].embedding.tolist() == pytest.approx([5.0, 10.0])


def test_merge_handles_topics_without_embeddings():
    """Merging works even when no embeddings were ever computed."""
    topics = Topics().initialize([0] * 8 + [1] * 2)
    topics.merge({0: 0, 1: 0})

    assert topics[0].nr_documents == 10
    assert topics[0].embedding.size == 0


# --------------------------------------------------------------------------------------
# Deleting
# --------------------------------------------------------------------------------------


def test_delete_moves_documents_to_the_outlier():
    """Deleted topics hand their documents to the outlier topic."""
    topics = build_topics({-1: 5, 0: 8, 1: 4, 2: 2})
    topics.delete([2])

    assert topics.topic_ids() == [-1, 0, 1]
    assert topics[-1].nr_documents == 7
    assert set(topics.predictions) == {-1, 0, 1}


def test_delete_creates_an_outlier_topic_when_none_exists():
    """Deleting from a model without outliers introduces topic -1."""
    topics = build_topics({0: 8, 1: 4, 2: 2})
    topics.delete([2])

    assert -1 in topics.topic_ids()
    assert topics[-1].nr_documents == 2
    assert topics[-1].topic_type == TopicType.OUTLIER


def test_delete_accepts_a_single_topic_id():
    """A bare integer is treated the same as a one-element list."""
    topics = build_topics({-1: 5, 0: 8, 1: 4})
    topics.delete(1)

    assert topics.topic_ids() == [-1, 0]
    assert topics[-1].nr_documents == 9


@pytest.mark.xfail(
    strict=True,
    reason="delete_topics never touches probabilities, leaving them stale; fixed in unit 5",
)
def test_delete_sums_probability_mass_into_the_outlier():
    """A deleted topic's probability mass moves to the outlier, mirroring its documents."""
    probabilities = make_probabilities({-1: 0.1, 0: 0.2, 1: 0.6, 2: 0.1}, nr_documents=19)
    topics = build_topics({-1: 5, 0: 8, 1: 4, 2: 2}, probabilities)
    topics.delete([2])

    # Topic 2's 0.1 is added to the outlier's existing 0.1
    assert topics.probabilities[0].tolist() == pytest.approx([0.2, 0.2, 0.6])


# --------------------------------------------------------------------------------------
# TopicMapping
# --------------------------------------------------------------------------------------


def test_mapping_composes_successive_operations():
    """Applying two mappings records the original to current relationship, not the last step."""
    mapping = TopicMapping()
    mapping.apply({0: 2, 1: 0, 2: 1})
    mapping.apply({0: 0, 1: 1, 2: 0})

    # Original 0 went to 2 then to 0, original 1 went to 0 then stayed, original 2 went to 1 then 1
    assert mapping.map(0, from_original=True) == 0
    assert mapping.map(1, from_original=True) == 0
    assert mapping.map(2, from_original=True) == 1


def test_mapping_reports_the_most_recent_step_separately():
    """The recent mapping describes the last step only, which is what Corpus consumes."""
    mapping = TopicMapping()
    mapping.apply({0: 2, 1: 0, 2: 1})
    mapping.apply({0: 0, 1: 1, 2: 0})

    assert mapping.map(2, from_original=False) == 0


def test_unknown_topic_ids_map_to_themselves():
    """An ID that was never mapped passes through untouched."""
    mapping = TopicMapping()
    mapping.apply({0: 1, 1: 0})

    assert mapping.map(99, from_original=True) == 99


def test_mapping_rejects_an_incomplete_new_mapping():
    """Omitting a current topic is a caller bug, and is reported as one.

    `_reduce_to_n_topics` used to build its mapping by zipping over documents, so a topic
    holding no documents never appeared in it and the composition died on an opaque
    `KeyError`. Callers now build from `topic_ids()`; this guards that from regressing.
    """
    mapping = TopicMapping()
    mapping.apply({0: 0, 1: 1, 2: 2})

    with pytest.raises(ValueError, match=r"missing topics \[2\]"):
        mapping.apply({0: 0, 1: 1})


# --------------------------------------------------------------------------------------
# Representations
# --------------------------------------------------------------------------------------


def test_images_have_no_words_until_they_are_captioned():
    """A representation may hold a modality that has no text view yet."""
    images = Images(data="collage")

    assert images.words == []
    assert str(images) == "No captions available"


def test_captioned_images_expose_their_captions_as_words():
    """Once converted, the captions are the representation's text view."""
    images = Images(data="collage", captions=["a cat", "a dog"])

    assert images.words == ["a cat", "a dog"]


def test_images_round_trip_captions_but_not_the_images():
    """Images are regenerated by re-running the model, so only captions are stored."""
    restored = Images.from_dict(Images(data="collage", captions=["a cat"]).to_dict())

    assert restored.captions == ["a cat"]
    assert restored.data is None


def test_a_topic_carries_an_image_representation_alongside_its_keywords():
    """Visual output is an aspect like any other, not a special case."""
    topics = build_topics({0: 4})
    topics.set_data(representations={"Visual": {0: Images(data="collage", captions=["a cat"])}})

    assert topics[0].representations["Visual"].words == ["a cat"]


# --------------------------------------------------------------------------------------
# Serialisation round-trips
# --------------------------------------------------------------------------------------


def test_round_trip_preserves_topics_and_mapping():
    """A Topics collection survives serialisation without losing structure."""
    topics = build_topics({-1: 5, 0: 8, 1: 4})
    topics.set_data(representations={"Main": {-1: Keywords([("noise", 0.1)]), 0: Keywords([("car", 0.9)])}})
    topics.sort_by_frequency()

    restored = Topics.from_dict(topics.to_dict(full=True))

    assert restored.topic_ids() == topics.topic_ids()
    assert restored.frequencies() == topics.frequencies()
    assert restored.predictions == topics.predictions
    assert restored.get_mappings(from_original=True) == topics.get_mappings(from_original=True)
    assert restored[0].representations["Main"].words == topics[0].representations["Main"].words


def test_full_round_trip_preserves_data_matrices():
    """Embeddings and c-TF-IDF survive a full round-trip, which is what `copy()` relies on."""
    topics = build_topics({-1: 5, 0: 8, 1: 4})

    restored = Topics.from_dict(topics.to_dict(full=True))

    assert restored[0].embedding.tolist() == topics[0].embedding.tolist()
    assert restored[0].c_tf_idf.toarray().tolist() == topics[0].c_tf_idf.toarray().tolist()


def test_disk_round_trip_omits_data_matrices():
    """The disk format deliberately leaves out the large arrays."""
    topics = build_topics({-1: 5, 0: 8, 1: 4})

    restored = Topics.from_dict(topics.to_dict(full=False))

    assert restored.topic_ids() == topics.topic_ids()
    assert restored[0].embedding.size == 0


def test_round_trip_preserves_the_width_of_an_all_zero_c_tf_idf_row():
    """A topic whose c-TF-IDF is entirely zero keeps its column count.

    `delete()` gives the outlier topic an explicitly all-zero row, so losing the width
    here means `Topics.c_tf_idf` can no longer stack the topics after a save and load.
    """
    topics = Topics().initialize([0] * 4 + [1] * 3 + [2] * 2)
    topics.set_data(c_tf_idf=csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])))
    topics.delete([1, 2])

    restored = Topics.from_dict(topics.to_dict(full=True))

    assert restored[-1].c_tf_idf.shape == (1, 2)
    assert restored.c_tf_idf.shape == (2, 2)


def test_hierarchy_round_trip_preserves_node_data():
    """Hierarchy nodes keep their embeddings and c-TF-IDF through serialisation."""
    hierarchy = TopicHierarchy(n_leaves=1)
    hierarchy.nodes[0] = Topic(
        id=0,
        embedding=np.array([1.0, 2.0, 3.0]),
        c_tf_idf=csr_matrix(np.array([[1.0, 2.0]])),
        nr_documents=4,
    )

    restored = TopicHierarchy.from_dict(hierarchy.to_dict())

    assert restored.nodes[0].embedding.tolist() == [1.0, 2.0, 3.0]


# --------------------------------------------------------------------------------------
# Target design: a single store of document assignments
# --------------------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="Corpus holds a second copy of assignments that needs manual syncing; fixed in unit 5",
)
def test_corpus_assignments_follow_topic_mutations_without_a_manual_sync():
    """Reading assignments after a mutation must not require a separate sync step.

    Today `Topics` and `Corpus` each hold document assignments, kept in step by hand
    through `map_topics_and_probabilities` at nine call sites. Collapsing them to one
    store is what makes this test pass.
    """
    topics = build_topics({-1: 2, 0: 4, 1: 2})
    corpus = Corpus(documents=[f"document {index}" for index in range(8)])
    corpus.topics = np.array(topics.predictions)

    topics.merge({-1: -1, 0: 0, 1: 0})

    assert list(corpus.topics) == topics.predictions
