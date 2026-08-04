"""Tests for positional indexing in `_extract_representative_docs`.

Verifies that representative documents map back to the correct topic when the
same text appears in multiple topics, instead of matching by text membership.

Run from BERTopic repo root:
    pytest tests/test_repr_docs_indexing.py -v
"""

import pytest


def test_duplicate_text_across_topics(minimal_topic_model):
    """Documents with identical text in different topics get correct doc_ids."""
    # "shared text" appears in both topic 0 and topic 1
    docs = [
        "shared text",
        "unique topic zero content",
        "shared text",
        "unique topic one content",
    ]
    topics_list = [0, 0, 1, 1]

    model, c_tf_idf, documents, topics = minimal_topic_model(docs, topics_list)

    _repr_docs_mappings, _repr_docs, _repr_docs_indices, repr_docs_ids = model._extract_representative_docs(
        c_tf_idf, documents, topics, nr_samples=500, nr_repr_docs=3
    )

    # Verify each topic's representative doc_ids point to documents
    # that actually belong to that topic
    for topic_id, doc_ids in zip(sorted(topics.keys()), repr_docs_ids):
        for doc_id in doc_ids:
            actual_topic = documents.loc[doc_id, "Topic"]
            assert actual_topic == topic_id, (
                f"doc_id {doc_id} has topic {actual_topic} but was assigned as representative of topic {topic_id}"
            )


def test_all_identical_docs(minimal_topic_model):
    """When all docs are identical, doc_ids should still be correct per topic.

    All 3 docs per topic share the same text, so dedup collapses each topic
    down to 1 candidate; `nr_repr_docs=2` can only return that 1. Asserting
    the count pins the dedup interaction instead of looping over doc_ids
    that a `[]` return would also satisfy.
    """
    docs = ["same text"] * 6
    topics_list = [0, 0, 0, 1, 1, 1]

    model, c_tf_idf, documents, topics = minimal_topic_model(docs, topics_list)

    _repr_docs_mappings, _repr_docs, _repr_docs_indices, repr_docs_ids = model._extract_representative_docs(
        c_tf_idf, documents, topics, nr_samples=500, nr_repr_docs=2
    )

    for topic_id, doc_ids in zip(sorted(topics.keys()), repr_docs_ids):
        assert len(doc_ids) == 1, f"topic {topic_id}: expected exactly 1 doc_id after dedup, got {doc_ids}"
        for doc_id in doc_ids:
            actual_topic = documents.loc[doc_id, "Topic"]
            assert actual_topic == topic_id, (
                f"doc_id {doc_id} mapped to topic {actual_topic}, expected topic {topic_id}"
            )


def test_no_cross_topic_contamination(minimal_topic_model):
    """Representative docs for a topic should not contain docs from another topic."""
    docs = [
        "alpha beta gamma",
        "alpha beta delta",
        "epsilon zeta eta",
        "epsilon zeta theta",
    ]
    topics_list = [0, 0, 1, 1]

    model, c_tf_idf, documents, topics = minimal_topic_model(docs, topics_list)

    repr_docs_mappings, _, _, _repr_docs_ids = model._extract_representative_docs(
        c_tf_idf, documents, topics, nr_samples=500, nr_repr_docs=2
    )

    for topic_id in topics.keys():
        repr_doc_texts = repr_docs_mappings[topic_id]
        topic_doc_texts = documents.loc[documents.Topic == topic_id, "Document"].tolist()
        for doc in repr_doc_texts:
            assert doc in topic_doc_texts, (
                f"Representative doc '{doc}' for topic {topic_id} not found in that topic's documents"
            )


def test_selected_indices_variable_used(minimal_topic_model):
    """doc_ids count should match nr_repr_docs per topic."""
    docs = [
        "doc alpha one",
        "doc beta two",
        "doc gamma three",
        "doc delta four",
        "doc epsilon five",
        "doc zeta six",
    ]
    topics_list = [0, 0, 0, 1, 1, 1]

    model, c_tf_idf, documents, topics = minimal_topic_model(docs, topics_list)

    _, _, _, repr_docs_ids = model._extract_representative_docs(
        c_tf_idf, documents, topics, nr_samples=500, nr_repr_docs=2
    )

    for topic_id, doc_ids in zip(sorted(topics.keys()), repr_docs_ids):
        assert len(doc_ids) == 2, f"Topic {topic_id} should have 2 doc_ids, got {len(doc_ids)}"


def test_doc_ids_are_valid_dataframe_indices(minimal_topic_model):
    """All returned doc_ids should be valid indices into the original DataFrame."""
    docs = [
        "shared text",
        "unique topic zero content",
        "shared text",
        "unique topic one content",
        "more topic one docs",
    ]
    topics_list = [0, 0, 1, 1, 1]

    model, c_tf_idf, documents, topics = minimal_topic_model(docs, topics_list)

    _, _, _, repr_docs_ids = model._extract_representative_docs(
        c_tf_idf, documents, topics, nr_samples=500, nr_repr_docs=3
    )

    valid_indices = set(documents.index.tolist())
    for doc_ids in repr_docs_ids:
        for doc_id in doc_ids:
            assert doc_id in valid_indices, f"doc_id {doc_id} not in DataFrame index"


def test_duplicate_text_with_diversity(minimal_topic_model):
    """MMR branch should also map doc_ids correctly with duplicate text."""
    docs = [
        "machine learning algorithms applied",
        "machine learning methods used",
        "natural language processing tasks",
        "natural language understanding models",
    ]
    topics_list = [0, 0, 1, 1]

    model, c_tf_idf, documents, topics = minimal_topic_model(docs, topics_list)

    _, _, _, repr_docs_ids = model._extract_representative_docs(
        c_tf_idf,
        documents,
        topics,
        nr_samples=500,
        nr_repr_docs=2,
        diversity=0.5,
    )

    for topic_id, doc_ids in zip(sorted(topics.keys()), repr_docs_ids):
        for doc_id in doc_ids:
            actual_topic = documents.loc[doc_id, "Topic"]
            assert actual_topic == topic_id, (
                f"doc_id {doc_id} has topic {actual_topic} but assigned to topic {topic_id}"
            )

    for topic_id, doc_ids in zip(sorted(topics.keys()), repr_docs_ids):
        # With positional indexing, doc_ids count should equal nr_repr_docs
        # (or fewer if topic has fewer docs)
        assert len(doc_ids) == 2, f"Topic {topic_id} should have 2 doc_ids, got {len(doc_ids)}"


@pytest.mark.parametrize("diversity", [None, 0.5])
def test_doc_ids_are_index_labels_not_positions(minimal_topic_model, diversity):
    """`doc_ids` must be DataFrame index labels, not positions into `documents`.

    Uses a non-contiguous, shifted index and an `ID` column deliberately distinct
    from the index (mirroring the zero-shot path where `ID` is reset to
    `range(len(documents))` independently of the original index labels, see
    `_bertopic.py`'s zero-shot handling). If a regression returned positions
    instead of labels, this test would catch it even though a default
    RangeIndex-based test could not (label == position there).
    """
    docs = [
        "doc alpha one",
        "doc beta two",
        "doc gamma three",
        "doc delta four",
        "doc epsilon five",
        "doc zeta six",
    ]
    topics_list = [0, 0, 0, 1, 1, 1]
    shifted_index = [100, 101, 102, 103, 104, 105]
    # ID intentionally different from both index and position
    ids = [900, 901, 902, 903, 904, 905]

    model, c_tf_idf, documents, topics = minimal_topic_model(docs, topics_list, index=shifted_index, ids=ids)

    _, _, _, repr_docs_ids = model._extract_representative_docs(
        c_tf_idf,
        documents,
        topics,
        nr_samples=500,
        nr_repr_docs=2,
        diversity=diversity,
    )

    valid_labels = set(documents.index.tolist())
    for topic_id, doc_ids in zip(sorted(topics.keys()), repr_docs_ids):
        for doc_id in doc_ids:
            assert doc_id in valid_labels, f"doc_id {doc_id} is not a valid index label"
            assert doc_id not in range(len(docs)), (
                f"doc_id {doc_id} looks like a position (0..{len(docs) - 1}), not a shifted index label"
            )
            assert documents.loc[doc_id, "Topic"] == topic_id, (
                f"doc_id {doc_id} has topic {documents.loc[doc_id, 'Topic']}, expected {topic_id}"
            )


@pytest.mark.parametrize("diversity", [None, 0.5])
def test_unsorted_topics_keys_map_docs_to_correct_topic(minimal_topic_model, diversity):
    """`repr_docs_mappings` must attach documents to the correct topic even when
    the `topics` dict's key insertion order is not sorted.

    The extraction loop iterates `sorted(topics.keys())` (see `_bertopic.py`,
    `labels = sorted(list(topics.keys()))`), so `repr_docs`/`repr_docs_indices`
    are built in sorted order. If `repr_docs_mappings` were instead built by
    zipping against `topics.keys()` in its original (unsorted) insertion order,
    documents would be attached to the wrong topic.
    """
    docs = [
        "alpha beta gamma",
        "alpha beta delta",
        "epsilon zeta eta",
        "epsilon zeta theta",
    ]
    topics_list = [0, 0, 1, 1]

    # Reversed insertion order: sorted order is [0, 1], insertion order is [1, 0]
    model, c_tf_idf, documents, topics = minimal_topic_model(docs, topics_list, topic_order=[1, 0])
    assert list(topics.keys()) == [1, 0]

    repr_docs_mappings, _, _, _ = model._extract_representative_docs(
        c_tf_idf,
        documents,
        topics,
        nr_samples=500,
        nr_repr_docs=2,
        diversity=diversity,
    )

    for topic_id in topics.keys():
        repr_doc_texts = repr_docs_mappings[topic_id]
        topic_doc_texts = documents.loc[documents.Topic == topic_id, "Document"].tolist()
        for doc in repr_doc_texts:
            assert doc in topic_doc_texts, (
                f"Representative doc '{doc}' for topic {topic_id} not found in that topic's "
                f"documents (topics dict insertion order was {list(topics.keys())})"
            )


@pytest.mark.parametrize("diversity", [None, 0.5])
def test_mappings_agree_with_repr_docs_ids(minimal_topic_model, diversity):
    """`repr_docs_mappings[t]` texts must correspond to the same documents as
    `repr_docs_ids` for topic `t`, regardless of the `topics` dict key order.
    """
    docs = [
        "doc alpha one",
        "doc beta two",
        "doc gamma three",
        "doc delta four",
        "doc epsilon five",
        "doc zeta six",
    ]
    topics_list = [0, 0, 0, 1, 1, 1]

    for topic_order in ([0, 1], [1, 0]):
        model, c_tf_idf, documents, topics = minimal_topic_model(docs, topics_list, topic_order=topic_order)

        repr_docs_mappings, _, _, repr_docs_ids = model._extract_representative_docs(
            c_tf_idf,
            documents,
            topics,
            nr_samples=500,
            nr_repr_docs=2,
            diversity=diversity,
        )

        # repr_docs_ids is built in sorted-label order regardless of topics dict order
        for topic_id, doc_ids in zip(sorted(topics.keys()), repr_docs_ids):
            expected_texts = set(documents.loc[doc_ids, "Document"].tolist())
            actual_texts = set(repr_docs_mappings[topic_id])
            assert actual_texts == expected_texts, (
                f"topic {topic_id} (topic_order={topic_order}): mappings {actual_texts} "
                f"do not match repr_docs_ids-derived texts {expected_texts}"
            )
