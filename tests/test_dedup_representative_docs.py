"""Tests for deduplicating representative documents sampling.

Verifies that `_extract_representative_docs` samples without replacement so a
topic never yields duplicate representative documents.

Run from BERTopic repo root:
    pytest tests/test_dedup_representative_docs.py -v
"""

import pytest


def test_no_duplicate_docs_per_topic(minimal_topic_model):
    """Each topic's representative docs should contain no duplicates."""
    docs = ["alpha", "beta", "gamma", "delta", "epsilon"] * 3
    topics_list = [0, 0, 0, 1, 1] * 3

    model, c_tf_idf, documents, topics = minimal_topic_model(docs, topics_list)

    repr_docs_mappings, _repr_docs, _, _ = model._extract_representative_docs(
        c_tf_idf,
        documents,
        topics,
        nr_samples=500,
        nr_repr_docs=5,
    )

    assert repr_docs_mappings
    for topic, topic_docs in repr_docs_mappings.items():
        assert len(topic_docs) == len(set(topic_docs)), (
            f"Topic {topic} has duplicate representative docs: {[d for d in topic_docs if topic_docs.count(d) > 1]}"
        )


def test_heavy_duplicates_no_duplicates_in_output(minimal_topic_model):
    """When a topic has 3 unique docs but nr_samples=500, no duplicates should appear."""
    docs = ["doc A", "doc B", "doc C"] * 2 + ["unique doc"]
    topics_list = [0, 0, 0, 1, 1, 1, 0]

    model, c_tf_idf, documents, topics = minimal_topic_model(docs, topics_list)

    repr_docs_mappings, _repr_docs, _, _ = model._extract_representative_docs(
        c_tf_idf,
        documents,
        topics,
        nr_samples=500,
        nr_repr_docs=3,
    )

    for topic, docs_list in repr_docs_mappings.items():
        assert len(docs_list) == len(set(docs_list)), f"Topic {topic} has duplicate representative docs"


def test_repr_docs_count_respects_topic_size(minimal_topic_model):
    """nr_repr_docs should be capped at the number of unique docs in the topic."""
    docs = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    topics_list = [0, 0, 1, 1, 1, 1]
    # Topic 0 has only 2 unique docs — requesting 5 should yield 2

    model, c_tf_idf, documents, topics = minimal_topic_model(docs, topics_list)

    repr_docs_mappings, _, _, _ = model._extract_representative_docs(
        c_tf_idf,
        documents,
        topics,
        nr_samples=500,
        nr_repr_docs=5,
    )

    assert len(repr_docs_mappings[0]) == 2
    assert len(repr_docs_mappings[1]) == 4


def test_repr_docs_count_with_nr_repr_docs_greater_than_topic_size(minimal_topic_model):
    """When nr_repr_docs > unique docs in a topic, return all unique docs."""
    docs = ["only one"] * 5 + ["other topic doc"] * 5
    topics_list = [0] * 5 + [1] * 5
    # Topic 0 has 1 unique doc, topic 1 has 1 unique doc

    model, c_tf_idf, documents, topics = minimal_topic_model(docs, topics_list)

    repr_docs_mappings, _, _, _ = model._extract_representative_docs(
        c_tf_idf,
        documents,
        topics,
        nr_samples=500,
        nr_repr_docs=10,
    )

    for topic, docs_list in repr_docs_mappings.items():
        assert len(docs_list) == 1


def test_nr_samples_caps_candidates_per_topic(minimal_topic_model):
    """`nr_samples` must cap the candidate pool per topic, independently of topic size.

    The cap is what makes this an *approximate* search: only `nr_samples` documents
    per topic are scored. It is enforced by taking each topic's first `nr_samples`
    rows from a globally shuffled frame, so - unlike the explicit `min(nr_samples,
    len(group))` it replaced - nothing in the expression names the cap. With
    `nr_samples=2` only 2 documents per topic are scored, so at most 2 can come back
    even though `nr_repr_docs=5` and each topic holds 6 unique documents. Without a
    cap this returns 5.
    """
    docs = [f"topic zero doc {i}" for i in range(6)] + [f"topic one doc {i}" for i in range(6)]
    topics_list = [0] * 6 + [1] * 6

    model, c_tf_idf, documents, topics = minimal_topic_model(docs, topics_list)

    repr_docs_mappings, _, _, repr_docs_ids = model._extract_representative_docs(
        c_tf_idf,
        documents,
        topics,
        nr_samples=2,
        nr_repr_docs=5,
    )

    assert len(repr_docs_mappings[0]) == 2
    assert len(repr_docs_mappings[1]) == 2

    # Shuffling the frame before grouping must not leak documents across topics:
    # every returned id has to belong to the topic it is reported under.
    for topic, doc_ids in zip(sorted(topics.keys()), repr_docs_ids):
        assert set(documents.loc[doc_ids, "Topic"]) == {topic}


def test_with_diversity_no_duplicates(minimal_topic_model):
    """MMR branch (diversity > 0) should also produce no duplicates."""
    docs = [
        "machine learning algorithms",
        "deep learning neural networks",
        "natural language processing",
        "computer vision image analysis",
        "data mining techniques",
        "statistical modeling methods",
    ]
    topics_list = [0, 0, 0, 1, 1, 1]

    model, c_tf_idf, documents, topics = minimal_topic_model(docs, topics_list)

    repr_docs_mappings, _, _, _ = model._extract_representative_docs(
        c_tf_idf,
        documents,
        topics,
        nr_samples=500,
        nr_repr_docs=3,
        diversity=0.5,
    )

    for topic, docs_list in repr_docs_mappings.items():
        assert len(docs_list) == len(set(docs_list)), f"Topic {topic} has duplicate representative docs with diversity"


def test_multimodal_dedup_preserves_distinct_images(minimal_topic_model):
    """Distinct images sharing identical captions must not collapse into one candidate.

    Regression test for M01: `_extract_representative_docs` deduplicated on
    `Document` text alone after dropping the `Image` column, so multiple images
    captioned identically by an image-to-text model were treated as a single
    candidate, starving `VisualRepresentation` of images below `nr_repr_images`.
    """
    docs = ["scenic view"] * 5 + ["street scene"] * 4
    images = [f"img_{i}.jpg" for i in range(9)]
    topics_list = [0] * 9

    model, c_tf_idf, documents, topics = minimal_topic_model(docs, topics_list, images=images)

    repr_docs_mappings, _, _, repr_docs_ids = model._extract_representative_docs(
        c_tf_idf,
        documents,
        topics,
        nr_samples=500,
        nr_repr_docs=9,
    )

    # Only 2 distinct captions exist, but 9 distinct images back them. The buggy
    # dedup collapsed this to 2 candidates total; the fix must keep all 9.
    assert len(repr_docs_mappings[0]) == 9
    assert len(repr_docs_ids[0]) == 9
    assert len(set(repr_docs_ids[0])) == 9


def test_diversity_with_duplicate_text_maps_correct_ids(minimal_topic_model):
    """MMR branch must map indices positionally, not via a text-keyed lookup.

    Regression test for the `doc_to_index` landmine noted in the PR review
    (L04): once the dedup fix lets duplicate `Document` text survive dedup
    (distinct images, same caption), a text-keyed reverse lookup collapses
    every row sharing that text onto a single (wrong) index. `selected_indices`
    must be computed positionally so `docs`/`repr_docs_ids` stay aligned.
    """
    docs = ["same caption"] * 4
    images = ["img_0.jpg", "img_1.jpg", "img_2.jpg", "img_3.jpg"]
    topics_list = [0, 0, 0, 0]

    model, c_tf_idf, documents, topics = minimal_topic_model(docs, topics_list, images=images)

    _, _, _, repr_docs_ids = model._extract_representative_docs(
        c_tf_idf,
        documents,
        topics,
        nr_samples=500,
        nr_repr_docs=4,
        diversity=0.5,
    )

    # All 4 rows share identical text; only the distinct `Image`/row identity
    # tells them apart. A text-keyed `doc_to_index` lookup would map every
    # returned document to the same (last-enumerated) index, collapsing ids.
    assert sorted(repr_docs_ids[0]) == [0, 1, 2, 3]


def test_multimodal_dedup_handles_unhashable_loaded_images(minimal_topic_model):
    """Loaded (non-`str`) images must not crash `_extract_representative_docs`.

    Regression test for the crash M01 introduced: `drop_duplicates` hashes its
    subset columns, but `PIL.Image` sets `__hash__ = None`, so putting the
    `Image` column directly into the dedup subset raises `TypeError:
    unhashable type: 'Image'` the moment a pipeline carries loaded images
    rather than string paths (e.g. the documented multimodal quickstart, which
    loads a `datasets` image column directly). Distinct images sharing a
    caption must still survive, and two images that are pixel-identical
    (PIL's own `Image.__eq__`) must still collapse to one candidate.
    """
    Image = pytest.importorskip("PIL.Image")

    distinct = [Image.new("RGB", (4, 4), color) for color in ("red", "green", "blue")]
    duplicate_of_first = Image.new("RGB", (4, 4), "red")  # pixel-identical to distinct[0]

    docs = ["same caption"] * 4
    images = [*distinct, duplicate_of_first]
    topics_list = [0, 0, 0, 0]

    model, c_tf_idf, documents, topics = minimal_topic_model(docs, topics_list, images=images)

    repr_docs_mappings, _, _, repr_docs_ids = model._extract_representative_docs(
        c_tf_idf,
        documents,
        topics,
        nr_samples=500,
        nr_repr_docs=4,
    )

    # The pixel-identical duplicate collapses; the 3 distinct images survive.
    assert len(repr_docs_mappings[0]) == 3
    assert sorted(repr_docs_ids[0]) == [0, 1, 2]
