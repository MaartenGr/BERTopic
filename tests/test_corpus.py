"""Contract tests for the Corpus container.

`Corpus` is the value object that carries documents, embeddings, and assignments
through the pipeline. Text and media are parallel channels: `documents` feeds
c-TF-IDF, `media` holds whatever a row is made of that is not text, and `modality`
says what `media` holds. These tests pin down that contract.
"""

import importlib.util

import numpy as np
import pytest

from bertopic._corpus import Corpus, Modality


def pillow_available():
    """Check whether Pillow is installed, since it is only in the `vision` extra."""
    try:
        return importlib.util.find_spec("PIL") is not None
    except ImportError:
        return False


def test_documents_are_normalised_to_a_list():
    """A single document may be passed as a bare string during inference."""
    corpus = Corpus(documents="a single document")

    assert corpus.documents == ["a single document"]


def test_numpy_documents_are_converted_to_a_list():
    """Document arrays are accepted and stored as a plain list."""
    corpus = Corpus(documents=np.array(["first", "second"]))

    assert corpus.documents == ["first", "second"]


def test_original_indices_default_to_positions():
    """Without explicit indices each document is identified by its position."""
    corpus = Corpus(documents=["first", "second", "third"])

    assert list(corpus.original_indices) == [0, 1, 2]


def test_mismatched_embeddings_are_rejected():
    """An embedding matrix must carry exactly one row per document."""
    with pytest.raises(ValueError):
        Corpus(documents=["first", "second"], embeddings=np.zeros((3, 8)))


def test_mismatched_topics_are_rejected():
    """Assignments must line up with documents, which the length guard enforces."""
    corpus = Corpus(documents=["first", "second"])

    with pytest.raises(ValueError):
        corpus.topics = np.array([0, 1, 2])


@pytest.mark.skipif(not pillow_available(), reason="Pillow not available")
def test_media_may_be_supplied_without_documents(image_paths):
    """Image-only input used to raise in `__post_init__` before any embedding ran."""
    corpus = Corpus(media=image_paths, modality=Modality.IMAGE)

    assert len(corpus) == len(image_paths)
    assert corpus.images == image_paths
    assert not any(corpus.documents)


def test_documents_default_to_the_text_modality():
    """Text rows are their own source, so nothing is stored twice."""
    corpus = Corpus(documents=["first", "second"])

    assert corpus.modality == [Modality.TEXT, Modality.TEXT]
    assert corpus.media == [None, None]


def test_a_single_modality_applies_to_every_row():
    """Most corpora are uniform, so a scalar modality is broadcast."""
    corpus = Corpus(media=["a.png", "b.png", "c.png"], modality=Modality.IMAGE)

    assert corpus.modality == [Modality.IMAGE] * 3


def test_a_row_may_carry_both_text_and_media():
    """An image with a caption is one row with both channels populated."""
    corpus = Corpus(documents=["a cat", "a dog"], media=["cat.png", "dog.png"], modality=Modality.IMAGE)

    assert corpus.documents == ["a cat", "a dog"]
    assert corpus.images == ["cat.png", "dog.png"]
    assert all(corpus.documents)


def test_rows_may_each_have_their_own_modality():
    """Unrelated sets of text and images can share a corpus."""
    corpus = Corpus(
        documents=["some text", ""],
        media=[None, "picture.png"],
        modality=[Modality.TEXT, Modality.IMAGE],
    )

    assert corpus.images == ["picture.png"]
    assert len(corpus) == 2


def test_rows_without_text_keep_an_empty_document():
    """The text channel stays addressable until a representation model fills it in."""
    corpus = Corpus(media=["a.png", "b.png"], modality=Modality.IMAGE)

    assert corpus.documents == ["", ""]


def test_selecting_by_index_carries_both_channels():
    """Slicing keeps every row's media and modality alongside its text."""
    corpus = Corpus(
        documents=["first", "second", "third"],
        media=["a.png", "b.png", "c.png"],
        modality=Modality.IMAGE,
    )

    selected = corpus.get_corpus_by_indices([0, 2])

    assert selected.documents == ["first", "third"]
    assert selected.media == ["a.png", "c.png"]
    assert selected.modality == [Modality.IMAGE, Modality.IMAGE]


def test_selecting_by_topic_carries_both_channels():
    """`get_topic` used to drop image media entirely."""
    corpus = Corpus(
        documents=["first", "second", "third"],
        media=["a.png", "b.png", "c.png"],
        modality=Modality.IMAGE,
        topics=np.array([0, 1, 0]),
    )

    selected = corpus.get_topic(0)

    assert selected.media == ["a.png", "c.png"]
    assert selected.images == ["a.png", "c.png"]


def test_combining_corpora_carries_both_channels():
    """Zero-shot recombines two corpora and must not lose media."""
    first = Corpus(
        documents=["first"],
        media=["a.png"],
        modality=Modality.IMAGE,
        topics=np.array([0]),
        embeddings=np.zeros((1, 4)),
        original_indices=[0],
    )
    second = Corpus(
        documents=["second"],
        media=["b.png"],
        modality=Modality.IMAGE,
        topics=np.array([1]),
        embeddings=np.ones((1, 4)),
        original_indices=[1],
    )

    combined = first + second

    assert combined.media == ["a.png", "b.png"]
    assert combined.modality == [Modality.IMAGE, Modality.IMAGE]


def test_mismatched_channels_are_rejected_at_construction():
    """A short channel would otherwise be silently truncated by every later zip."""
    with pytest.raises(ValueError):
        Corpus(documents=["first", "second"], media=["only-one.png"], modality=Modality.IMAGE)

    with pytest.raises(ValueError):
        Corpus(documents=["first", "second"], modality=[Modality.TEXT])


def test_mismatched_media_is_rejected():
    """Content is a per-row field, so its length is guarded like the others."""
    corpus = Corpus(documents=["first", "second"])

    with pytest.raises(ValueError):
        corpus.media = ["only one"]


def test_every_row_is_its_own_document_by_default():
    """1.0 fits on whole documents, so the grouping key is the identity."""
    corpus = Corpus(documents=["first", "second", "third"])

    assert list(corpus.document_ids) == [0, 1, 2]
    assert corpus.nr_documents == len(corpus)


def test_rows_may_share_a_document():
    """What a sentence splitter would produce: four rows from two documents.

    Nothing generates this in 1.0. The key exists so that 1.1 can add a splitter without
    a second pass over the data layer.
    """
    corpus = Corpus(
        documents=["first half", "second half", "another one", "and its tail"],
        document_ids=np.array([0, 0, 1, 1]),
    )

    assert corpus.nr_documents == 2
    assert len(corpus) == 4


def test_selecting_by_index_carries_document_ids():
    """Slicing a corpus keeps each row pointing at the document it came from."""
    corpus = Corpus(documents=["a", "b", "c", "d"], document_ids=np.array([0, 0, 1, 1]))

    selected = corpus.get_corpus_by_indices([1, 2])

    assert list(selected.document_ids) == [0, 1]
    assert selected.nr_documents == 2


def test_selecting_by_topic_carries_document_ids():
    corpus = Corpus(
        documents=["a", "b", "c"],
        document_ids=np.array([0, 0, 1]),
        topics=np.array([0, 1, 0]),
    )

    selected = corpus.get_topic(0)

    assert list(selected.document_ids) == [0, 1]


def test_combining_corpora_carries_document_ids():
    """Zero-shot splits a corpus and recombines it, which must not renumber the rows."""
    first = Corpus(
        documents=["first"], topics=np.array([0]), embeddings=np.zeros((1, 4)), original_indices=[0]
    )
    second = Corpus(
        documents=["second"], topics=np.array([1]), embeddings=np.ones((1, 4)), original_indices=[1]
    )

    combined = first + second

    assert len(combined.document_ids) == 2


def test_mismatched_document_ids_are_rejected():
    """A grouping key that is not one-per-row would silently misalign every later zip."""
    corpus = Corpus(documents=["first", "second"])

    with pytest.raises(ValueError):
        corpus.document_ids = np.array([0])


def test_recombining_a_split_corpus_keeps_documents_distinct():
    """Zero-shot builds two corpora from one input and adds them back together.

    Each defaults its own ids, so numbering them from scratch would make row 0 of each
    half look like the same document. `original_indices` is what keeps them apart.
    """
    first = Corpus(
        documents=["a", "c", "e"],
        topics=np.array([0, 1, 0]),
        embeddings=np.zeros((3, 4)),
        original_indices=[0, 2, 4],
    )
    second = Corpus(
        documents=["b", "d", "f"],
        topics=np.array([1, 0, 1]),
        embeddings=np.ones((3, 4)),
        original_indices=[1, 3, 5],
    )

    combined = first + second

    assert combined.nr_documents == 6


def test_candidate_sampling_is_repeatable():
    """Representative documents feed LLM representations, so they must not drift.

    The sampling was unseeded, so every fit chose different candidates and any label
    built from them changed run to run. `random_state` on UMAP does not reach here.
    """
    corpus = Corpus(
        documents=[f"document {index}" for index in range(1000)],
        topics=np.array([0] * 1000),
        embeddings=np.zeros((1000, 4)),
    )

    runs = [corpus.get_topic(0, nr_samples=100).documents for _ in range(3)]

    assert runs[0] == runs[1] == runs[2]


def test_sampling_only_kicks_in_above_the_limit():
    """A topic smaller than `nr_samples` keeps every document, in order."""
    corpus = Corpus(documents=["a", "b", "c"], topics=np.array([0, 0, 0]))

    assert corpus.get_topic(0, nr_samples=100).documents == ["a", "b", "c"]
