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
