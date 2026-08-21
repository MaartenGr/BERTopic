"""Contract tests for the Corpus container.

`Corpus` is the value object that carries documents, embeddings, and assignments
through the pipeline. These tests pin down what it accepts, which matters because
its `__post_init__` validation is what currently rejects image-only input.
"""

import importlib.util

import numpy as np
import pytest

from bertopic._corpus import Corpus


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
@pytest.mark.xfail(
    strict=True,
    reason="Bug 1: image-only input raises in __post_init__; fixed in unit 2",
)
def test_images_may_be_supplied_without_documents(image_paths):
    """Multimodal input has no documents, which is the documented API for images.

    `docs/getting_started/multimodal/multimodal.md` calls
    `fit_transform(documents=None, images=images)`, but `check_documents_type` rejects
    `None` before any embedding happens, so `has_only_images` can never be True and
    `_images_to_text` is unreachable.
    """
    corpus = Corpus(documents=None, images=image_paths)

    assert corpus.has_only_images
    assert len(corpus.images) == len(image_paths)
