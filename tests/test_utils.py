import pytest
import logging
import numpy as np
from bertopic._utils import check_documents_type, check_embeddings_shape, MyLogger, get_unique_distances


def test_logger():
    logger = MyLogger("DEBUG")
    assert isinstance(logger.logger, logging.Logger)
    assert logger.logger.level == 10

    logger = MyLogger("WARNING")
    assert isinstance(logger.logger, logging.Logger)
    assert logger.logger.level == 30


@pytest.mark.parametrize(
    "docs",
    [
        "A document not in an iterable",
        [None],
        5
    ],
)
def test_check_documents_type(docs):
    with pytest.raises(TypeError):
        check_documents_type(docs)


def test_check_embeddings_shape():
    docs = ["doc_one", "doc_two"]
    embeddings = np.array([[1, 2, 3],
                           [2, 3, 4]])
    check_embeddings_shape(embeddings, docs)


def test_make_unique_distances():
    dists = np.array([0, 0, 0.5, 0.75, 1, 1])
    unique_dists = get_unique_distances(dists)

    assert len(unique_dists) == len(dists), "The number of elements must be the same"
    assert len(dists) == len(np.unique(unique_dists)), "The distances must be unique"
