"""
Unit tests for utility functions
"""

import pytest
import logging
import numpy as np
from bertopic._utils import check_documents_type, check_embeddings_shape, MyLogger


def test_logger():
    """ Test whether the logger could correctly be instantiated """
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
    """ Making sure the checks for document type succeed """
    with pytest.raises(TypeError):
        check_documents_type(docs)


def test_check_embeddings_shape():
    """ Testing correct embeddings shape

    Checking for embeddings is typically done when custom embeddings
    are used instead of the default SentenceTransformer models. This means
    that the embeddings should have the correct shape, which is tested for
    in check_embeddings_shape.
    """
    docs = ["doc_one", "doc_two"]
    embeddings = np.array([[1, 2, 3],
                           [2, 3, 4]])
    check_embeddings_shape(embeddings, docs)
