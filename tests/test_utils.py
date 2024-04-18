import pytest
import logging
import numpy as np
from bertopic._utils import check_documents_type, check_embeddings_shape, MyLogger, select_topic_representation
from scipy.sparse import csr_matrix

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


def test_select_topic_representation():
    ctfidf_embeddings = np.array([[1, 1, 1]])
    topic_embeddings = np.array([[2, 2, 2]])

    # Use topic embeddings
    repr_, ctfidf_used = select_topic_representation(ctfidf_embeddings, topic_embeddings, use_ctfidf=False)
    np.testing.assert_array_equal(topic_embeddings, repr_)
    assert not ctfidf_used

    # Fallback to c-TF-IDF
    repr_, ctfidf_used = select_topic_representation(ctfidf_embeddings, None, use_ctfidf=False)
    np.testing.assert_array_equal(ctfidf_embeddings, repr_)
    assert ctfidf_used

    # Use c-TF-IDF
    repr_, ctfidf_used = select_topic_representation(ctfidf_embeddings, topic_embeddings, use_ctfidf=True)
    np.testing.assert_array_equal(
        ctfidf_embeddings,
        repr_
    )
    assert ctfidf_used

    # Fallback to topic embeddings
    repr_, ctfidf_used = select_topic_representation(None, topic_embeddings, use_ctfidf=True)
    np.testing.assert_array_equal(topic_embeddings, repr_)
    assert not ctfidf_used

    # No topic representation is provided
    with pytest.raises(ValueError):
        select_topic_representation(None, None, use_ctfidf=False)

    # `scipy.sparse.csr_matrix` can be used as c-TF-IDF embeddings
    np.testing.assert_array_equal(
        ctfidf_embeddings,
        select_topic_representation(
            csr_matrix(
                (ctfidf_embeddings.reshape(-1).tolist(), ([0, 0, 0], [0, 1, 2])),
                shape=ctfidf_embeddings.shape
            ),
            topic_embeddings,
            use_ctfidf=True
        )[0]
    )
