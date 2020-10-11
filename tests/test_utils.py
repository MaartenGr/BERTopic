import pytest
import logging
from bertopic.utils import check_documents_type, sentence_models, create_logger


def test_utils():
    """ Test if sentence models are correctly returned """
    models = sentence_models()
    assert len(models) >= 10
    assert isinstance(models, list)


def test_logger():
    logger = create_logger()

    assert isinstance(logger, logging.Logger)
    assert logger.level == 30


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
