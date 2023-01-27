import numpy as np
import logging
from collections.abc import Iterable
from scipy.sparse import csr_matrix


class MyLogger:
    def __init__(self, level):
        self.logger = logging.getLogger('BERTopic')
        self.set_level(level)
        self._add_handler()
        self.logger.propagate = False

    def info(self, message):
        self.logger.info("{}".format(message))

    def set_level(self, level):
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if level in levels:
            self.logger.setLevel(level)

    def _add_handler(self):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
        self.logger.addHandler(sh)

        # Remove duplicate handlers
        if len(self.logger.handlers) > 1:
            self.logger.handlers = [self.logger.handlers[0]]


def check_documents_type(documents):
    """ Check whether the input documents are indeed a list of strings """
    if isinstance(documents, Iterable) and not isinstance(documents, str):
        if not any([isinstance(doc, str) for doc in documents]):
            raise TypeError("Make sure that the iterable only contains strings.")

    else:
        raise TypeError("Make sure that the documents variable is an iterable containing strings only.")

def check_document_ids_type(document_ids):
    if not (isinstance(doc_ids, list) or isinstance(doc_ids, np.ndarray)):
        raise ValueError("document_ids must be of type List[str] or List[int]")
    if all((isinstance(doc_id, str) or isinstance(doc_id, np.str_)) for doc_id in document_ids):
        pass
    elif all((isinstance(doc_id, int) or isinstance(doc_id, np.int_)) for doc_id in document_ids):
        pass
    else:
        raise ValueError("Document ids must contain elements of the same type")

def check_embeddings_shape(embeddings, docs):
    """ Check if the embeddings have the correct shape """
    if embeddings is not None:
        if not any([isinstance(embeddings, np.ndarray), isinstance(embeddings, csr_matrix)]):
            raise ValueError("Make sure to input embeddings as a numpy array or scipy.sparse.csr.csr_matrix. ")
        else:
            if embeddings.shape[0] != len(docs):
                raise ValueError("Make sure that the embeddings are a numpy array with shape: "
                                 "(len(docs), vector_dim) where vector_dim is the dimensionality "
                                 "of the vector embeddings. ")


def check_is_fitted(topic_model):
    """ Checks if the model was fitted by verifying the presence of self.matches

    Arguments:
        model: BERTopic instance for which the check is performed.

    Returns:
        None

    Raises:
        ValueError: If the matches were not found.
    """
    msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
           "appropriate arguments before using this estimator.")

    if topic_model.topics_ is None:
        raise ValueError(msg % {'name': type(topic_model).__name__})

def validate_document_ids(documents_df, document_ids):
    doc_count = documents_df.shape[0]
    if isinstance(document_ids, list):
        if doc_count != len(document_ids):
            raise ValueError("document_ids must be the same size as documents")
        elif len(document_ids) != len(set(document_ids)):
            raise ValueError("document_ids must contain unique values")
    elif isinstance(document_ids, np.ndarray):
        if doc_count != document_ids.shape[0]:
            raise ValueError("document_ids must be the same size as documents")
        elif document_ids.shape[0] != np.unique(document_ids).shape[0]:
            raise ValueError("document_ids must contain unique values")

class NotInstalled:
    """
    This object is used to notify the user that additional dependencies need to be
    installed in order to use the string matching model.
    """

    def __init__(self, tool, dep):
        self.tool = tool
        self.dep = dep

        msg = f"In order to use {self.tool} you'll need to install via;\n\n"
        msg += f"pip install bertopic[{self.dep}]\n\n"
        self.msg = msg

    def __getattr__(self, *args, **kwargs):
        raise ModuleNotFoundError(self.msg)

    def __call__(self, *args, **kwargs):
        raise ModuleNotFoundError(self.msg)
