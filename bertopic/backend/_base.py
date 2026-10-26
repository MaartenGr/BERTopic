import numpy as np
from typing import List


class BaseEmbedder:
    """The Base Embedder used for creating embedding models.

    Arguments:
        embedding_model: The main embedding model to be used for extracting
                         document and word embedding
    """

    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model

    def embed(self, documents: List[str], verbose: bool = False) -> np.ndarray:
        """Embed a list of n documents/words into an n-dimensional
        matrix of embeddings.

        Arguments:
            documents: A list of documents or words to be embedded
            verbose: Controls the verbosity of the process

        Returns:
            Document/words embeddings with shape (n, m) with `n` documents/words
            that each have an embeddings size of `m`
        """
        pass

    def embed_words(self, words: List[str], verbose: bool = False) -> np.ndarray:
        """Embed a list of n words into an n-dimensional
        matrix of embeddings.

        Words are the short texts BERTopic compares with documents or topics, such as
        a search term, a zero-shot label or a keyword. They are embedded as documents,
        unless a backend embeds them differently, such as without a prompt meant for documents.

        Arguments:
            words: A list of words to be embedded
            verbose: Controls the verbosity of the process

        Returns:
            Word embeddings with shape (n, m) with `n` words
            that each have an embeddings size of `m`
        """
        return self.embed_documents(words, verbose)

    def embed_documents(self, document: List[str], verbose: bool = False) -> np.ndarray:
        """Embed a list of n documents into an n-dimensional
        matrix of embeddings.

        Arguments:
            document: A list of documents to be embedded
            verbose: Controls the verbosity of the process

        Returns:
            Document embeddings with shape (n, m) with `n` documents
            that each have an embeddings size of `m`
        """
        return self.embed(document, verbose)
