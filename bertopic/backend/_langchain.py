from typing import List

import numpy as np
from bertopic.backend import BaseEmbedder


class LangchainBackend(BaseEmbedder):
    """Langchain Embedding Model.

    This class uses the Langchain Embedding class to embed the documents.
    Argument:
        embedding_model: A Langchain Embedding Instance.

    Examples:
    ```python
    from langchain_community.embeddings import HuggingFaceInstructEmbeddings
    from bertopic.backend import LangchainBackend

    hf_embedding = HuggingFaceInstructEmbeddings()
    langchain_embedder = LangchainBackend(hf_embedding)
    """
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
        # Prepare documents, replacing empty strings with a single space
        prepared_documents = [" " if doc == "" else doc for doc in documents]
        response = self.embedding_model.embed_documents(prepared_documents)
        return np.array(response)
