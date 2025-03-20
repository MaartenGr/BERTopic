import numpy as np
from typing import List, Union
from fastembed import TextEmbedding

from bertopic.backend import BaseEmbedder


class FastEmbedBackend(BaseEmbedder):
    """Sentence-transformers embedding model.

    The sentence-transformers embedding model used for generating document and
    word embeddings.

    Arguments:
        embedding_model: A sentence-transformers embedding model

    Examples:
    To create a model, you can load in a string pointing to a
    sentence-transformers model:

    ```python
    from bertopic.backend import FastEmbedBackend

    sentence_model = FastEmbedBackend("BAAI/bge-small-en-v1.5")
    ```
    """

    def __init__(self, embedding_model: str = "BAAI/bge-small-en-v1.5"):
        super().__init__()

        supported_models = [m['model'] for m in TextEmbedding.list_supported_models()]

        if isinstance(embedding_model, str) and embedding_model in supported_models:
            self.embedding_model = TextEmbedding(model_name=embedding_model)
        else:
            raise ValueError(
                "Please select a correct FasteEmbed model: \n"
                "the model must be a string and must be supported. \n"
                "The supported TextEmbedding model list is here: https://qdrant.github.io/fastembed/examples/Supported_Models/"
            )

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
        embeddings_generator = self.embedding_model.embed(documents, show_progress_bar=verbose)
        embeddings_list = list(embeddings_generator)
        embeddings_array = np.stack(embeddings_list)
        return embeddings_array
