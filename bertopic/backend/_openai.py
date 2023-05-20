import time
import openai
import numpy as np
from tqdm import tqdm
from typing import List
from bertopic.backend import BaseEmbedder


class OpenAIBackend(BaseEmbedder):
    """ OpenAI Embedding Model
    
    Arguments:
        embedding_model: An OpenAI model. Default is 
                         For an overview of models see:
                         https://platform.openai.com/docs/models/embeddings
        delay_in_seconds: If a `batch_size` is given, use this set
                          the delay in seconds between batches.
        batch_size: The size of each batch.

    Examples:

    ```python
    import openai
    from bertopic.backend import OpenAIBackend

    openai.api_key = MY_API_KEY
    openai_embedder = OpenAIBackend("text-embedding-ada-002")
    ```
    """
    def __init__(self, 
                 embedding_model: str = "text-embedding-ada-002",
                 delay_in_seconds: float = None,
                 batch_size: int = None):
        super().__init__()
        self.embedding_model = embedding_model
        self.delay_in_seconds = delay_in_seconds
        self.batch_size = batch_size

    def embed(self,
              documents: List[str],
              verbose: bool = False) -> np.ndarray:
        """ Embed a list of n documents/words into an n-dimensional
        matrix of embeddings

        Arguments:
            documents: A list of documents or words to be embedded
            verbose: Controls the verbosity of the process

        Returns:
            Document/words embeddings with shape (n, m) with `n` documents/words
            that each have an embeddings size of `m`
        """
        # Batch-wise embedding extraction
        if self.batch_size is not None:
            embeddings = []
            for batch in tqdm(self._chunks(documents), disable=not verbose):
                response = openai.Embedding.create(input=batch, model=self.embedding_model)
                embeddings.extend([r["embedding"] for r in response["data"]])

                # Delay subsequent calls
                if self.delay_in_seconds:
                    time.sleep(self.delay_in_seconds)

        # Extract embeddings all at once
        else:
            response = openai.Embedding.create(input=documents, model=self.embedding_model)
            embeddings = [r["embedding"] for r in response["data"]]
        return np.array(embeddings)
    
    def _chunks(self, documents):     
        for i in range(0, len(documents), self.batch_size):
            yield documents[i:i + self.batch_size]
