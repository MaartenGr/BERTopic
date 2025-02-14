from typing import List

import numpy as np
from bertopic.backend import BaseEmbedder


class LangchainBackend(BaseEmbedder):
    """Langchain Embedding Model"""

    def embed(self, documents: List[str], verbose: bool=False) -> np.ndarray:
        # Prepare documents, replacing empty strings with a single space
        prepared_documents = [" " if doc == "" else doc for doc in documents]
        response = self.embedding_model.embed_documents(prepared_documents)
        return np.array(response)
