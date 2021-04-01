from ._base import BaseEmbedder
from sentence_transformers import SentenceTransformer


class SentenceTransformerBackend(BaseEmbedder):
    def __init__(self, embedding_model):
        super().__init__()

        if isinstance(embedding_model, SentenceTransformer):
            self.embedding_model = embedding_model
        elif isinstance(embedding_model, str):
            self.embedding_model = SentenceTransformer(embedding_model)

    def embed(self, documents, verbose):
        embeddings = self.embedding_model.encode(documents, show_progress_bar=verbose)
        return embeddings
