import numpy as np
from tqdm import tqdm
from typing import List
from bertopic.backend import BaseEmbedder
from gensim.models.keyedvectors import Word2VecKeyedVectors


class GensimBackend(BaseEmbedder):
    """ Gensim Embedding Model

    The Gensim embedding model is typically used for word embeddings with
    GloVe, Word2Vec or FastText.

    Arguments:
        embedding_model: A Gensim embedding model

    Usage:

    ```python
    from bertopic.backend import GensimBackend
    import gensim.downloader as api

    ft = api.load('fasttext-wiki-news-subwords-300')
    ft_embedder = GensimBackend(ft)
    ```
    """
    def __init__(self, embedding_model: Word2VecKeyedVectors):
        super().__init__()

        if isinstance(embedding_model, Word2VecKeyedVectors):
            self.embedding_model = embedding_model
        else:
            raise ValueError("Please select a correct Gensim model: \n"
                             "`import gensim.downloader as api` \n"
                             "`ft = api.load('fasttext-wiki-news-subwords-300')`")

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
        vector_shape = self.embedding_model.word_vec(list(self.embedding_model.vocab.keys())[0]).shape
        empty_vector = np.zeros(vector_shape[0])

        embeddings = []
        for doc in tqdm(documents, disable=not verbose, position=0, leave=True):
            doc_embedding = []

            # Extract word embeddings
            for word in doc.split(" "):
                try:
                    word_embedding = self.embedding_model.word_vec(word)
                    doc_embedding.append(word_embedding)
                except KeyError:
                    doc_embedding.append(empty_vector)

            # Pool word embeddings
            doc_embedding = np.mean(doc_embedding, axis=0)
            embeddings.append(doc_embedding)

        embeddings = np.array(embeddings)
        return embeddings
