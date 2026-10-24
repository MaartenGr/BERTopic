import numpy as np
import torch
from typing import List, Union
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import StaticEmbedding
from transformers.tokenization_utils_base import VERY_LARGE_INTEGER

from bertopic.backend import BaseEmbedder


def load_sentence_transformer(name: str, **kwargs) -> SentenceTransformer:
    """Load a sentence-transformers model by name, ready to embed documents for clustering.

    A model trained with a clustering prompt, such as EmbeddingGemma 2, prepends it to every text.
    A model that sets no length limit of its own, which transformers reports as `VERY_LARGE_INTEGER`,
    embeds the first 1,024 tokens of a document rather than all of it, however long it is. And
    without a GPU that runs bfloat16 natively, as on most CPUs, bfloat16 weights run several times
    slower than float32 ones, so the model is loaded in float32 there.

    Arguments:
        name: The name of the model on the Hugging Face Hub, or a local path
        **kwargs: Passed to `SentenceTransformer`, such as `config_kwargs`

    Returns:
        model: The loaded model
    """
    # Ampere and newer GPUs run bfloat16 natively
    if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8):
        kwargs["model_kwargs"] = {"dtype": torch.float32, **kwargs.get("model_kwargs", {})}
    model = SentenceTransformer(name, **kwargs)
    if "Clustering" in model.prompts:
        model.default_prompt_name = "Clustering"
    if model.max_seq_length == VERY_LARGE_INTEGER:
        model.max_seq_length = 1024
    return model


class SentenceTransformerBackend(BaseEmbedder):
    """Sentence-transformers embedding model.

    The sentence-transformers embedding model used for generating document and
    word embeddings.

    Arguments:
        embedding_model: A sentence-transformers embedding model
        model2vec: Indicates whether `embedding_model` is a model2vec model.
                   NOTE: Only works if `embedding_model` is a string.
                   Otherwise, you can pass the model2vec model directly to `embedding_model`.
        **kwargs: Passed to `SentenceTransformer` when `embedding_model` is a name, such as `config_kwargs`.

    Examples:
    To create a model, you can load in a string pointing to a
    sentence-transformers model:

    ```python
    from bertopic.backend import SentenceTransformerBackend

    sentence_model = SentenceTransformerBackend("all-MiniLM-L6-v2")
    ```

    or  you can instantiate a model yourself:

    ```python
    from bertopic.backend import SentenceTransformerBackend
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_model = SentenceTransformerBackend(embedding_model)
    ```

    If you want to use a model2vec model without having to install model2vec,
    you can pass the model2vec model as a string:

    ```python
    from bertopic.backend import SentenceTransformerBackend
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer("minishlab/potion-base-8M", model2vec=True)
    sentence_model = SentenceTransformerBackend(embedding_model)
    ```
    """

    def __init__(self, embedding_model: Union[str, SentenceTransformer], model2vec: bool = False, **kwargs):
        super().__init__()

        self._hf_model = None
        if model2vec and isinstance(embedding_model, str):
            static_embedding = StaticEmbedding.from_model2vec(embedding_model)
            self.embedding_model = SentenceTransformer(modules=[static_embedding])
        elif isinstance(embedding_model, SentenceTransformer):
            self.embedding_model = embedding_model
        elif isinstance(embedding_model, str):
            self.embedding_model = load_sentence_transformer(embedding_model, **kwargs)
            self._hf_model = embedding_model
        else:
            raise ValueError(
                "Please select a correct SentenceTransformers model: \n"
                "`from sentence_transformers import SentenceTransformer` \n"
                "`model = SentenceTransformer('all-MiniLM-L6-v2')`"
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
        embeddings = self.embedding_model.encode(documents, show_progress_bar=verbose)
        return embeddings
