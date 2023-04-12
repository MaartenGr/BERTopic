import numpy as np
from tqdm import tqdm
from typing import List
from bertopic.backend import BaseEmbedder


class SpacyBackend(BaseEmbedder):
    """ Spacy embedding model

    The Spacy embedding model used for generating document and
    word embeddings.

    Arguments:
        embedding_model: A spacy embedding model

    Examples:

    To create a Spacy backend, you need to create an nlp object and
    pass it through this backend:

    ```python
    import spacy
    from bertopic.backend import SpacyBackend

    nlp = spacy.load("en_core_web_md", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
    spacy_model = SpacyBackend(nlp)
    ```

    To load in a transformer model use the following:

    ```python
    import spacy
    from thinc.api import set_gpu_allocator, require_gpu
    from bertopic.backend import SpacyBackend

    nlp = spacy.load("en_core_web_trf", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
    set_gpu_allocator("pytorch")
    require_gpu(0)
    spacy_model = SpacyBackend(nlp)
    ```

    If you run into gpu/memory-issues, please use:

    ```python
    import spacy
    from bertopic.backend import SpacyBackend

    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
    spacy_model = SpacyBackend(nlp)
    ```
    """
    def __init__(self, embedding_model):
        super().__init__()

        if "spacy" in str(type(embedding_model)):
            self.embedding_model = embedding_model
        else:
            raise ValueError("Please select a correct Spacy model by either using a string such as 'en_core_web_md' "
                             "or create a nlp model using: `nlp = spacy.load('en_core_web_md')")

    def _get_if_cupy(embedding):
        """Convert to numpy arrays depending on whether cupy was used or not"""
        if (method := getattr(embedding, 'get', None) and callable(method):
            embedding = embedding.get()
        return embedding

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
        # Handle empty documents, spaCy models automatically map
        # empty strings to the zero vector
        empty_document = " "
        embeddings = []

        for doc in tqdm(documents, position=0, leave=True, disable=not verbose):
            spacy_embedding = self.emebdding_model(doc or empty_document)

            # Extract embeddings from a transformer model
            if "transformer" in self.embedding_model.component_names:
                embedding = spacy_embedding._.trf_data.tensors[-1][0]
                embedding = self._get_if_cupy(embedding).tolist()

            # Extract embeddings from a general spacy model
            else:
                embedding = self._get_if_cupy(embedding.vector)

            embeddings.append(embedding.tolist())

        return np.array(embeddings)
