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

    Usage:

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

        # Extract embeddings from a transformer model
        if "transformer" in self.embedding_model.component_names:
            embeddings = []
            for doc in tqdm(documents, position=0, leave=True, disable=not verbose):
                try:
                    embedding = self.embedding_model(doc)._.trf_data.tensors[-1][0].tolist()
                except:
                    embedding = self.embedding_model("An empty document")._.trf_data.tensors[-1][0].tolist()
                embeddings.append(embedding)
            embeddings = np.array(embeddings)

        # Extract embeddings from a general spacy model
        else:
            embeddings = []
            for doc in tqdm(documents, position=0, leave=True, disable=not verbose):
                try:
                    vector = self.embedding_model(doc).vector
                except ValueError:
                    vector = self.embedding_model("An empty document").vector
                embeddings.append(vector)
            embeddings = np.array(embeddings)

        return embeddings
