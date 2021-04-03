import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer

from bertopic.backend import BaseEmbedder


class SentenceTransformerBackend(BaseEmbedder):
    """ Sentence-transformers embedding model

    The sentence-transformers embedding model used for generating document and
    word embeddings.

    Arguments:
        embedding_model: A sentence-transformers embedding model

    Usage:

    To create a model, you can load in a string pointing to a
    sentence-transformers model:

    ```python
    from bertopic.backend import SentenceTransformerBackend

    sentence_model = SentenceTransformerBackend("distilbert-base-nli-stsb-mean-tokens")
    ```

    or  you can instantiate a model yourself:
    ```python
    from bertopic.backend import SentenceTransformerBackend
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")
    sentence_model = SentenceTransformerBackend(embedding_model)
    ```
    """
    def __init__(self, embedding_model: Union[str, SentenceTransformer]):
        super().__init__()

        if isinstance(embedding_model, SentenceTransformer):
            self.embedding_model = embedding_model
        elif isinstance(embedding_model, str):
            self.embedding_model = SentenceTransformer(embedding_model)

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
        embeddings = self.embedding_model.encode(documents, show_progress_bar=verbose)
        return embeddings


languages = ['afrikaans', 'albanian', 'amharic', 'arabic', 'armenian', 'assamese',
             'azerbaijani', 'basque', 'belarusian', 'bengali', 'bengali romanize',
             'bosnian', 'breton', 'bulgarian', 'burmese', 'burmese zawgyi font', 'catalan',
             'chinese (simplified)', 'chinese (traditional)', 'croatian', 'czech', 'danish',
             'dutch', 'english', 'esperanto', 'estonian', 'filipino', 'finnish', 'french',
             'galician', 'georgian', 'german', 'greek', 'gujarati', 'hausa', 'hebrew', 'hindi',
             'hindi romanize', 'hungarian', 'icelandic', 'indonesian', 'irish', 'italian', 'japanese',
             'javanese', 'kannada', 'kazakh', 'khmer', 'korean', 'kurdish (kurmanji)', 'kyrgyz',
             'lao', 'latin', 'latvian', 'lithuanian', 'macedonian', 'malagasy', 'malay', 'malayalam',
             'marathi', 'mongolian', 'nepali', 'norwegian', 'oriya', 'oromo', 'pashto', 'persian',
             'polish', 'portuguese', 'punjabi', 'romanian', 'russian', 'sanskrit', 'scottish gaelic',
             'serbian', 'sindhi', 'sinhala', 'slovak', 'slovenian', 'somali', 'spanish', 'sundanese',
             'swahili', 'swedish', 'tamil', 'tamil romanize', 'telugu', 'telugu romanize', 'thai',
             'turkish', 'ukrainian', 'urdu', 'urdu romanize', 'uyghur', 'uzbek', 'vietnamese',
             'welsh', 'western frisian', 'xhosa', 'yiddish']
