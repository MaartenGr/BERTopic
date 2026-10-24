import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer

from bertopic._corpus import Modality
from bertopic.backend import BaseEmbedder
from bertopic.backend._sentencetransformers import load_sentence_transformer


class MultiModalBackend(BaseEmbedder):
    """Multimodal backend using Sentence-transformers.

    One model may cover every modality, or each may be given its own. A modality without
    its own model falls back to `embedding_model`.

    Note that rows of different modalities are only comparable when a single model embeds
    them all, since two separately trained encoders do not share a vector space. Per-modality
    models are therefore for corpora of one modality; mixed corpora want a joint model.

    Arguments:
        embedding_model: A sentence-transformers embedding model covering every modality
                         that has no model of its own, text included.
        image_model: A sentence-transformers embedding model used to embed only images.
        audio_model: A sentence-transformers embedding model used to embed only audio.
        video_model: A sentence-transformers embedding model used to embed only video.
        code_model: A sentence-transformers embedding model used to embed only code.
        batch_size: The number of items to embed at a time

    Examples:
    To create a model, you can load in a string pointing to a
    sentence-transformers model:

    ```python
    from bertopic.backend import MultiModalBackend

    sentence_model = MultiModalBackend("google/embeddinggemma-2")
    ```

    or  you can instantiate a model yourself:
    ```python
    from bertopic.backend import MultiModalBackend
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer("google/embeddinggemma-2", default_prompt_name="Clustering")
    embedding_model.max_seq_length = 1024
    sentence_model = MultiModalBackend(embedding_model)
    ```
    """

    def __init__(
        self,
        embedding_model: Union[str, SentenceTransformer],
        image_model: Union[str, SentenceTransformer] = None,
        batch_size: int = 32,
        *,
        audio_model: Union[str, SentenceTransformer] = None,
        video_model: Union[str, SentenceTransformer] = None,
        code_model: Union[str, SentenceTransformer] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.models = {}

        # Text is resolved first, since it is what every other modality falls back to
        for modality, model in (
            (Modality.TEXT, embedding_model),
            (Modality.IMAGE, image_model),
            (Modality.AUDIO, audio_model),
            (Modality.VIDEO, video_model),
            (Modality.CODE, code_model),
        ):
            if isinstance(model, SentenceTransformer):
                self.models[modality] = model
            elif isinstance(model, str):
                self.models[modality] = load_sentence_transformer(model)
            elif model is None:
                self.models[modality] = self.models[Modality.TEXT]
            else:
                raise ValueError(
                    "Please select a correct SentenceTransformers model: \n"
                    "`from sentence_transformers import SentenceTransformer` \n"
                    "`model = SentenceTransformer('google/embeddinggemma-2')`"
                )

        self.embedding_model = self.models[Modality.TEXT]

    def embed(self, documents: List[str], verbose: bool = False) -> np.ndarray:
        """Embed a list of n documents/words into an n-dimensional matrix of embeddings.

        Arguments:
            documents: A list of documents or words to be embedded
            verbose: Controls the verbosity of the process

        Returns:
            Document/words embeddings with shape (n, m) with `n` documents/words
            that each have an embeddings size of `m`
        """
        return self.embed_media(documents, Modality.TEXT, verbose)

    def embed_media(
        self, media: List, modality: Modality = Modality.TEXT, verbose: bool = False
    ) -> np.ndarray:
        """Embed a list of n items sharing one modality, with the model that handles it.

        Arguments:
            media: A list of items, all of the same modality
            modality: Which modality the items are, choosing the model that embeds them
            verbose: Controls the verbosity of the process

        Returns:
            Embeddings with shape (n, m) with `n` items that each have an
            embeddings size of `m`
        """
        modality = Modality(modality)
        model = self.models[modality]

        # A model that takes this kind of media is told what each item is, so a path is never read as text.
        # The clustering prompt and the length limit are for text, so media goes without both
        if modality != Modality.TEXT and modality in model.modalities:
            items = [{modality.value: item} for item in media]
            return model.encode(
                items,
                batch_size=self.batch_size,
                show_progress_bar=verbose,
                prompt="",
                processing_kwargs={"text": {"truncation": False}},
            )
        return model.encode(media, batch_size=self.batch_size, show_progress_bar=verbose)
