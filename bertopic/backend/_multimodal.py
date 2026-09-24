import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Union
from sentence_transformers import SentenceTransformer

from bertopic._corpus import Modality
from bertopic.backend import BaseEmbedder


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
        batch_size: The sizes of image batches to pass

    Examples:
    To create a model, you can load in a string pointing to a
    sentence-transformers model:

    ```python
    from bertopic.backend import MultiModalBackend

    sentence_model = MultiModalBackend("clip-ViT-B-32")
    ```

    or  you can instantiate a model yourself:
    ```python
    from bertopic.backend import MultiModalBackend
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer("clip-ViT-B-32")
    sentence_model = MultiModalBackend(embedding_model)
    ```
    """

    def __init__(
        self,
        embedding_model: Union[str, SentenceTransformer],
        image_model: Union[str, SentenceTransformer] = None,
        audio_model: Union[str, SentenceTransformer] = None,
        video_model: Union[str, SentenceTransformer] = None,
        code_model: Union[str, SentenceTransformer] = None,
        batch_size: int = 32,
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
                self.models[modality] = SentenceTransformer(model)
            elif model is None:
                self.models[modality] = self.models[Modality.TEXT]
            else:
                raise ValueError(
                    "Please select a correct SentenceTransformers model: \n"
                    "`from sentence_transformers import SentenceTransformer` \n"
                    "`model = SentenceTransformer('clip-ViT-B-32')`"
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
        model = self.models[modality]

        if modality == Modality.IMAGE:
            return self._embed_images(media, model, verbose)
        return model.encode(media, batch_size=self.batch_size, show_progress_bar=verbose)

    def _embed_images(self, images: List, model: SentenceTransformer, verbose: bool) -> np.ndarray:
        """Embed images a batch at a time, opening any paths and closing them again.

        `encode` decides what an item is from its Python type, so a path reaches the text
        tower and returns an embedding of the filename rather than of the picture. Opening
        them in batches keeps that correct without holding every image in memory at once,
        which is why paths are the documented way to pass a large set.
        """
        embeddings = []
        for start_index in tqdm(range(0, len(images), self.batch_size), disable=not verbose):
            batch = images[start_index : start_index + self.batch_size]
            opened = [Image.open(image) if isinstance(image, str) else image for image in batch]
            embeddings.extend(model.encode(opened, show_progress_bar=False))

            # Close only the handles we opened ourselves
            for image, handle in zip(batch, opened):
                if isinstance(image, str):
                    handle.close()

        return np.array(embeddings)
