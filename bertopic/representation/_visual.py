import numpy as np

from PIL import Image
from tqdm import tqdm
from scipy.sparse import csr_matrix
from transformers.pipelines import Pipeline, pipeline

from bertopic.representation._mmr import mmr
from bertopic.representation._base import TextConverter
from bertopic._corpus import Corpus, Modality
from bertopic._topics import Images, Keywords

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bertopic import BERTopic


class VisualRepresentation(TextConverter):
    """From a collection of representative documents, extract
    images to represent topics. These topics are represented by a
    collage of images.

    Arguments:
        nr_repr_images: Number of representative images to extract
        nr_samples: The number of candidate documents to extract per cluster.
        image_height: The height of the resulting collage
        image_square: Whether to resize each image in the collage
                      to a square. This can be visually more appealing
                      if all input images are all almost squares.
        image_to_text_model: The model to caption images.
        batch_size: The number of images to pass to the
                    `image_to_text_model`.

    Usage:

    ```python
    from bertopic.representation import VisualRepresentation
    from bertopic import BERTopic

    # The visual representation is typically not a core representation
    # and is advised to pass to BERTopic as an additional aspect.
    # Aspects can be labeled with dictionaries as shown below:
    representation_model = {
        "Visual_Aspect": VisualRepresentation()
    }

    # Use the representation model in BERTopic as a separate aspect
    topic_model = BERTopic(representation_model=representation_model)
    ```
    """

    def __init__(
        self,
        nr_repr_images: int = 9,
        nr_samples: int = 500,
        image_height: tuple[int, int] = 600,
        image_squares: bool = False,
        image_to_text_model: str | Pipeline = None,
        batch_size: int = 32,
    ):
        self.nr_repr_images = nr_repr_images
        self.nr_samples = nr_samples
        self.image_height = image_height
        self.image_squares = image_squares

        # Text-to-image model
        if isinstance(image_to_text_model, Pipeline):
            self.image_to_text_model = image_to_text_model
        elif isinstance(image_to_text_model, str):
            self.image_to_text_model = pipeline("image-to-text", model=image_to_text_model)
        elif image_to_text_model is None:
            self.image_to_text_model = None
        else:
            raise ValueError(
                "Please select a correct transformers pipeline. For example:"
                "pipeline('image-to-text', model='nlpconnect/vit-gpt2-image-captioning')"
            )
        self.batch_size = batch_size
        self.modality = Modality.IMAGE

    def extract_topics(
        self,
        topic_model: "BERTopic",
        corpus: Corpus,
        topic_representations: dict[int, Keywords],
        c_tf_idf: csr_matrix,
        embeddings: np.ndarray = None,
    ) -> dict[int, Images]:
        """Extract a collage of representative images per topic.

        Arguments:
            topic_model: A BERTopic model
            corpus: The input documents including (calculated) embeddings
            topic_representations: The candidate topic representations
            c_tf_idf: The topic c-TF-IDF representation
            embeddings: Pre-trained document embeddings (unused, for API compatibility)

        Returns:
            An `Images` representation per topic, carrying the collage and its captions
        """
        # Find the rows that best represent each topic
        (_, _, _, repr_docs_ids) = topic_model._extract_representative_docs(
            c_tf_idf=c_tf_idf,
            corpus=corpus,
            nr_samples=self.nr_samples,
            nr_repr_docs=self.nr_repr_images,
        )

        # Combine each topic's images into a single collage
        representations = {}
        for index, topic in enumerate(tqdm(sorted(topic_representations))):
            row_indices = repr_docs_ids[index]
            images = [self._open(corpus.media[row]) for row in row_indices]

            # Tile the images three to a row before resizing them into one image
            rows = [images[start : start + 3] for start in range(0, len(images), 3)]
            collage = get_concat_tile_resize(rows, self.image_height, self.image_squares)

            captions = [corpus.documents[row] for row in row_indices if corpus.documents[row]]
            representations[topic] = Images(data=collage, captions=captions)

            for image in images:
                image.close()

        return representations

    @staticmethod
    def _open(image):
        """Open an image from a path, or copy one that is already loaded."""
        return Image.open(image) if isinstance(image, str) else image.copy()

    def _convert_image_to_text(self, images: list[str], verbose: bool = False) -> list[str]:
        """Convert a list of images to captions.

        Arguments:
            images: A list of images or words to be converted to text.
            verbose: Controls the verbosity of the process

        Returns:
            List of captions
        """
        # Batch-wise image conversion
        if self.batch_size is not None:
            documents = []
            for batch in tqdm(self._chunks(images), disable=not verbose):
                outputs = self.image_to_text_model(batch)
                captions = [output[0]["generated_text"] for output in outputs]
                documents.extend(captions)

        # Convert images to text
        else:
            outputs = self.image_to_text_model(images)
            documents = [output[0]["generated_text"] for output in outputs]

        return documents

    def to_text(self, corpus: Corpus) -> Corpus:
        """Caption the most representative images per topic, keeping only those rows.

        Captioning every image is the expensive step, so a diverse sample per topic
        stands in for the whole: MMR picks images near each topic centroid while
        discarding near-duplicates. The returned corpus holds only those rows, with
        their captions in the text channel for c-TF-IDF to read.
        """
        image_rows = [index for index, modality in enumerate(corpus.modality) if modality == self.modality]
        if self.image_to_text_model is None or not image_rows:
            return corpus

        # Pick a diverse sample of images near each topic's centroid
        selected_indices = []
        for topic, topic_embedding in corpus.average_embeddings_by_topic().items():
            indices = np.array([index for index in image_rows if corpus.topics[index] == topic])
            if not len(indices):
                continue
            selected_indices.extend(
                mmr(
                    topic_embedding.reshape(1, -1),
                    corpus.embeddings[indices],
                    indices,
                    top_n=min(self.nr_repr_images, len(indices)),
                    diversity=0.1,
                )
            )

        # Caption them, so the text channel describes what the images show
        selected_corpus = corpus.get_corpus_by_indices(selected_indices)
        images = [self._open(image) for image in selected_corpus.media]
        selected_corpus.documents = self._convert_image_to_text(images)

        for image in images:
            image.close()

        return selected_corpus

    def _chunks(self, images):
        for i in range(0, len(images), self.batch_size):
            yield images[i : i + self.batch_size]


def get_concat_h_multi_resize(im_list):
    """Code adapted from: https://note.nkmk.me/en/python-pillow-concat-images/."""
    min_height = min(im.height for im in im_list)
    min_height = max(im.height for im in im_list)
    im_list_resize = []
    for im in im_list:
        im.resize((int(im.width * min_height / im.height), min_height), resample=0)
        im_list_resize.append(im)

    total_width = sum(im.width for im in im_list_resize)
    dst = Image.new("RGB", (total_width, min_height), (255, 255, 255))
    pos_x = 0
    for im in im_list_resize:
        dst.paste(im, (pos_x, 0))
        pos_x += im.width
    return dst


def get_concat_v_multi_resize(im_list):
    """Code adapted from: https://note.nkmk.me/en/python-pillow-concat-images/."""
    min_width = min(im.width for im in im_list)
    min_width = max(im.width for im in im_list)
    im_list_resize = [
        im.resize((min_width, int(im.height * min_width / im.width)), resample=0) for im in im_list
    ]
    total_height = sum(im.height for im in im_list_resize)
    dst = Image.new("RGB", (min_width, total_height), (255, 255, 255))
    pos_y = 0
    for im in im_list_resize:
        dst.paste(im, (0, pos_y))
        pos_y += im.height
    return dst


def get_concat_tile_resize(im_list_2d, image_height=600, image_squares=False):
    """Code adapted from: https://note.nkmk.me/en/python-pillow-concat-images/."""
    images = [[image.copy() for image in images] for images in im_list_2d]

    # Create
    if image_squares:
        width = int(image_height / 3)
        height = int(image_height / 3)
        images = [[image.resize((width, height)) for image in images] for images in im_list_2d]

    # Resize images based on minimum size
    else:
        min_width = min([min([img.width for img in imgs]) for imgs in im_list_2d])
        min_height = min([min([img.height for img in imgs]) for imgs in im_list_2d])
        for i, imgs in enumerate(images):
            for j, img in enumerate(imgs):
                if img.height > img.width:
                    images[i][j] = img.resize(
                        (int(img.width * min_height / img.height), min_height),
                        resample=0,
                    )
                elif img.width > img.height:
                    images[i][j] = img.resize(
                        (min_width, int(img.height * min_width / img.width)), resample=0
                    )
                else:
                    images[i][j] = img.resize((min_width, min_width))

    # Resize grid image
    images = [get_concat_h_multi_resize(im_list_h) for im_list_h in images]
    img = get_concat_v_multi_resize(images)
    height_percentage = image_height / float(img.size[1])
    adjusted_width = int((float(img.size[0]) * float(height_percentage)))
    img = img.resize((adjusted_width, image_height), Image.Resampling.LANCZOS)

    return img
