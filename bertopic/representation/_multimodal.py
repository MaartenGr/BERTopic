import numpy as np

from PIL import Image
from tqdm import tqdm
from scipy.sparse import csr_matrix
from transformers.pipelines import Pipeline, pipeline
from typing import Callable, ClassVar

from bertopic.representation._mmr import mmr
from bertopic.representation._base import TextConverter
from bertopic._corpus import Corpus, Modality
from bertopic._topics import Keywords, Media

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bertopic import BERTopic

# The transformers task that turns each modality into words. Image and video share one,
# since a video is described by describing a handful of its frames.
TASKS = {
    Modality.IMAGE: "image-text-to-text",
    Modality.VIDEO: "image-text-to-text",
    Modality.AUDIO: "automatic-speech-recognition",
}


class MultiModalRepresentation(TextConverter):
    """Represent topics by their media, and describe that media so c-TF-IDF has words.

    A topic that holds photographs and voice notes is one topic, so it gets one
    representation carrying both. The same sample of representative media serves twice:
    it is described in text, which is what gives a media-only corpus any keywords at all,
    and it is kept as the topic's `Media` representation.

    Describing every item would be prohibitive on a corpus of millions, so only
    `nr_repr_media` items per topic are described. MMR picks them near each topic's
    centroid while discarding near-duplicates.

    A modality without its own model is described by `model`. Note that a model *name*
    is loaded into that modality's own task, so one name rarely covers both captioning
    and transcription; a callable can, since it decides for itself. Models are loaded on
    first use, so a modality absent from the corpus costs nothing.

    Arguments:
        model: The model that describes media in words, for any modality without its own.
               Either the name of a model to load into a `transformers` pipeline, an
               already built pipeline, or any callable taking a list of items and
               returning one string each. Without one, topics are represented by their
               media alone.
        image_model: The model that describes only images.
        audio_model: The model that transcribes only audio.
        video_model: The model that describes only video.
        nr_repr_media: Number of representative media items to describe per topic.
        nr_frames: Number of frames to sample from each video, since one still cannot
                   stand for a clip.
        image_height: The height of the resulting collage.
        image_squares: Whether to resize each image in the collage to a square. This can
                       be visually more appealing if all input images are almost squares.
        prompt: What to ask the model for, when it is a pipeline rather than a callable.
        batch_size: The number of items to describe at a time.

    Usage:

    ```python
    from bertopic import BERTopic
    from bertopic.representation import MultiModalRepresentation

    # Media is an additional way of looking at a topic, so it is passed as an aspect
    representation_model = {
        "Media": MultiModalRepresentation(
            "HuggingFaceTB/SmolVLM-256M-Instruct", audio_model="openai/whisper-base"
        )
    }
    topic_model = BERTopic(representation_model=representation_model)
    ```
    """

    modalities: ClassVar[set[Modality]] = {Modality.IMAGE, Modality.AUDIO, Modality.VIDEO}

    def __init__(
        self,
        model: str | Pipeline | Callable | None = None,
        image_model: str | Pipeline | Callable | None = None,
        audio_model: str | Pipeline | Callable | None = None,
        video_model: str | Pipeline | Callable | None = None,
        nr_repr_media: int = 9,
        nr_frames: int = 3,
        image_height: int = 600,
        image_squares: bool = False,
        prompt: str = "Describe this image in one short sentence.",
        batch_size: int = 32,
    ):
        # A modality without its own model is described by the main one
        self.models = {}
        for modality, given in (
            (Modality.IMAGE, image_model),
            (Modality.AUDIO, audio_model),
            (Modality.VIDEO, video_model),
        ):
            given = model if given is None else given
            if given is not None and not isinstance(given, str) and not callable(given):
                raise ValueError(
                    "Please pass the name of a model, a transformers pipeline, or a callable "
                    "taking a list of items and returning one description each. For example:"
                    "MultiModalRepresentation('HuggingFaceTB/SmolVLM-256M-Instruct')"
                )
            self.models[modality] = given

        self.pipelines = {}
        self.nr_repr_media = nr_repr_media
        self.nr_frames = nr_frames
        self.image_height = image_height
        self.image_squares = image_squares
        self.prompt = prompt
        self.batch_size = batch_size

    def to_text(self, corpus: Corpus) -> Corpus:
        """Describe a sample of each topic's media, writing into those rows' text channel.

        The corpus keeps its shape. Only sampled rows without text gain some, so documents,
        captions the user gave, and modalities this model does not handle are left exactly
        as they were, and several converters can run one after another without overwriting
        each other.
        """
        documents = list(corpus.documents)
        for modality in self.modalities:
            if self.models[modality] is None:
                continue

            # Describe a sample of every topic at once, so the model batches them. A row that
            # already has text, such as a captioned image, keeps the words it was given
            rows = [
                row
                for topic in corpus.topic_ids()
                for row in self._sample(corpus, topic, modality)
                if not corpus.documents[row]
            ]
            if not rows:
                continue

            described = self._convert_media_to_text([corpus.media[row] for row in rows], modality)
            for row, description in zip(rows, described):
                documents[row] = description

        corpus.documents = documents
        return corpus

    def extract_topics(
        self,
        topic_model: "BERTopic",
        corpus: Corpus,
        topic_representations: dict[int, Keywords],
        c_tf_idf: csr_matrix,
        embeddings: np.ndarray = None,
    ) -> dict[int, Media]:
        """Collect each topic's representative media into a single representation.

        Arguments:
            topic_model: A BERTopic model
            corpus: The input documents including (calculated) embeddings
            topic_representations: The candidate topic representations
            c_tf_idf: The topic c-TF-IDF representation (unused, for API compatibility)
            embeddings: Pre-trained document embeddings (unused, for API compatibility)

        Returns:
            A `Media` representation per topic, carrying its media and their descriptions,
            and empty for a topic without media so every topic has the same columns
        """
        representations = {}
        for topic in tqdm(sorted(topic_representations), disable=not topic_model.verbose):
            # The same rows `to_text` described, since MMR over unchanged embeddings repeats. Sorted,
            # since a set's order changes between runs and a topic should list its media the same way
            rows = {modality: self._sample(corpus, topic, modality) for modality in sorted(self.modalities)}
            rows = {modality: found for modality, found in rows.items() if found}

            items = {modality: [corpus.media[row] for row in found] for modality, found in rows.items()}
            representations[topic] = Media(
                items=items,
                collage=self._collage(items.get(Modality.IMAGE, [])),
                captions=[
                    corpus.documents[row] for found in rows.values() for row in found if corpus.documents[row]
                ],
            )

        return representations

    def _sample(self, corpus: Corpus, topic: int, modality: Modality) -> list[int]:
        """The rows of one modality that best represent one topic, without near-duplicates."""
        rows = np.array(
            [
                index
                for index, (value, assigned) in enumerate(zip(corpus.modality, corpus.topics))
                if value == modality and assigned == topic
            ]
        )
        if not len(rows):
            return []

        # Pick a diverse sample near the topic's centroid
        centroid = corpus.embeddings[rows].mean(axis=0).reshape(1, -1)
        return list(
            mmr(
                centroid,
                corpus.embeddings[rows],
                rows,
                top_n=min(self.nr_repr_media, len(rows)),
                diversity=0.1,
            )
        )

    def _convert_media_to_text(self, items: list, modality: Modality) -> list[str]:
        """Describe media of one modality, with the model that handles it.

        Arguments:
            items: A list of media items, all of the same modality.
            modality: Which modality the items are.

        Returns:
            List of descriptions
        """
        model = self._model_for(modality)
        if not isinstance(model, Pipeline):
            return model(items)

        if modality == Modality.AUDIO:
            return [spoken["text"].strip() for spoken in model(items, batch_size=self.batch_size)]

        if modality == Modality.IMAGE:
            images = [self._open(image) for image in items]
            captions = self._caption(images, model)
            for image in images:
                image.close()
            return captions

        # A video is described by describing the frames sampled from it, joined back up
        frames = [frame for video in items for frame in self._frames(video)]
        captions = self._caption(frames, model)
        return [
            " ".join(captions[start : start + self.nr_frames])
            for start in range(0, len(captions), self.nr_frames)
        ]

    def _caption(self, images: list, model: Pipeline) -> list[str]:
        """Caption opened images through a vision-language pipeline."""
        # One chat turn per image, which is how `image-text-to-text` is served
        messages = [
            [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": image}, {"type": "text", "text": self.prompt}],
                }
            ]
            for image in images
        ]

        # A caption, not an essay: c-TF-IDF wants the words rather than the prose
        outputs = model(text=messages, max_new_tokens=30, batch_size=self.batch_size)
        return [output[0]["generated_text"][-1]["content"].strip() for output in outputs]

    def _collage(self, images: list) -> Image.Image | None:
        """Tile a topic's images three to a row, so one picture stands for the topic."""
        if not images:
            return None

        opened = [self._open(image) for image in images]
        tiles = [opened[start : start + 3] for start in range(0, len(opened), 3)]
        collage = get_concat_tile_resize(tiles, self.image_height, self.image_squares)

        for image in opened:
            image.close()
        return collage

    def _model_for(self, modality: Modality):
        """Load a modality's model on first use, so an absent modality costs nothing."""
        model = self.models[modality]
        if not isinstance(model, str):
            return model

        # Image and video share a task, so one pipeline serves both when the name matches
        key = (TASKS[modality], model)
        if key not in self.pipelines:
            self.pipelines[key] = pipeline(key[0], model=key[1])
        return self.pipelines[key]

    def _frames(self, video) -> list:
        """Sample frames evenly across a clip, since one still cannot stand for a video."""
        # Imported here because decoding video is the only thing that needs it, and video is
        # an extra: describing images and audio must work without it installed
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(video)
        last = decoder.metadata.num_frames - 1
        indices = [round(index * last / max(self.nr_frames - 1, 1)) for index in range(self.nr_frames)]
        return [Image.fromarray(decoder[index].permute(1, 2, 0).numpy()) for index in indices]

    @staticmethod
    def _open(image):
        """Open an image from a path, or copy one that is already loaded."""
        return Image.open(image) if isinstance(image, str) else image.copy()


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
