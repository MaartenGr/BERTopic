"""Contract tests for the multimodal embedding backend.

`MultiModalBackend` routes each modality to the model that handles it, falling back to the
main model when a modality has none of its own. These tests swap recording stubs into that
routing table so the dispatch is covered without loading a model per modality.
"""

import copy

import numpy as np
import pytest

from bertopic import BERTopic
from bertopic._corpus import Corpus, Modality
from bertopic.backend import MultiModalBackend
from bertopic.backend._sentencetransformers import SentenceTransformerBackend


class RecordingModel:
    """Stands in for a sentence-transformers model and remembers what it was asked to embed."""

    def __init__(self, value: float = 0.0, dimension: int = 4, modalities: tuple = ("text",)):
        self.value = value
        self.dimension = dimension
        self.modalities = list(modalities)
        self.seen = []

    def encode(self, items, **kwargs):
        self.seen.append(list(items))
        self.kwargs = kwargs
        return np.full((len(items), self.dimension), self.value)


def test_a_modality_without_its_own_model_uses_the_main_one(embedding_model):
    """One joint model is the common case, and the only one that shares a vector space."""
    backend = MultiModalBackend(embedding_model)

    assert set(backend.models) == set(Modality)
    assert all(model is embedding_model for model in backend.models.values())


def test_a_per_modality_model_overrides_the_main_one(embedding_model):
    """Naming a model per modality is how a single-modality corpus gets a specialist."""
    audio_model = copy.deepcopy(embedding_model)
    backend = MultiModalBackend(embedding_model, audio_model=audio_model)

    assert backend.models[Modality.AUDIO] is audio_model
    assert backend.models[Modality.VIDEO] is embedding_model


def test_each_modality_reaches_its_own_model(embedding_model):
    """The whole point of the dispatcher: rows go to the encoder that understands them."""
    backend = MultiModalBackend(embedding_model)
    backend.models = {modality: RecordingModel() for modality in Modality}

    backend.embed_media(["clip.wav"], Modality.AUDIO)
    backend.embed_media(["scene.mp4"], Modality.VIDEO)
    backend.embed_media(["print(1)"], Modality.CODE)

    assert backend.models[Modality.AUDIO].seen == [["clip.wav"]]
    assert backend.models[Modality.VIDEO].seen == [["scene.mp4"]]
    assert backend.models[Modality.CODE].seen == [["print(1)"]]
    assert backend.models[Modality.IMAGE].seen == []


def test_embedding_documents_goes_through_the_text_model(embedding_model):
    """`embed_documents` is the contract every backend shares, so it must still route."""
    backend = MultiModalBackend(embedding_model)
    backend.models = {modality: RecordingModel() for modality in Modality}

    backend.embed_documents(["first", "second"])

    assert backend.models[Modality.TEXT].seen == [["first", "second"]]


@pytest.mark.parametrize("modality", [Modality.AUDIO, "audio"], ids=["enum", "string"])
def test_media_a_model_takes_is_passed_as_media(embedding_model, modality):
    """A path could be read as text, and the clustering prompt and the length limit are for text only."""
    backend = MultiModalBackend(embedding_model)
    backend.models[Modality.AUDIO] = RecordingModel(modalities=("text", "audio"))

    # The embeddings guide names the modality as a plain string
    backend.embed_media(["clip.m4a"], modality)

    model = backend.models[Modality.AUDIO]
    assert model.seen == [[{"audio": "clip.m4a"}]]
    assert model.kwargs["prompt"] == ""
    assert model.kwargs["processing_kwargs"] == {"text": {"truncation": False}}


def test_media_without_text_needs_a_model_that_embeds_it(embedding_model, image_paths):
    """A text-only model would embed every image as an empty string, so they would all look alike."""
    with pytest.raises(ValueError, match="google/embeddinggemma-2"):
        BERTopic(embedding_model=embedding_model).fit(images=image_paths)

    # Captioned media is embedded through its captions, as on 0.17
    topic_model = BERTopic(embedding_model=SentenceTransformerBackend(embedding_model))
    captioned = Corpus.from_inputs(documents=["a cat", "a dog"], images=image_paths[:2])
    assert topic_model._embed_corpus(captioned).shape == (2, embedding_model.get_embedding_dimension())


def test_documents_reach_the_model_whole(embedding_model):
    """Truncation is the model's job; the backend used to crop everything to CLIP's 77."""
    backend = MultiModalBackend(embedding_model)
    backend.models[Modality.TEXT] = RecordingModel()
    long_document = " ".join(f"word{number}" for number in range(200))

    backend.embed_documents([long_document])

    assert backend.models[Modality.TEXT].seen == [[long_document]]


def test_text_only_input_does_not_take_the_image_branch(embedding_model):
    """`Corpus.images` is an empty list rather than None, which used to reach `np.mean`."""
    topic_model = BERTopic(embedding_model=MultiModalBackend(embedding_model))

    embeddings = topic_model._embed_corpus(Corpus.from_inputs(documents=["first", "second"]))

    assert embeddings.shape == (2, embedding_model.get_embedding_dimension())


def test_captioned_media_averages_both_channels(embedding_model, image_paths):
    """A captioned image is half text, which is what `documents=` beside `images=` means."""
    backend = MultiModalBackend(embedding_model)
    backend.models[Modality.TEXT] = RecordingModel(value=1.0)
    backend.models[Modality.IMAGE] = RecordingModel(value=3.0)
    topic_model = BERTopic(embedding_model=backend)

    averaged = topic_model._embed_corpus(Corpus.from_inputs(documents=["a cat"], images=image_paths[:1]))

    assert np.allclose(averaged, 2.0)


def test_rows_are_embedded_by_modality_and_kept_in_corpus_order(embedding_model, image_paths):
    """Grouping must not disturb row order, or every topic assignment silently shifts."""
    backend = MultiModalBackend(embedding_model)
    values = {
        Modality.TEXT: 1.0,
        Modality.CODE: 2.0,
        Modality.IMAGE: 3.0,
        Modality.AUDIO: 4.0,
        Modality.VIDEO: 5.0,
    }
    backend.models = {modality: RecordingModel(value=value) for modality, value in values.items()}
    topic_model = BERTopic(embedding_model=backend)

    corpus = Corpus.from_inputs(
        documents=["some text"], images=image_paths[:1], audio=["clip.wav"], code=["print(1)"]
    )
    embeddings = topic_model._embed_corpus(corpus)

    # `from_inputs` lays rows out as images, then audio, then video, then text, then code
    assert [modality.value for modality in corpus.modality] == ["image", "audio", "text", "code"]
    assert [row[0] for row in embeddings] == [3.0, 4.0, 1.0, 2.0]
