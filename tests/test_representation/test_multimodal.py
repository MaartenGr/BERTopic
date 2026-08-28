"""Tests for turning non-text modalities into the words c-TF-IDF needs.

A converter is a representation model that also knows how to describe its own
modality in text. These tests use a stub captioner so the protocol, the corpus
reduction, and the plumbing are covered without downloading a captioning model.
"""

import os

import numpy as np
import pytest

from bertopic import BERTopic
from bertopic._corpus import Corpus, Modality
from bertopic._topics import Images
from bertopic.representation._base import TextConverter


class StubCaptioner(TextConverter):
    """Caption images without loading a model, by naming the file."""

    def __init__(self):
        self.modality = Modality.IMAGE

    def to_text(self, corpus: Corpus) -> Corpus:
        rows = [index for index, modality in enumerate(corpus.modality) if modality == self.modality]
        if not rows:
            return corpus

        captioned = corpus.get_corpus_by_indices(rows)
        captioned.documents = [f"a picture of {image}" for image in captioned.media]
        return captioned


def image_corpus(nr_images: int = 4) -> Corpus:
    """An image-only corpus with the embeddings and assignments a converter expects."""
    return Corpus(
        media=[f"image_{index}.png" for index in range(nr_images)],
        modality=Modality.IMAGE,
        topics=np.zeros(nr_images, dtype=int),
        embeddings=np.eye(nr_images, 4),
    )


def test_a_converter_fills_the_text_channel():
    """Rows that were only images come back with words for c-TF-IDF to read."""
    topic_model = BERTopic(representation_model={"Visual": StubCaptioner()})

    corpus = topic_model._convert_media_to_text(image_corpus())

    assert corpus.documents == [f"a picture of image_{index}.png" for index in range(4)]


def test_a_converter_leaves_text_corpora_alone():
    """A converter whose modality is absent must not touch the corpus."""
    topic_model = BERTopic(representation_model={"Visual": StubCaptioner()})
    corpus = Corpus(documents=["first", "second"])

    assert topic_model._convert_media_to_text(corpus).documents == ["first", "second"]


def test_models_without_the_protocol_are_ignored():
    """Only models declaring the converter contract are asked to convert."""
    from bertopic.representation import KeyBERTInspired

    topic_model = BERTopic(representation_model=KeyBERTInspired())
    corpus = image_corpus()

    assert topic_model._convert_media_to_text(corpus).documents == ["", "", "", ""]


@pytest.mark.parametrize(
    "representation_model, expected",
    [
        (None, 0),
        (StubCaptioner(), 1),
        ([StubCaptioner(), StubCaptioner()], 2),
        ({"Main": StubCaptioner(), "Visual": [StubCaptioner(), StubCaptioner()]}, 3),
    ],
    ids=["none", "single", "list", "dict_with_nested_list"],
)
def test_every_configuration_shape_is_flattened(representation_model, expected):
    """Representation models may be passed alone, in a list, or per aspect."""
    topic_model = BERTopic(representation_model=representation_model)

    assert len(topic_model._flatten_representation_models()) == expected


@pytest.mark.skipif(
    not os.environ.get("BERTOPIC_MULTIMODAL_E2E"),
    reason="Set BERTOPIC_MULTIMODAL_E2E=1 to run; downloads CLIP and a captioning model",
)
def test_images_are_modelled_end_to_end(image_paths):
    """The documented image-only pipeline, with the real embedding and captioning models."""
    from bertopic.backend import MultiModalBackend
    from bertopic.representation import VisualRepresentation

    topic_model = BERTopic(
        embedding_model=MultiModalBackend("clip-ViT-B-32", batch_size=32),
        representation_model={
            "Visual_Aspect": VisualRepresentation(image_to_text_model="nlpconnect/vit-gpt2-image-captioning")
        },
        min_topic_size=2,
    )
    topic_model.fit(documents=None, images=image_paths)

    first_topic = topic_model._topics[topic_model._topics.topic_ids()[0]]

    # Captions reached c-TF-IDF, so the topic is described by real words
    assert any(word for word, _ in first_topic.representations["Main"].data)

    # The visual aspect is a first-class representation, and its images are exposed
    assert isinstance(first_topic.representations["Visual_Aspect"], Images)
    assert topic_model.representative_images_
