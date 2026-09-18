"""Tests for turning non-text modalities into the words c-TF-IDF needs.

A converter is a representation model that also knows how to describe its own
modality in text. These tests use a stub captioner so the protocol, the corpus
reduction, and the plumbing are covered without downloading a captioning model.
"""

import os

import numpy as np
import pytest
from PIL import Image
from typing import ClassVar

import bertopic
from bertopic import BERTopic
from bertopic.cluster import BaseCluster
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.representation import MultiModalRepresentation
from bertopic._corpus import Corpus, Modality
from bertopic._topics import Media
from bertopic.representation._base import TextConverter


class StubCaptioner(TextConverter):
    """Caption images without loading a model, by naming the file."""

    modalities: ClassVar[set[Modality]] = {Modality.IMAGE}

    def to_text(self, corpus: Corpus) -> Corpus:
        documents = list(corpus.documents)
        for index, modality in enumerate(corpus.modality):
            if modality in self.modalities:
                documents[index] = f"a picture of {corpus.media[index]}"
        corpus.documents = documents
        return corpus


def image_corpus(nr_images: int = 4) -> Corpus:
    """An image-only corpus with the embeddings and assignments a converter expects."""
    return Corpus(
        media=[f"image_{index}.png" for index in range(nr_images)],
        modality=Modality.IMAGE,
        topics=np.zeros(nr_images, dtype=int),
        embeddings=np.eye(nr_images, 4),
    )


class StubTranscriber(TextConverter):
    """Transcribe audio without a model, so two converters can be seen composing."""

    modalities: ClassVar[set[Modality]] = {Modality.AUDIO}

    def to_text(self, corpus: Corpus) -> Corpus:
        documents = list(corpus.documents)
        for index, modality in enumerate(corpus.modality):
            if modality in self.modalities:
                documents[index] = f"someone saying {corpus.media[index]}"
        corpus.documents = documents
        return corpus


def mixed_corpus() -> Corpus:
    """Independent text, images and audio in one corpus, which is the interesting case."""
    corpus = Corpus.from_inputs(
        documents=["a report on rainfall", "notes on trains", "a review", "a letter"],
        images=["cat.png", "dog.png"],
        audio=["call.wav"],
    )
    corpus.topics = np.zeros(len(corpus.documents), dtype=int)
    corpus.embeddings = np.eye(len(corpus.documents), 4)
    return corpus


def test_a_converter_leaves_the_rest_of_the_corpus_alone():
    """Returning a subset would discard every document and every other modality."""
    topic_model = BERTopic(representation_model={"Visual": StubCaptioner()})

    corpus = topic_model._convert_media_to_text(mixed_corpus())

    assert len(corpus.documents) == 7
    assert [modality.value for modality in corpus.modality] == ["image", "image", "audio"] + ["text"] * 4
    assert corpus.documents[:2] == ["a picture of cat.png", "a picture of dog.png"]
    assert corpus.documents[3:] == ["a report on rainfall", "notes on trains", "a review", "a letter"]


def test_converters_compose():
    """Each fills only its own rows, so a corpus of several modalities is fully described."""
    topic_model = BERTopic(representation_model={"Visual": StubCaptioner(), "Audio": StubTranscriber()})

    corpus = topic_model._convert_media_to_text(mixed_corpus())

    assert corpus.documents[0] == "a picture of cat.png"
    assert corpus.documents[2] == "someone saying call.wav"
    assert corpus.documents[3] == "a report on rainfall"


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


def describing(prefix):
    """A stand-in model: it names what it was given, so the text is checkable."""
    return lambda items: [f"{prefix} {item}" for item in items]


def media_corpus(image: str = "cat.png") -> Corpus:
    """One topic holding an image, a clip and a video, which is what mixed media means."""
    corpus = Corpus.from_inputs(images=[image], audio=["call.wav"], video=["scene.mp4"])
    corpus.topics = np.zeros(3, dtype=int)
    corpus.embeddings = np.eye(3, 4)
    return corpus


def fit_documents_and(kind: str, media: list, nr_documents: int) -> BERTopic:
    """Fit unrelated documents and media as a topic each, the larger becoming topic 0."""
    topic_model = BERTopic(
        umap_model=BaseDimensionalityReduction(),
        hdbscan_model=BaseCluster(),
        representation_model={"Media": MultiModalRepresentation(model=describing("this is"))},
    )
    documents = [f"the quarterly budget report {index}" for index in range(nr_documents)]

    # Media rows come before documents in the corpus, so embeddings and labels follow that order
    embeddings = np.repeat(np.eye(2, 4), [len(media), nr_documents], axis=0)
    labels = [1] * len(media) + [0] * nr_documents
    return topic_model.fit(documents=documents, embeddings=embeddings, y=labels, **{kind: media})


def test_each_modality_is_described_by_its_own_model():
    """One model rarely captions and transcribes, so each modality may name its own."""
    converter = MultiModalRepresentation(
        model=describing("a picture of"),
        audio_model=describing("someone saying"),
        video_model=describing("a clip of"),
    )

    corpus = converter.to_text(media_corpus())

    assert corpus.documents == ["a picture of cat.png", "someone saying call.wav", "a clip of scene.mp4"]


def test_one_model_covers_every_modality():
    """A callable decides for itself, so it can serve all three without being named thrice."""
    converter = MultiModalRepresentation(model=describing("this is"))

    corpus = converter.to_text(media_corpus())

    assert corpus.documents == ["this is cat.png", "this is call.wav", "this is scene.mp4"]


def test_a_modality_without_a_model_is_left_alone():
    """Its rows keep an empty text channel rather than borrowing another modality's model."""
    converter = MultiModalRepresentation(audio_model=describing("someone saying"))

    corpus = converter.to_text(media_corpus())

    assert corpus.documents == ["", "someone saying call.wav", ""]


def test_models_are_loaded_only_when_their_modality_appears():
    """A name is loaded into its own task, so building all three upfront would fail on audio."""
    converter = MultiModalRepresentation("HuggingFaceTB/SmolVLM-256M-Instruct")

    assert converter.pipelines == {}


def test_a_topic_carries_every_modality_it_holds(image_paths):
    """A topic of photographs and voice notes is one representation carrying both."""
    converter = MultiModalRepresentation(model=describing("this is"))
    corpus = converter.to_text(media_corpus(image_paths[0]))

    representations = converter.extract_topics(BERTopic(verbose=False), corpus, {0: None}, None)

    assert set(representations[0].items) == {Modality.IMAGE, Modality.AUDIO, Modality.VIDEO}
    assert representations[0].images == [image_paths[0]]
    assert representations[0].collage is not None
    assert len(representations[0].captions) == 3


def test_every_topic_gets_media_even_without_any():
    """An empty representation rather than none, so a topic of text still fills the column."""
    corpus = Corpus.from_inputs(documents=["a report on rainfall", "notes on trains"], audio=["call.wav"])
    corpus.topics = np.array([0, 1, 1])
    corpus.embeddings = np.eye(3, 4)

    representations = MultiModalRepresentation().extract_topics(
        BERTopic(verbose=False), corpus, {0: None, 1: None}, None
    )

    assert representations[0].items == {Modality.AUDIO: ["call.wav"]}
    assert representations[1] == Media()


@pytest.mark.parametrize("nr_documents, nr_clips", [(3, 5), (5, 3)], ids=["clips_first", "documents_first"])
def test_representative_items_is_a_column_whichever_topic_comes_first(nr_documents, nr_clips):
    """The columns follow what was configured, not which topic the frequency sort puts first."""
    clips = [f"clip_{index}.wav" for index in range(nr_clips)]
    topic_model = fit_documents_and("audio", clips, nr_documents)

    info = topic_model.get_topic_info()

    assert "Media" not in info.columns
    assert sorted(len(items) for items in info["Representative_Items"]) == [0, nr_clips]
    assert [sorted(items) for items in topic_model.representative_items_.values()] == [sorted(clips)]


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_representative_items_hold_images_on_either_backend(backend):
    """Polars has no type for a list of images, so the column keeps them as the objects they are."""
    pytest.importorskip(backend)
    images = [Image.new("RGB", (40, 40), "red") for _ in range(3)]
    topic_model = fit_documents_and("images", images, nr_documents=5)

    bertopic.set_output(backend)
    try:
        info = topic_model.get_topic_info()
    finally:
        bertopic.set_output("pandas")

    assert sorted(len(items) for items in info["Representative_Items"]) == [0, 3]


@pytest.mark.parametrize("nr_documents, nr_images", [(3, 5), (5, 3)], ids=["images_first", "documents_first"])
def test_collages_survive_save_and_load_whichever_topic_has_them(nr_documents, nr_images, tmp_path):
    """Loading used to look for topic 0's collage, then expect every other topic to have one too."""
    images = [Image.new("RGB", (40, 40), "red") for _ in range(nr_images)]
    topic_model = fit_documents_and("images", images, nr_documents)

    topic_model.save(tmp_path, serialization="safetensors")
    loaded = BERTopic.load(tmp_path)

    assert topic_model.representative_images_
    assert sorted(loaded.representative_images_) == sorted(topic_model.representative_images_)


def test_a_model_without_collages_saves_no_images_folder(tmp_path):
    """Only a model with pictures to keep gets a folder for them."""
    topic_model = fit_documents_and("audio", ["clip_0.wav", "clip_1.wav"], nr_documents=3)

    topic_model.save(tmp_path, serialization="safetensors")

    assert not (tmp_path / "images").exists()


def test_a_converter_keeps_the_text_it_was_given():
    """A captioned clip is described already, so its caption is not replaced."""
    corpus = Corpus.from_inputs(documents=["a caller asks about a card"], audio=["call.wav"])
    corpus.topics = np.zeros(1, dtype=int)
    corpus.embeddings = np.eye(1, 4)

    corpus = MultiModalRepresentation(model=describing("someone saying")).to_text(corpus)

    assert corpus.documents == ["a caller asks about a card"]


def test_representative_docs_come_from_the_rows_that_were_described():
    """Sampling a big topic before dropping text-less rows kept about one of its nine captions."""
    clips = [f"clip_{index}.wav" for index in range(5000)]
    topic_model = fit_documents_and("audio", clips, nr_documents=3)

    representative_docs = topic_model.representative_docs_[next(iter(topic_model.representative_items_))]

    assert len(representative_docs) == 3
    assert all(representative_docs)


def test_video_only_input_yields_keywords():
    """The bar for 13c: a corpus of nothing but clips still has words to describe it."""
    clips = [f"clip_{index}.mp4" for index in range(6)]
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=BaseDimensionalityReduction(),
        hdbscan_model=BaseCluster(),
        representation_model={"Media": MultiModalRepresentation(video_model=describing("a video of"))},
    )
    embeddings = np.repeat(np.eye(2, 4), 3, axis=0)
    topic_model.fit(video=clips, embeddings=embeddings, y=[0, 0, 0, 1, 1, 1])

    assert all(topic_model.get_topic(topic) for topic in topic_model._topics.topic_ids())
    assert any("video" in word for word, _ in topic_model.get_topic(0))


@pytest.mark.skipif(
    not os.environ.get("BERTOPIC_MULTIMODAL_E2E"),
    reason="Set BERTOPIC_MULTIMODAL_E2E=1 to run; downloads CLIP and a captioning model",
)
def test_images_are_modelled_end_to_end(image_paths):
    """The documented image-only pipeline, with the real embedding and captioning models."""
    from bertopic.backend import MultiModalBackend
    from bertopic.representation import MultiModalRepresentation

    topic_model = BERTopic(
        embedding_model=MultiModalBackend("clip-ViT-B-32", batch_size=32),
        representation_model={"Media": MultiModalRepresentation("HuggingFaceTB/SmolVLM-256M-Instruct")},
        min_topic_size=2,
    )
    topic_model.fit(documents=None, images=image_paths)

    first_topic = topic_model._topics[topic_model._topics.topic_ids()[0]]

    # Captions reached c-TF-IDF, so the topic is described by real words
    assert any(word for word, _ in first_topic.representations["Main"].data)

    # The media aspect is a first-class representation, and its collage is exposed
    assert isinstance(first_topic.representations["Media"], Media)
    assert first_topic.representations["Media"].images
    assert topic_model.representative_images_
