"""The media page: every summary a topic holds, shown or played beside its name."""

import base64
import io
import re

import numpy as np
from IPython.display import HTML
from PIL import Image

from bertopic import BERTopic
from bertopic.cluster import BaseCluster
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.representation import MultiModalRepresentation
from bertopic.representation._multimodal import SAMPLING_RATE


def media_model() -> BERTopic:
    """A topic of images, a topic of audio and a topic of text, fitted without any real model."""
    topic_model = BERTopic(
        umap_model=BaseDimensionalityReduction(),
        hdbscan_model=BaseCluster(),
        representation_model={
            "Media": MultiModalRepresentation(model=lambda items: ["a described item"] * len(items))
        },
    )
    images = [Image.new("RGB", (400, 300), "red") for _ in range(3)]
    clips = [np.zeros(SAMPLING_RATE) for _ in range(4)]
    documents = [f"the quarterly budget report {index}" for index in range(5)]

    # Media rows come first in the corpus, images before audio, so labels follow that order
    embeddings = np.repeat(np.eye(3, 4), [3, 4, 5], axis=0)
    labels = [2] * 3 + [1] * 4 + [0] * 5
    return topic_model.fit(documents=documents, images=images, audio=clips, embeddings=embeddings, y=labels)


def test_every_summary_is_on_the_page():
    """A collage shows as a picture and a montage as a player, each captioned with what it is."""
    page = media_model().visualize_media().data

    assert page.count('<img src="data:image/jpeg;base64,') == 1
    assert page.count('<audio controls src="data:audio/wav;base64,') == 1
    assert "<figcaption>Collage</figcaption>" in page
    assert "<figcaption>Montage</figcaption>" in page


def test_a_topic_without_media_keeps_its_row():
    """The topic of text still gets a row, with an empty media cell, so no topic goes missing."""
    page = media_model().visualize_media().data

    assert page.count("<tr><td") == 3
    assert "<td></td></tr>" in page


def test_topics_choose_the_rows():
    """`topics` and `top_n_topics` work as they do in every other plot."""
    topic_model = media_model()

    assert topic_model.visualize_media(topics=[0]).data.count("<tr><td") == 1
    assert topic_model.visualize_media(top_n_topics=2).data.count("<tr><td") == 2


def test_pictures_are_shrunk_to_the_height_asked_for():
    """A 600-pixel collage would bloat the page, so each picture is embedded at `height`."""
    page = media_model().visualize_media(height=50).data

    encoded = re.search(r'data:image/jpeg;base64,([^"]+)', page).group(1)

    assert Image.open(io.BytesIO(base64.b64decode(encoded))).height == 50


def test_names_are_escaped():
    """A label is shown as text, never read as markup."""
    topic_model = media_model()
    topic_model.set_topic_labels({0: "<b>budget</b>"})

    page = topic_model.visualize_media().data

    assert "&lt;b&gt;budget&lt;/b&gt;" in page
    assert "<b>budget</b>" not in page


def test_the_page_is_html_that_a_notebook_shows():
    """It is IPython's `HTML`, like any other page a notebook renders inline."""
    assert isinstance(media_model().visualize_media(), HTML)
