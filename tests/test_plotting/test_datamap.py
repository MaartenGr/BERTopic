"""The datamap's hover: every point's text, and a thumbnail for images and videos."""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import bertopic.plotting._datamap as datamap
from bertopic import BERTopic
from bertopic.cluster import BaseCluster
from bertopic.dimensionality import BaseDimensionalityReduction

DOCUMENTS = [
    "the quarterly <b>budget</b> report",
    "the budget for next year",
    "a report on costs",
    "costs rose",
]


@pytest.fixture
def hover(monkeypatch):
    """What the datamap hands to datamapplot's interactive plot, kept instead of drawn.

    This checks BERTopic's side of the hand-over without datamapplot, an extra that CI does not install.
    """
    handed_over = {}
    stand_in = SimpleNamespace(
        create_interactive_plot=lambda *layers, **settings: handed_over.update(settings)
    )
    monkeypatch.setattr(datamap, "datamapplot", stand_in, raising=False)
    monkeypatch.setattr(datamap, "HAS_DATAMAPPLOT", True)
    return handed_over


@pytest.fixture
def image_paths(tmp_path):
    """Three images on disk, which is how the docs advise passing them."""
    paths = [str(tmp_path / f"{index}.png") for index in range(3)]
    for path in paths:
        Image.new("RGB", (60, 40), "red").save(path)
    return paths


def fitted(images: list | None = None) -> BERTopic:
    """A topic of the images, when there are any, and a topic of the documents."""
    nr_images = len(images) if images else 0
    embeddings = np.repeat(np.eye(2, 4), [nr_images, len(DOCUMENTS)], axis=0)
    labels = [1] * nr_images + [0] * len(DOCUMENTS)
    topic_model = BERTopic(umap_model=BaseDimensionalityReduction(), hdbscan_model=BaseCluster())
    return topic_model.fit(DOCUMENTS, images=images, embeddings=embeddings, y=labels)


def test_images_show_as_thumbnails_above_their_path(hover, image_paths):
    """Media rows come first, as in `fit`, and an image without a caption is named by its path."""
    topic_model = fitted(image_paths)

    topic_model.visualize_document_datamap(
        DOCUMENTS, images=image_paths, reduced_embeddings=np.zeros((7, 2)), interactive=True
    )
    thumbnails = list(hover["extra_point_data"]["thumbnail"])

    assert all(thumbnail.startswith('<img src="data:image/jpeg;base64,') for thumbnail in thumbnails[:3])
    assert thumbnails[3:] == [""] * len(DOCUMENTS)
    assert hover["hover_text"][:3] == image_paths
    assert hover["hover_text_html_template"] == "{thumbnail}<div>{hover_text}</div>"


def test_text_is_escaped_once_the_hover_holds_pictures(hover, image_paths):
    """The hover becomes HTML, where a document's own markup must still read as text."""
    topic_model = fitted(image_paths)

    topic_model.visualize_document_datamap(
        DOCUMENTS, images=image_paths, reduced_embeddings=np.zeros((7, 2)), interactive=True
    )

    assert hover["hover_text"][3] == "the quarterly &lt;b&gt;budget&lt;/b&gt; report"


def test_hover_settings_passed_in_take_precedence(hover, image_paths):
    """A template of your own replaces the one made here, rather than clashing with it."""
    topic_model = fitted(image_paths)

    topic_model.visualize_document_datamap(
        DOCUMENTS,
        images=image_paths,
        reduced_embeddings=np.zeros((7, 2)),
        interactive=True,
        int_datamap_kwds={"hover_text_html_template": "<b>{hover_text}</b>"},
    )

    assert hover["hover_text_html_template"] == "<b>{hover_text}</b>"


def test_documents_alone_keep_their_plain_hover(hover):
    """Without images or video nothing changes: the documents are the hover, as they always were."""
    topic_model = fitted()

    topic_model.visualize_document_datamap(DOCUMENTS, reduced_embeddings=np.zeros((4, 2)), interactive=True)

    assert hover["hover_text"] == DOCUMENTS
    assert "extra_point_data" not in hover
