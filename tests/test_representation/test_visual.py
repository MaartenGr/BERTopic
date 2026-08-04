import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from bertopic.representation import _visual as visual_module  # noqa: E402
from bertopic.representation import VisualRepresentation  # noqa: E402


def test_extract_topics_indexes_images_by_label_not_position(minimal_topic_model, monkeypatch):
    """`_extract_representative_docs` returns index labels (not positions) in
    `repr_docs_ids`. `VisualRepresentation.extract_topics` must look images up by
    those labels; using a non-contiguous/shifted DataFrame index catches any
    accidental positional lookup.
    """
    docs = [
        "doc alpha one",
        "doc beta two",
        "doc gamma three",
        "doc delta four",
        "doc epsilon five",
        "doc zeta six",
    ]
    topics_list = [0, 0, 0, 1, 1, 1]
    shifted_index = [50, 51, 52, 53, 54, 55]

    model, c_tf_idf, documents, topics = minimal_topic_model(docs, topics_list, index=shifted_index)
    # `_outliers` inspects `topic_sizes_` to know whether topic -1 is present.
    model.topic_sizes_ = {0: 3, 1: 3}

    # Attach a distinctive, non-string "Image" per document, keyed by its
    # (shifted) index label so we can verify which document each captured
    # image actually corresponds to.
    images_by_label = {}
    for label in documents.index:
        image = Image.new("RGB", (10, 10))
        image.info["label"] = label
        images_by_label[label] = image
    documents = documents.copy()
    documents["Image"] = [images_by_label[label] for label in documents.index]

    captured_images_to_combine = {}

    def fake_get_concat_tile_resize(im_list_2d, image_height=600, image_squares=False):
        # Flatten the 2D grid of images and remember which labels were passed in.
        captured_images_to_combine[current_topic[0]] = [image.info["label"] for row in im_list_2d for image in row]
        return Image.new("RGB", (10, 10))

    monkeypatch.setattr(visual_module, "get_concat_tile_resize", fake_get_concat_tile_resize)

    # `extract_topics` doesn't expose which topic is currently being processed
    # to `get_concat_tile_resize`, so track it via the tqdm loop order, which is
    # `sorted(topics.keys())` (see `_visual.py`).
    current_topic = [None]
    original_tqdm = visual_module.tqdm

    def fake_tqdm(iterable, *args, **kwargs):
        for item in iterable:
            current_topic[0] = item
            yield item

    monkeypatch.setattr(visual_module, "tqdm", fake_tqdm)

    representation_model = VisualRepresentation(nr_repr_images=3, nr_samples=500)
    representation_model.extract_topics(model, documents, c_tf_idf, topics)

    assert set(captured_images_to_combine.keys()) == {0, 1}
    for topic_id, labels in captured_images_to_combine.items():
        assert labels, f"no images captured for topic {topic_id}"
        for label in labels:
            assert documents.loc[label, "Topic"] == topic_id, (
                f"image with label {label} (topic {documents.loc[label, 'Topic']}) leaked into "
                f"topic {topic_id}'s collage"
            )

    monkeypatch.setattr(visual_module, "tqdm", original_tqdm)
