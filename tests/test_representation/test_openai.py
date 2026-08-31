"""Unit tests for the OpenAI representation model."""

from unittest.mock import MagicMock

import pytest

# The OpenAI representation imports the ``openai`` package at module load time,
# so skip these tests entirely when it is not installed.
pytest.importorskip("openai")

from bertopic.representation._openai import OpenAI


def _make_client(content):
    """Build a mock OpenAI client whose chat completion returns ``content``."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]

    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


def _make_topic_model():
    """Build a mock topic model returning a single topic with representative docs."""
    topic_model = MagicMock()
    topic_model.verbose = False
    topic_model._extract_representative_docs.return_value = (
        {0: ["A document about pets.", "Another document about pets."]},
        None,
        None,
        None,
    )
    return topic_model


def test_openai_extract_topics_handles_none_content():
    """Regression test for #2353.

    When OpenAI returns a message whose ``content`` is ``None`` (for example when the
    response is blocked by the content filter and ``finish_reason='content_filter'``),
    the ``content`` field is still present, so the previous ``hasattr`` guard passed
    and ``None.strip()`` raised ``AttributeError``. The representation must instead
    fall back to a placeholder label.
    """
    representation = OpenAI(_make_client(content=None))

    updated_topics = representation.extract_topics(
        _make_topic_model(), documents=None, c_tf_idf=None, topics={0: [("pets", 0.9)]}
    )

    assert updated_topics == {0: [("No label returned", 1)]}


def test_openai_extract_topics_strips_normal_content():
    """A normal (non-None) response is stripped and the ``topic: `` prefix removed."""
    representation = OpenAI(_make_client(content="topic: Pets\n"))

    updated_topics = representation.extract_topics(
        _make_topic_model(), documents=None, c_tf_idf=None, topics={0: [("pets", 0.9)]}
    )

    assert updated_topics == {0: [("Pets", 1)]}
