"""Tests for configurable nr_repr_docs, nr_samples, and recalculate_representative_docs().

Run from BERTopic repo root:
    pytest tests/test_configurable_repr_docs.py -v
"""

import copy

import pytest

from bertopic import BERTopic


# --- Constructor defaults ---


def test_default_nr_repr_docs_is_3(base_topic_model):
    """Default nr_repr_docs should be 3 (backward compatible)."""
    model = copy.deepcopy(base_topic_model)
    assert model.nr_repr_docs == 3


def test_default_nr_samples_is_500(base_topic_model):
    """Default nr_repr_docs_nr_samples should be 500 (backward compatible)."""
    model = copy.deepcopy(base_topic_model)
    assert model.nr_repr_docs_nr_samples == 500


# --- Constructor params wired through fit ---


@pytest.mark.parametrize("nr_repr_docs", [1, 3, 5, 10])
def test_custom_nr_repr_docs(nr_repr_docs, documents, document_embeddings, embedding_model):
    """Representative docs per topic should respect nr_repr_docs."""
    model = BERTopic(
        embedding_model=embedding_model,
        nr_repr_docs=nr_repr_docs,
    )
    model.fit(documents, embeddings=document_embeddings)

    assert model.representative_docs_
    for topic, docs in model.representative_docs_.items():
        assert len(docs) <= nr_repr_docs, f"Topic {topic} has {len(docs)} repr docs, expected <= {nr_repr_docs}"


@pytest.mark.parametrize("nr_samples", [10, 100, 1000])
def test_custom_nr_samples(nr_samples, documents, document_embeddings, embedding_model):
    """Model should accept custom nr_repr_docs_nr_samples."""
    model = BERTopic(
        embedding_model=embedding_model,
        nr_repr_docs_nr_samples=nr_samples,
    )
    model.fit(documents, embeddings=document_embeddings)

    assert hasattr(model, "representative_docs_")
    assert model.nr_repr_docs_nr_samples == nr_samples


def test_nr_samples_one(documents, document_embeddings, embedding_model):
    """nr_repr_docs_nr_samples=1 should work (minimal sampling pool)."""
    model = BERTopic(
        embedding_model=embedding_model,
        nr_repr_docs_nr_samples=1,
        nr_repr_docs=1,
    )
    model.fit(documents, embeddings=document_embeddings)

    assert model.representative_docs_
    for topic, docs_list in model.representative_docs_.items():
        assert len(docs_list) <= 1


# --- recalculate_representative_docs() ---


def test_recalculate_increases_repr_docs(documents, document_embeddings, embedding_model):
    """recalculate_representative_docs should update representative_docs_ count."""
    model = BERTopic(embedding_model=embedding_model)
    model.fit(documents, embeddings=document_embeddings)

    # Default is 3
    original = copy.deepcopy(model.representative_docs_)
    assert original

    # Recalculate with more
    model.recalculate_representative_docs(documents, nr_repr_docs=10)

    for topic, docs in model.representative_docs_.items():
        assert len(docs) <= 10
        # Topics with enough documents should have more than the original 3
        if len(docs) > 3:
            assert len(docs) > len(original.get(topic, []))


def test_recalculate_with_fewer_repr_docs(documents, document_embeddings, embedding_model):
    """recalculate_representative_docs with nr_repr_docs=1."""
    model = BERTopic(embedding_model=embedding_model)
    model.fit(documents, embeddings=document_embeddings)

    model.recalculate_representative_docs(documents, nr_repr_docs=1)

    for topic, docs in model.representative_docs_.items():
        assert len(docs) <= 1


def test_recalculate_uses_defaults(documents, document_embeddings, embedding_model):
    """recalculate_representative_docs without args uses constructor defaults."""
    model = BERTopic(
        embedding_model=embedding_model,
        nr_repr_docs=5,
        nr_repr_docs_nr_samples=200,
    )
    model.fit(documents, embeddings=document_embeddings)

    # Recalculate without explicit args — should use self.nr_repr_docs (5)
    model.recalculate_representative_docs(documents)

    for topic, docs in model.representative_docs_.items():
        assert len(docs) <= 5


def test_recalculate_custom_nr_samples(documents, document_embeddings, embedding_model):
    """recalculate_representative_docs should accept nr_samples."""
    model = BERTopic(embedding_model=embedding_model)
    model.fit(documents, embeddings=document_embeddings)

    # Small sampling pool
    model.recalculate_representative_docs(documents, nr_repr_docs=3, nr_samples=10)

    assert model.representative_docs_
    for topic, docs in model.representative_docs_.items():
        assert len(docs) <= 3
