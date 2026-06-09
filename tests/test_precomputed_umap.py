"""Tests for PR10: pre-computed UMAP embeddings support.

Run from BERTopic repo root:
    pytest tests/test_precomputed_umap.py -v
"""

from unittest.mock import patch

import numpy as np
import pytest

from bertopic import BERTopic


def test_fit_transform_with_precomputed_umap(documents, document_embeddings, reduced_embeddings, embedding_model):
    """fit_transform with umap_embeddings should skip UMAP and produce valid topics."""
    model = BERTopic(embedding_model=embedding_model)

    topics, _ = model.fit_transform(
        documents,
        embeddings=document_embeddings,
        umap_embeddings=reduced_embeddings,
    )

    assert len(topics) == len(documents)
    assert len(set(topics)) > 1  # Should find multiple topics
    assert hasattr(model, "topic_representations_")


def test_fit_with_precomputed_umap(documents, document_embeddings, reduced_embeddings, embedding_model):
    """Fit with umap_embeddings should work and be equivalent to fit_transform."""
    model = BERTopic(embedding_model=embedding_model)

    model.fit(
        documents,
        embeddings=document_embeddings,
        umap_embeddings=reduced_embeddings,
    )

    assert hasattr(model, "topics_")
    assert len(model.topics_) == len(documents)


def test_umap_not_called_when_precomputed(documents, document_embeddings, reduced_embeddings, embedding_model):
    """UMAP's fit_transform should not be called when umap_embeddings is provided."""
    model = BERTopic(embedding_model=embedding_model)

    with patch.object(model, "_reduce_dimensionality") as mock_reduce:
        model.fit_transform(
            documents,
            embeddings=document_embeddings,
            umap_embeddings=reduced_embeddings,
        )
        mock_reduce.assert_not_called()


def test_transform_with_precomputed_umap(documents, document_embeddings, reduced_embeddings, embedding_model):
    """Transform with umap_embeddings should skip UMAP transform."""
    # Fit a model with precomputed UMAP so HDBSCAN is trained on the right dimensions
    model = BERTopic(embedding_model=embedding_model)
    model.fit_transform(
        documents,
        embeddings=document_embeddings,
        umap_embeddings=reduced_embeddings,
    )

    topics, _ = model.transform(
        documents[:10],
        embeddings=document_embeddings[:10],
        umap_embeddings=reduced_embeddings[:10],
    )

    assert len(topics) == 10


def test_transform_umap_not_called_when_precomputed(
    documents, document_embeddings, reduced_embeddings, embedding_model
):
    """UMAP's transform should not be called when umap_embeddings is provided."""
    # Fit a model with precomputed UMAP
    model = BERTopic(embedding_model=embedding_model)
    model.fit_transform(
        documents,
        embeddings=document_embeddings,
        umap_embeddings=reduced_embeddings,
    )

    # Only test if model has a real umap_model
    if hasattr(model.umap_model, "transform"):
        with patch.object(model.umap_model, "transform") as mock_transform:
            model.transform(
                documents[:10],
                embeddings=document_embeddings[:10],
                umap_embeddings=reduced_embeddings[:10],
            )
            mock_transform.assert_not_called()


def test_results_consistent_with_internal_umap(documents, document_embeddings, embedding_model):
    """Pre-computed UMAP should give same results as internal computation."""
    from umap import UMAP

    umap_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric="cosine", random_state=42)

    # Model 1: internal UMAP
    model1 = BERTopic(embedding_model=embedding_model, umap_model=umap_model)
    topics1, _ = model1.fit_transform(documents, embeddings=document_embeddings)

    # Model 2: pre-computed UMAP (same embeddings)
    umap_embeddings = umap_model.fit_transform(document_embeddings)
    model2 = BERTopic(embedding_model=embedding_model, umap_model=umap_model)
    topics2, _ = model2.fit_transform(documents, embeddings=document_embeddings, umap_embeddings=umap_embeddings)

    # Results should be identical (same UMAP output)
    assert topics1 == topics2


def test_without_precomputed_umap_uses_internal(documents, document_embeddings, embedding_model):
    """Without umap_embeddings, should use internal UMAP as before."""
    model = BERTopic(embedding_model=embedding_model)
    model.umap_model.random_state = 42
    model.hdbscan_model.min_cluster_size = 3

    topics, _ = model.fit_transform(documents, embeddings=document_embeddings)
    # Should produce valid topics using internal UMAP
    assert len(topics) == len(documents)
    assert len(set(topics)) > 1


def test_wrong_shape_umap_embeddings_raises(documents, document_embeddings, embedding_model):
    """umap_embeddings with wrong number of rows should raise."""
    model = BERTopic(embedding_model=embedding_model)

    wrong_shape = np.random.rand(5, 2)  # 5 rows vs len(documents) rows
    with pytest.raises((ValueError, IndexError)):
        model.fit_transform(
            documents,
            embeddings=document_embeddings,
            umap_embeddings=wrong_shape,
        )
