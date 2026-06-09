"""Tests for FeatureImportance representation model."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from bertopic.representation._feature_importance import FeatureImportance
from scipy.sparse import csr_matrix


@pytest.fixture
def mock_topic_model():
    """Create a mock BERTopic model with basic attributes."""
    model = MagicMock()
    model._outliers = 1
    model.topic_embeddings_ = np.random.rand(4, 10)  # 1 outlier + 3 topics
    model.vectorizer_model.get_feature_names_out.return_value = np.array(["alpha", "beta", "gamma", "delta", "epsilon"])
    model._preprocess_text.side_effect = lambda x: x
    # Create a sparse document-term matrix
    dtm = csr_matrix(
        np.array(
            [
                [2, 1, 0, 0, 1],
                [3, 0, 1, 0, 0],
                [0, 0, 2, 1, 0],
                [0, 1, 3, 0, 1],
                [1, 0, 0, 2, 0],
                [0, 0, 0, 3, 1],
            ]
        )
    )
    model.vectorizer_model.transform.return_value = dtm
    # Embedding model for centroid_distance
    model.embedding_model.embed_words.return_value = np.random.rand(5, 10)
    return model


@pytest.fixture
def sample_documents():
    """Sample documents DataFrame."""
    return pd.DataFrame(
        {
            "Document": ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"],
            "Topic": [0, 0, 1, 1, 2, 2],
        }
    )


@pytest.fixture
def c_tf_idf():
    """Sample c-TF-IDF matrix."""
    return csr_matrix(np.random.rand(3, 5))


@pytest.fixture
def default_topics():
    """Default topic representations."""
    return {
        0: [("alpha", 0.5), ("beta", 0.3), ("epsilon", 0.2)],
        1: [("gamma", 0.6), ("delta", 0.2), ("beta", 0.1)],
        2: [("delta", 0.7), ("epsilon", 0.2), ("alpha", 0.1)],
    }


class TestFeatureImportanceInit:
    def test_default_params(self):
        fi = FeatureImportance()
        assert fi.method == "fighting_words"
        assert fi.top_n_words == 10
        assert fi.prior == "corpus"

    def test_custom_params(self):
        fi = FeatureImportance(method="centroid_distance", top_n_words=5, prior=0.1)
        assert fi.method == "centroid_distance"
        assert fi.top_n_words == 5
        assert fi.prior == 0.1


class TestFightingWords:
    def test_returns_correct_format(self, mock_topic_model, sample_documents, c_tf_idf, default_topics):
        fi = FeatureImportance(method="fighting_words", top_n_words=3)
        result = fi.extract_topics(mock_topic_model, sample_documents, c_tf_idf, default_topics)

        assert isinstance(result, dict)
        assert set(result.keys()) == {0, 1, 2}
        for topic_id, words in result.items():
            assert len(words) == 3
            for word, score in words:
                assert isinstance(word, str)
                assert isinstance(score, float)

    def test_corpus_prior(self, mock_topic_model, sample_documents, c_tf_idf, default_topics):
        fi = FeatureImportance(method="fighting_words", top_n_words=3, prior="corpus")
        result = fi.extract_topics(mock_topic_model, sample_documents, c_tf_idf, default_topics)
        assert len(result) == 3

    def test_numeric_prior(self, mock_topic_model, sample_documents, c_tf_idf, default_topics):
        fi = FeatureImportance(method="fighting_words", top_n_words=3, prior=0.01)
        result = fi.extract_topics(mock_topic_model, sample_documents, c_tf_idf, default_topics)
        assert len(result) == 3

    def test_scores_are_zscores(self, mock_topic_model, sample_documents, c_tf_idf, default_topics):
        """Fighting words produces z-scores that can be positive or negative."""
        fi = FeatureImportance(method="fighting_words", top_n_words=5)
        result = fi.extract_topics(mock_topic_model, sample_documents, c_tf_idf, default_topics)
        # Top words should have positive z-scores (distinguishing)
        for topic_id, words in result.items():
            top_score = words[0][1]
            assert top_score > 0


class TestCentroidDistance:
    def test_returns_correct_format(self, mock_topic_model, sample_documents, c_tf_idf, default_topics):
        fi = FeatureImportance(method="centroid_distance", top_n_words=3)
        result = fi.extract_topics(mock_topic_model, sample_documents, c_tf_idf, default_topics)

        assert isinstance(result, dict)
        assert set(result.keys()) == {0, 1, 2}
        for topic_id, words in result.items():
            assert len(words) == 3

    def test_no_embedding_model_raises(self, mock_topic_model, sample_documents, c_tf_idf, default_topics):
        mock_topic_model.embedding_model = None
        fi = FeatureImportance(method="centroid_distance")
        with pytest.raises(ValueError, match="centroid_distance requires an embedding model"):
            fi.extract_topics(mock_topic_model, sample_documents, c_tf_idf, default_topics)

    def test_fallback_to_embed_documents(self, mock_topic_model, sample_documents, c_tf_idf, default_topics):
        mock_topic_model.embedding_model.embed_words.return_value = None
        mock_topic_model.embedding_model.embed_documents.return_value = np.random.rand(5, 10)
        fi = FeatureImportance(method="centroid_distance", top_n_words=3)
        result = fi.extract_topics(mock_topic_model, sample_documents, c_tf_idf, default_topics)
        assert len(result) == 3


class TestInvalidMethod:
    def test_unknown_method_raises(self, mock_topic_model, sample_documents, c_tf_idf, default_topics):
        fi = FeatureImportance(method="nonexistent")
        with pytest.raises(ValueError, match="Unknown method"):
            fi.extract_topics(mock_topic_model, sample_documents, c_tf_idf, default_topics)


class TestEdgeCases:
    def test_single_topic(self, mock_topic_model, c_tf_idf, default_topics):
        documents = pd.DataFrame(
            {
                "Document": ["doc1", "doc2"],
                "Topic": [0, 0],
            }
        )
        dtm = csr_matrix(np.array([[2, 1, 0, 0, 1], [3, 0, 1, 0, 0]]))
        mock_topic_model.vectorizer_model.transform.return_value = dtm

        fi = FeatureImportance(method="fighting_words", top_n_words=3)
        result = fi.extract_topics(mock_topic_model, documents, c_tf_idf, default_topics)
        assert 0 in result

    def test_top_n_words_exceeds_vocab(self, mock_topic_model, sample_documents, c_tf_idf, default_topics):
        fi = FeatureImportance(method="fighting_words", top_n_words=100)
        result = fi.extract_topics(mock_topic_model, sample_documents, c_tf_idf, default_topics)
        # Should return all 5 words (vocab size)
        for topic_id, words in result.items():
            assert len(words) == 5


class TestFeatureImportanceIntegration:
    """Integration test with a real fitted model."""

    def test_as_representation_model(self, base_topic_model, documents, document_embeddings, embedding_model):
        """FeatureImportance should work as a representation model in the BERTopic pipeline."""
        from bertopic import BERTopic

        fi = FeatureImportance(method="fighting_words", top_n_words=5)
        model = BERTopic(
            embedding_model=embedding_model,
            representation_model={"Main": fi},
        )
        model.umap_model.random_state = 42
        model.hdbscan_model.min_cluster_size = 3
        model.fit(documents, embeddings=document_embeddings)

        # Should produce topic representations
        assert hasattr(model, "topic_representations_")
        for topic_id, words in model.topic_representations_.items():
            if topic_id != -1:
                assert len(words) > 0

    def test_as_aspect_model(self, base_topic_model, documents, document_embeddings):
        """FeatureImportance should integrate with topic_aspects_ via multi-aspect."""
        import copy

        model = copy.deepcopy(base_topic_model)

        fi = FeatureImportance(method="fighting_words", top_n_words=5)
        result = fi.extract_topics(
            model,
            pd.DataFrame({"Document": documents, "Topic": model.topics_}),
            model.c_tf_idf_,
            model.topic_representations_,
        )

        assert isinstance(result, dict)
        assert len(result) > 0
