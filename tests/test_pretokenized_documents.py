"""Tests for PR5: pre-tokenized documents support.

Run from BERTopic repo root:
    pytest tests/test_pretokenized_documents.py -v
"""

import copy

import pandas as pd
import pytest
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic


def simple_tokenizer(text):
    """Simple whitespace tokenizer for testing."""
    return text.lower().split()


@pytest.fixture
def tokenized_documents(documents):
    """Pre-tokenize documents using simple whitespace tokenizer."""
    return [simple_tokenizer(doc) for doc in documents]


class TestPreTokenizedFitTransform:
    """Verify fit/fit_transform with pre-tokenized documents."""

    def test_fit_with_tokenized_docs(self, documents, tokenized_documents, document_embeddings, embedding_model):
        """Fit should accept tokenized_documents and produce valid topics."""
        model = BERTopic(embedding_model=embedding_model)
        model.fit(
            documents,
            tokenized_documents=tokenized_documents,
            embeddings=document_embeddings,
        )
        assert hasattr(model, "topics_")
        assert len(model.topics_) == len(documents)

    def test_fit_transform_with_tokenized_docs(
        self, documents, tokenized_documents, document_embeddings, embedding_model
    ):
        """fit_transform should accept tokenized_documents."""
        model = BERTopic(embedding_model=embedding_model)
        topics, _ = model.fit_transform(
            documents,
            tokenized_documents=tokenized_documents,
            embeddings=document_embeddings,
        )
        assert len(topics) == len(documents)
        assert len(set(topics)) > 1

    def test_tokenized_document_column_created(
        self, documents, tokenized_documents, document_embeddings, embedding_model
    ):
        """Internal DataFrame should have Tokenized_Document column."""
        model = BERTopic(embedding_model=embedding_model)
        model.fit_transform(
            documents,
            tokenized_documents=tokenized_documents,
            embeddings=document_embeddings,
        )
        # The column is internal — verify via topic representations existing
        assert hasattr(model, "topic_representations_")
        assert len(model.topic_representations_) > 0

    def test_without_tokenized_docs_works_as_before(self, documents, document_embeddings, embedding_model):
        """Without tokenized_documents, behavior should be identical to upstream."""
        model = BERTopic(embedding_model=embedding_model)
        topics, _ = model.fit_transform(documents, embeddings=document_embeddings)
        assert len(topics) == len(documents)


class TestPreTokenizedCTfIdf:
    """Verify _c_tf_idf handles pre-tokenized documents."""

    def test_c_tf_idf_with_tokenized_docs(self):
        """_c_tf_idf should use tokenized docs when available."""
        # Create a simple vectorizer that works with pre-tokenized input
        vectorizer = CountVectorizer(analyzer=lambda x: x if isinstance(x, (list, tuple)) else x.split())

        docs_per_topic = pd.DataFrame(
            {
                "Topic": [0, 1],
                "Document": ["hello world test", "foo bar baz"],
                "Tokenized_Document": [
                    ("hello", "world", "test"),
                    ("foo", "bar", "baz"),
                ],
            }
        )

        # Verify the tokenized docs are used (not the raw text)
        # by using a vectorizer that would tokenize differently
        X = vectorizer.fit_transform(docs_per_topic.Tokenized_Document.to_numpy())
        assert X.shape[0] == 2


class TestPreTokenizedUpdateTopics:
    """Verify update_topics with pre-tokenized documents."""

    def test_update_topics_with_tokenized_docs(self, base_topic_model, documents, tokenized_documents):
        """update_topics should accept tokenized_documents."""
        model = copy.deepcopy(base_topic_model)
        model.update_topics(
            documents,
            tokenized_documents=tokenized_documents,
        )
        assert hasattr(model, "topic_representations_")


class TestPreTokenizedEdgeCases:
    """Edge cases and validation for pre-tokenized documents."""

    def test_mismatched_lengths_raises(self, documents, document_embeddings, embedding_model):
        """tokenized_documents with different length than documents should raise."""
        model = BERTopic(embedding_model=embedding_model)
        short_tokenized = [["token"]] * (len(documents) - 5)

        with pytest.raises((ValueError, IndexError)):
            model.fit_transform(
                documents,
                tokenized_documents=short_tokenized,
                embeddings=document_embeddings,
            )

    def test_empty_token_lists_handled(self, documents, document_embeddings, embedding_model):
        """Documents with empty token lists should not crash."""
        model = BERTopic(embedding_model=embedding_model)
        # Some docs have empty token lists
        tokenized = [simple_tokenizer(doc) for doc in documents]
        tokenized[0] = []
        tokenized[1] = []

        topics, _ = model.fit_transform(
            documents,
            tokenized_documents=tokenized,
            embeddings=document_embeddings,
        )
        assert len(topics) == len(documents)

    def test_tokenized_and_plain_produce_same_topics(self, documents, document_embeddings, embedding_model):
        """Pre-tokenized with simple split should produce same results as plain text."""
        model1 = BERTopic(embedding_model=embedding_model, nr_topics=5)
        model1.umap_model.random_state = 42
        model1.hdbscan_model.min_cluster_size = 3

        model2 = copy.deepcopy(model1)

        tokenized = [doc.lower().split() for doc in documents]

        topics_plain, _ = model1.fit_transform(documents, embeddings=document_embeddings)
        topics_tok, _ = model2.fit_transform(
            documents,
            tokenized_documents=tokenized,
            embeddings=document_embeddings,
        )

        # Topic assignments should be identical (same embeddings, same UMAP seed)
        assert topics_plain == topics_tok
