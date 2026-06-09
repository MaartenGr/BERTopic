"""Feature importance representation model for BERTopic.

New file: bertopic/representation/_feature_importance.py

Provides two methods for computing term importance per topic:
- "fighting_words": Bayesian log-odds with Dirichlet priors (Monroe et al. 2008)
- "centroid_distance": Cosine similarity between topic centroid and term embeddings
"""

from typing import List, Mapping, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from bertopic.representation._base import BaseRepresentation


class FeatureImportance(BaseRepresentation):
    """Compute alternative feature importance scores for topic terms.

    This representation model re-ranks the top words per topic using either
    contrastive (fighting words) or embedding-based (centroid distance) methods.

    Arguments:
        method: The importance method to use. Options:
                * ``"fighting_words"`` — Bayesian log-odds with Dirichlet priors.
                  Highlights terms that distinguish a topic from the rest.
                * ``"centroid_distance"`` — Cosine similarity between the topic
                  centroid embedding and term embeddings. Highlights terms that
                  are semantically central to the topic.
        top_n_words: Number of top words to return per topic.
        prior: Dirichlet prior for fighting words. Use ``"corpus"`` to derive
               priors from corpus word frequencies, or a float for a symmetric prior.

    Examples:
    ```python
    from bertopic import BERTopic
    from bertopic.representation import FeatureImportance

    model = BERTopic(
        representation_model={
            "Main": some_model,
            "FightingWords": FeatureImportance(method="fighting_words"),
            "CentroidDist": FeatureImportance(method="centroid_distance"),
        }
    )
    topics, probs = model.fit_transform(docs)
    model.get_topic(0, aspect="FightingWords")
    ```
    """

    def __init__(
        self,
        method: str = "fighting_words",
        top_n_words: int = 10,
        prior: float | str = "corpus",
    ):
        self.method = method
        self.top_n_words = top_n_words
        self.prior = prior

    def extract_topics(
        self,
        topic_model,
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix,
        topics: Mapping[str, List[Tuple[str, float]]],
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """Extract topics with alternative importance scores.

        Arguments:
            topic_model: The fitted BERTopic model.
            documents: DataFrame with "Document" and "Topic" columns.
            c_tf_idf: The c-TF-IDF matrix for the topics.
            topics: Default topic representations from c-TF-IDF.

        Returns:
            Updated topic representations with re-ranked words.
        """
        # Get feature names from the vectorizer
        try:
            words = topic_model.vectorizer_model.get_feature_names_out()
        except AttributeError:
            words = topic_model.vectorizer_model.get_feature_names()

        # Get unique topics (excluding outliers)
        unique_topics = sorted([t for t in documents.Topic.unique() if t != -1])

        if self.method == "fighting_words":
            importance = self._fighting_words(topic_model, documents, unique_topics, words)
        elif self.method == "centroid_distance":
            importance = self._centroid_distance(topic_model, unique_topics, words)
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'fighting_words' or 'centroid_distance'.")

        # Convert importance matrix to topic representations
        updated_topics = {}
        for idx, topic_id in enumerate(unique_topics):
            scores = importance[idx]
            top_indices = np.argsort(scores)[::-1][: self.top_n_words]
            updated_topics[topic_id] = [(words[i], float(scores[i])) for i in top_indices]

        return updated_topics

    def _fighting_words(
        self,
        topic_model,
        documents: pd.DataFrame,
        unique_topics: list,
        words: np.ndarray,
    ) -> np.ndarray:
        """Compute fighting words (Bayesian log-odds) importance.

        Arguments:
            topic_model: The fitted BERTopic model.
            documents: DataFrame with "Document" and "Topic" columns.
            unique_topics: Sorted list of non-outlier topic IDs.
            words: Feature names from the vectorizer.

        Returns:
            Importance matrix of shape (n_topics, vocab_size).
        """
        # Build document-term matrix from the documents
        clean_docs = topic_model._preprocess_text(documents.Document.values)
        doc_term_matrix = topic_model.vectorizer_model.transform(clean_docs)

        n_topics = len(unique_topics)
        n_vocab = doc_term_matrix.shape[1]

        # Map document topics to sequential indices
        topic_to_idx = {t: i for i, t in enumerate(unique_topics)}
        labels = np.array([topic_to_idx.get(t, -1) for t in documents.Topic.to_numpy()])

        # Compute priors
        if self.prior == "corpus":
            priors = np.ravel(np.asarray(doc_term_matrix.sum(axis=0)))
        else:
            priors = np.full(n_vocab, float(self.prior))
        a0 = np.sum(priors)

        components = []
        for i_topic in range(n_topics):
            mask = labels == i_topic
            topic_freq = np.ravel(np.asarray(doc_term_matrix[mask].sum(axis=0)))
            rest_freq = np.ravel(np.asarray(doc_term_matrix[~mask].sum(axis=0)))
            n1 = np.sum(topic_freq)
            n2 = np.sum(rest_freq)
            topic_logodds = np.log((topic_freq + priors) / (n1 + a0 - topic_freq - priors))
            rest_logodds = np.log((rest_freq + priors) / (n2 + a0 - rest_freq - priors))
            delta = topic_logodds - rest_logodds
            delta_var = 1 / (topic_freq + priors) + 1 / (rest_freq + priors)
            zscore = delta / np.sqrt(delta_var)
            components.append(zscore)

        return np.stack(components)

    def _centroid_distance(
        self,
        topic_model,
        unique_topics: list,
        words: np.ndarray,
    ) -> np.ndarray:
        """Compute centroid distance importance.

        Arguments:
            topic_model: The fitted BERTopic model.
            unique_topics: Sorted list of non-outlier topic IDs.
            words: Feature names from the vectorizer.

        Returns:
            Importance matrix of shape (n_topics, vocab_size).
        """
        if topic_model.embedding_model is None:
            raise ValueError(
                "centroid_distance requires an embedding model. Pass embedding_model when instantiating BERTopic."
            )

        # Get topic centroids (skip outlier topic if present)
        topic_embeddings = topic_model.topic_embeddings_[topic_model._outliers :]

        # Embed vocabulary terms
        vocab_embeddings = topic_model.embedding_model.embed_words(list(words))
        if vocab_embeddings is None:
            # Fallback: embed as documents
            vocab_embeddings = topic_model.embedding_model.embed_documents(list(words))

        vocab_embeddings = np.array(vocab_embeddings)

        # Compute cosine similarity between topic centroids and term embeddings
        n_topics = len(unique_topics)
        n_vocab = len(words)
        components = np.full((n_topics, n_vocab), np.nan)
        valid = np.all(np.isfinite(topic_embeddings), axis=1)
        similarities = cosine_similarity(topic_embeddings[valid], vocab_embeddings)
        components[valid, :] = similarities

        return components
