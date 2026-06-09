import warnings
from collections.abc import Mapping
from typing import List

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from bertopic.representation._base import BaseRepresentation


class MaximalMarginalRelevance(BaseRepresentation):
    """Calculate Maximal Marginal Relevance (MMR)
    between candidate keywords and the document.

    MMR considers the similarity of keywords/keyphrases with the
    document, along with the similarity of already selected
    keywords and keyphrases. This results in a selection of keywords
    that maximize their within diversity with respect to the document.

    Arguments:
        diversity: How diverse the select keywords/keyphrases are.
                    Values range between 0 and 1 with 0 being not diverse at all
                    and 1 being most diverse.
        top_n_words: The number of keywords/keyhprases to return

    Usage:

    ```python
    from bertopic.representation import MaximalMarginalRelevance
    from bertopic import BERTopic

    # Create your representation model
    representation_model = MaximalMarginalRelevance(diversity=0.3)

    # Use the representation model in BERTopic on top of the default pipeline
    topic_model = BERTopic(representation_model=representation_model)
    ```
    """

    def __init__(self, diversity: float = 0.1, top_n_words: int = 10):
        self.diversity = diversity
        self.top_n_words = top_n_words

    def extract_topics(
        self,
        topic_model,
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix,
        topics: Mapping[str, list[tuple[str, float]]],
    ) -> Mapping[str, list[tuple[str, float]]]:
        """Extract topic representations using batched embedding extraction.

        Instead of calling _extract_embeddings 2N times (once for words and once
        for the concatenated sentence per topic), this collects all items and makes
        a single embedding call.

        Arguments:
            topic_model: The BERTopic model
            documents: Not used
            c_tf_idf: Not used
            topics: The candidate topics as calculated with c-TF-IDF

        Returns:
            updated_topics: Updated topic representations
        """
        if topic_model.embedding_model is None:
            warnings.warn(
                "MaximalMarginalRelevance can only be used if BERTopic was instantiated "
                "with the `embedding_model` parameter."
            )
            return topics

        # ---- CHANGED: batch all embedding calls into one ----
        # Collect all items to embed: individual words + joined sentence per topic
        items_to_embed = []
        words_index_ranges = {}  # topic -> (start, end) for word embeddings
        sentence_indices = {}  # topic -> index for sentence embedding

        for topic, topic_words in topics.items():
            words = [word[0] for word in topic_words]

            # Record word embedding indices
            start = len(items_to_embed)
            items_to_embed.extend(words)
            words_index_ranges[topic] = (start, len(items_to_embed))

            # Record sentence embedding index
            sentence_indices[topic] = len(items_to_embed)
            items_to_embed.append(" ".join(words))

        # Single embedding call for all items across all topics
        all_embeddings = topic_model._extract_embeddings(items_to_embed, method="word", verbose=False)
        # ---- END CHANGED ----

        updated_topics = {}
        for topic, topic_words in topics.items():
            words = [word[0] for word in topic_words]
            w_start, w_end = words_index_ranges[topic]
            word_embeddings = all_embeddings[w_start:w_end]
            topic_embedding = all_embeddings[sentence_indices[topic]].reshape(1, -1)

            topic_words_selected = mmr(
                topic_embedding,
                word_embeddings,
                words,
                self.diversity,
                self.top_n_words,
            )
            updated_topics[topic] = [(word, value) for word, value in topics[topic] if word in topic_words_selected]
        return updated_topics


def mmr(
    doc_embedding: np.ndarray,
    word_embeddings: np.ndarray,
    words: List[str],
    diversity: float = 0.1,
    top_n: int = 10,
) -> List[str]:
    """Maximal Marginal Relevance.

    Arguments:
        doc_embedding: The document embeddings
        word_embeddings: The embeddings of the selected candidate keywords/phrases
        words: The selected candidate keywords/keyphrases
        diversity: The diversity of the selected embeddings.
                   Values between 0 and 1.
        top_n: The top n items to return

    Returns:
            List[str]: The selected keywords/keyphrases
    """
    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(min(top_n - 1, len(candidates_idx))):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr_score = (1 - diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr_score)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]
