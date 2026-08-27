"""Guided Topic Modeling Variation for BERTopic.
See: https://maartengr.github.io/BERTopic/getting_started/guided/guided.html.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from bertopic._corpus import Corpus


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bertopic import BERTopic


def guided_tm(topic_model: "BERTopic", corpus: Corpus) -> Corpus:
    """Apply Guided Topic Modeling.

    We transform the seeded topics to embeddings using the
    same embedder as used for generating document embeddings.

    Then, we apply cosine similarity between the embeddings
    and set labels for documents that are more similar to
    one of the topics than the average document.

    If a document is more similar to the average document
    than any of the topics, it gets the -1 label and is
    thereby not included in UMAP.

    Arguments:
        topic_model: The BERTopic instance
        corpus: A Corpus object containing documents and their embeddings

    Returns:
        y: The labels for each seeded topic
        embeddings: Updated embeddings
    """
    if topic_model.seed_topic_list is not None and topic_model.embedding_model is not None:
        topic_model.logger.info("Guided - Find embeddings highly related to seeded topics.")
        embeddings = corpus.embeddings

        # Create embeddings from the seeded topics
        seed_topic_list = [" ".join(seed_topic) for seed_topic in topic_model.seed_topic_list]
        seed_topic_embeddings = topic_model._extract_embeddings(seed_topic_list, verbose=topic_model.verbose)
        seed_topic_embeddings = np.vstack([seed_topic_embeddings, embeddings.mean(axis=0)])

        # Label documents that are most similar to one of the seeded topics
        sim_matrix = cosine_similarity(embeddings, seed_topic_embeddings)
        y = [np.argmax(sim_matrix[index]) for index in range(sim_matrix.shape[0])]
        y = [val if val != len(seed_topic_list) else -1 for val in y]

        # Average the document embeddings related to the seeded topics with the
        # embedding of the seeded topic to force the documents in a cluster
        for seed_topic in range(len(seed_topic_list)):
            indices = [index for index, topic in enumerate(y) if topic == seed_topic]
            embeddings[indices] = embeddings[indices] * 0.75 + seed_topic_embeddings[seed_topic] * 0.25

        # Update docs
        corpus.y = np.array(corpus.y) if corpus.y is not None else None
        corpus.embeddings = embeddings
        topic_model.logger.info("Guided - Completed \u2713")

    return corpus
