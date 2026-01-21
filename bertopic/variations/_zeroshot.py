"""Zero-Shot Topic Modeling Variation for BERTopic.

# Methodology

This method works as follows:

* Create labels for predefined topics and embed each label
* For each document, compute cosine similarity with each predefined topic label
* If the similarity exceeds a threshold, assign the topic to the document
* Otherwise, cluster the document using BERTopic's regular pipeline

# Three Scenarios

There are three main scenarios when using Zero-Shot Topic Modeling:

* Both zero-shot topics and clustered topics are detected --> both techqniques are used and topics combined
* Only zero-shot topics are detected --> no clustering is needed.
* No zero-shot topics are detected --> regular BERTopic is run.

"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from bertopic._topics import Topics
from bertopic._corpus import Corpus
from bertopic._utils import MyLogger
from bertopic.cluster import BaseCluster

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bertopic import BERTopic

logger = MyLogger()
logger.configure("WARNING")


def zeroshot_tm(topic_model: "BERTopic", corpus: Corpus) -> Corpus:
    """Find documents that could be assigned to either one of the topics in topic_model.zeroshot_topic_list.

    We transform the topics in `topic_model.zeroshot_topic_list` to embeddings and
    compare them through cosine similarity with the document embeddings.
    If they pass the `topic_model.zeroshot_min_similarity` threshold, they are assigned.

    Arguments:
        topic_model: The BERTopic instance
        corpus: The documents and their embeddings

    Returns:
        zeroshot_data: Documents containing documents assigned to zero-shot topics
        cluster_data: Documents containing documents to be clustered
    """
    zeroshot_data = Corpus()
    if not topic_model._is_zeroshot():
        cluster_data = corpus
        return cluster_data, zeroshot_data

    logger.info(
        "Zeroshot Step 1 - Finding documents that could be assigned to either one of the zero-shot topics"
    )

    # Similarity between document and zero-shot topic embeddings
    zeroshot_embeddings = topic_model._extract_embeddings(topic_model.zeroshot_topic_list)
    cosine_similarities = cosine_similarity(corpus.embeddings, zeroshot_embeddings)
    assignment = np.argmax(cosine_similarities, 1)
    assignment_vals = np.max(cosine_similarities, 1)
    assigned_ids = [
        index for index, value in enumerate(assignment_vals) if value >= topic_model.zeroshot_min_similarity
    ]
    non_assigned_ids = [
        index for index, value in enumerate(assignment_vals) if value < topic_model.zeroshot_min_similarity
    ]

    # Assign topics to zero-shot documents
    zeroshot_data = Corpus()
    if len(assigned_ids) > 0:
        zeroshot_documents = corpus.get_documents_by_indices(assigned_ids)
        zeroshot_topics = [topic for topic in assignment[assigned_ids]]
        zeroshot_labels = [topic_model.zeroshot_topic_list[i] for i in sorted(list(set(zeroshot_topics)))]
        zeroshot_data = Corpus(
            documents=zeroshot_documents,
            embeddings=corpus.embeddings[assigned_ids],
            topics=zeroshot_topics,
            original_indices=assigned_ids,
            _zeroshot_labels=zeroshot_labels,
        )

    # Select non-assigned topics to be clustered
    cluster_data = Corpus(
        documents=corpus.get_documents_by_indices(non_assigned_ids),
        embeddings=corpus.embeddings[non_assigned_ids],
        original_indices=non_assigned_ids,
        y=corpus.y[non_assigned_ids] if corpus.y is not None else None,
    )
    cluster_data.umap_embeddings = topic_model.umap_model.transform(cluster_data.embeddings)
    # If all documents were assigned to zero-shot topics
    if len(assigned_ids) == len(corpus.documents):
        corpus.topics = zeroshot_data.topics

    logger.info("Zeroshot Step 1 - Completed \u2713")

    # Check that if a number of topics was specified, it exceeds the number of zeroshot topics matched
    num_zeroshot_topics = zeroshot_data.nr_topics() if zeroshot_data is not None else 0
    if topic_model.nr_topics != "auto":
        if topic_model.nr_topics and not topic_model.nr_topics > num_zeroshot_topics:
            raise ValueError(
                f"The set nr_topics ({topic_model.nr_topics}) must exceed the number of matched zero-shot topics "
                f"({num_zeroshot_topics}). Consider raising nr_topics or raising the "
                f"zeroshot_min_similarity ({topic_model.zeroshot_min_similarity})."
            )

    return cluster_data, zeroshot_data


def combine_zeroshot_topics(topic_model: "BERTopic", corpus: Corpus, zeroshot_data: Corpus) -> Corpus:
    """Combine the zero-shot topics with the clustered topics.

    The zero-shot topics will be inserted between the outlier topic (that may or may not exist) and the rest of the
    topics from clustering. The rest of the topics from clustering will be given new IDs to correspond to topics
    after zero-shot topics.

    Documents and embeddings used in zero-shot topic modeling and clustering and re-merged.

    Arguments:
        topic_model: The BERTopic instance
        corpus: Clustered documents
        zeroshot_data: Documents belonging to zero-shot topics

    Returns:
        corpus: Documents containing all the original documents with their topic assignments
    """
    if len(corpus) > 0 and len(zeroshot_data) > 0:
        logger.info(
            "Zeroshot Step 2 - Combining topics from zero-shot topic modeling with topics from clustering..."
        )

        # Combine data
        corpus.sort_topics_by_frequency()
        zeroshot_data.sort_topics_by_frequency()
        corpus = corpus + zeroshot_data

        # Create new Topics
        topic_model.topics_ = Topics().initialize(corpus.topics, corpus._zeroshot_labels).sort_by_frequency()
        corpus.map_topics_and_probabilities(topic_model.topics_, from_original=True)
        logger.info("Zeroshot Step 2 - Completed \u2713")

    return corpus


def update_probabilities(topic_model: "BERTopic", corpus: Corpus, zeroshot_docs: Corpus) -> Corpus:
    """Update probabilities in the case of zero-shot topics.

    Arguments:
        topic_model: The BERTopic instance
        corpus: The documents containing all topic assignments
        zeroshot_docs: The documents assigned to zero-shot topics

    Returns:
        corpus: The documents containing updated probabilities
    """
    # In the case of zero-shot topics, probability will come from cosine similarity,
    # and the HDBSCAN model will be removed
    if len(zeroshot_docs) > 0:
        topic_model.hdbscan_model = BaseCluster()
        sim_matrix = cosine_similarity(corpus.embeddings, topic_model.topics_.embeddings)
        corpus.probabilities = (
            sim_matrix if topic_model.calculate_probabilities else np.max(sim_matrix, axis=1)
        )
        topic_model.topics_._zeroshot_probabilities = corpus.probabilities

    return corpus
