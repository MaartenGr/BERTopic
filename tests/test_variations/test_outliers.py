import copy

import numpy as np


def test_outlier_flow_keeps_transform(base_topic_model, documents, document_embeddings):
    topic_model = copy.deepcopy(base_topic_model)
    inliers = np.array(topic_model.topics_) != -1
    new_topics = topic_model.reduce_outliers(
        documents, topic_model.topics_, probabilities=topic_model.probabilities_, strategy="probabilities"
    )
    assert -1 not in new_topics
    topic_model.update_topics(documents, topics=new_topics)

    # Documents that were never outliers keep their topic, so transform still predicts it
    predicted, _ = topic_model.transform(documents, document_embeddings)
    assert np.mean(np.array(predicted)[inliers] == np.array(topic_model.topics_)[inliers]) > 0.9


def test_reduce_outliers_with_probabilities(base_topic_model, documents):
    # Column 0 holds the outlier topic: the document is most likely an outlier, but of the topics, topic 1
    probabilities = np.zeros((1, len(base_topic_model.topic_sizes_)))
    probabilities[0, 0], probabilities[0, 2] = 0.6, 0.3

    new_topics = base_topic_model.reduce_outliers(
        documents[:1], [-1], probabilities=probabilities, strategy="probabilities"
    )

    assert new_topics == [1]
