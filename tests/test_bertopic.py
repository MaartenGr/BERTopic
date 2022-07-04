"""
Test fitted BERTopic

These test relate to a typical order of BERTopic usage. From training the model
and predicting new instances, to further reducing topics and visualizing the results.

TO DO:
    * Add Evaluation measures to check for quality
"""
import pytest
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from bertopic import BERTopic

newsgroup_docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data'][:2000]
base_model = BERTopic(language="english", verbose=True, min_topic_size=5)
kmeans_model = BERTopic(language="english", verbose=True, hdbscan_model=KMeans(n_clusters=10, random_state=42))


@pytest.mark.parametrize("topic_model", [base_model, kmeans_model])
def test_full_model(topic_model):
    """ Tests the entire pipeline in one go. This serves as a sanity check to see if the default
    settings result in a good separation of topics.

    NOTE: This does not cover all cases but merely combines it all together
    """
    # Test fit
    topic_model.calculate_probabilities = True
    topics, probs = topic_model.fit_transform(newsgroup_docs)

    for topic in set(topics):
        words = topic_model.get_topic(topic)[:10]
        assert len(words) == 10

    for topic in topic_model.get_topic_freq().Topic:
        words = topic_model.get_topic(topic)[:10]
        assert len(words) == 10

    assert len(topic_model.get_topic_freq()) > 2
    assert len(topic_model.get_topics()) == len(topic_model.get_topic_freq())

    # Test transform
    doc = "This is a new document to predict."
    topics_test, probs_test = topic_model.transform([doc])

    assert len(topics_test) == 1

    # Test topics over time
    timestamps = [i % 10 for i in range(2000)]
    topics_over_time = topic_model.topics_over_time(newsgroup_docs, topics, timestamps)

    assert topics_over_time.Frequency.sum() == 2000
    assert len(topics_over_time.Topic.unique()) == len(set(topics))

    # Test hierarchical topics
    hier_topics = topic_model.hierarchical_topics(newsgroup_docs, topics)

    assert len(hier_topics) > 0
    assert hier_topics.Parent_ID.astype(int).min() > max(topics)

    # Test creation of topic tree
    tree = topic_model.get_topic_tree(hier_topics, tight_layout=False)
    assert isinstance(tree, str)
    assert len(tree) > 10

    # Test find topic
    similar_topics, similarity = topic_model.find_topics("query", top_n=2)
    assert len(similar_topics) == 2
    assert len(similarity) == 2
    assert max(similarity) <= 1

    # Test topic reduction
    nr_topics = len(set(topics))
    nr_topics = 2 if nr_topics < 2 else nr_topics - 1
    new_topics, new_probs = topic_model.reduce_topics(newsgroup_docs, topics, probs, nr_topics=nr_topics)

    assert len(topic_model.get_topic_freq()) == nr_topics + topic_model._outliers
    assert len(new_topics) == len(topics)

    if probs is not None:
        assert len(new_probs) == len(probs)

    # Test update topics
    topic = topic_model.get_topic(1)[:10]
    topic_model.update_topics(newsgroup_docs, new_topics, n_gram_range=(2, 2))
    updated_topic = topic_model.get_topic(1)[:10]
    topic_model.update_topics(newsgroup_docs, new_topics)
    original_topic = topic_model.get_topic(1)[:10]

    assert topic != updated_topic
    assert topic == original_topic

    # Test updating topic labels
    topic_labels = topic_model.generate_topic_labels(nr_words=3, topic_prefix=False, word_length=10, separator=", ")
    assert len(topic_labels) == len(set(new_topics))

    # Test setting topic labels
    topic_model.set_topic_labels(topic_labels)
    assert topic_model.custom_labels == topic_labels

    # Test merging topics
    freq = topic_model.get_topic_freq(0)
    topics_to_merge = [0, 1]
    topic_model.merge_topics(newsgroup_docs, new_topics, topics_to_merge)
    assert freq < topic_model.get_topic_freq(0)
