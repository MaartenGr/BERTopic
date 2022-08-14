"""
Test fitted BERTopic

These test relate to a typical order of BERTopic usage. From training the model
and predicting new instances, to further reducing topics and visualizing the results.

"""
import copy
import pytest
from sklearn.datasets import fetch_20newsgroups

docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data'][:500]


@pytest.mark.parametrize('model', [('kmeans_pca_topic_model'), ('custom_topic_model'), ('merged_topic_model'), ('reduced_topic_model'), ('online_topic_model')])
def test_full_model(model, request):
    """ Tests the entire pipeline in one go. This serves as a sanity check to see if the default
    settings result in a good separation of topics.

    NOTE: This does not cover all cases but merely combines it all together
    """
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    topics = topic_model.topics_

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
    timestamps = [i % 10 for i in range(len(docs))]
    topics_over_time = topic_model.topics_over_time(docs, timestamps)

    assert topics_over_time.Frequency.sum() == len(docs)
    assert len(topics_over_time.Topic.unique()) == len(set(topics))

    # Test hierarchical topics
    hier_topics = topic_model.hierarchical_topics(docs)

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
    new_topics, new_probs = topic_model.reduce_topics(docs, nr_topics=nr_topics)

    assert len(topic_model.get_topic_freq()) == nr_topics + topic_model._outliers
    assert len(new_topics) == len(topics)

    # Test update topics
    topic = topic_model.get_topic(1)[:10]
    vectorizer_model = topic_model.vectorizer_model
    topic_model.update_topics(docs, n_gram_range=(2, 2))

    updated_topic = topic_model.get_topic(1)[:10]

    topic_model.update_topics(docs, vectorizer_model=vectorizer_model)
    original_topic = topic_model.get_topic(1)[:10]

    assert topic != updated_topic
    assert topic == original_topic

    # Test updating topic labels
    topic_labels = topic_model.generate_topic_labels(nr_words=3, topic_prefix=False, word_length=10, separator=", ")
    assert len(topic_labels) == len(set(new_topics))

    # Test setting topic labels
    topic_model.set_topic_labels(topic_labels)
    assert topic_model.custom_labels_ == topic_labels

    # Test merging topics
    freq = topic_model.get_topic_freq(0)
    topics_to_merge = [0, 1]
    topic_model.merge_topics(docs, topics_to_merge)
    assert freq < topic_model.get_topic_freq(0)
