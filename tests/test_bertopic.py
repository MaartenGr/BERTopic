"""
Test fitted BERTopic

These test relate to a typical order of BERTopic usage. From training the model
and predicting new instances, to further reducing topics and visualizing the results.

TO DO:
    * Add Evaluation measures to check for quality
"""


from sklearn.datasets import fetch_20newsgroups

newsgroup_docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data'][:1000]


def test_full_model(base_bertopic):
    """ Tests the entire pipeline in one go. This serves as a sanity check to see if the default
    settings result in a good separation of topics.

    NOTE: This does not cover all cases but merely combines it all together
    """
    # Test fit
    base_bertopic.calculate_probabilities = True
    topics, probs = base_bertopic.fit_transform(newsgroup_docs)

    for topic in set(topics):
        words = base_bertopic.get_topic(topic)[:10]
        assert len(words) == 10

    for topic in base_bertopic.get_topic_freq().Topic:
        words = base_bertopic.get_topic(topic)[:10]
        assert len(words) == 10

    assert len(base_bertopic.get_topic_freq()) > 2
    assert len(base_bertopic.get_topics()) == len(base_bertopic.get_topic_freq())

    # Test transform
    doc = "This is a new document to predict."
    topics_test, probs_test = base_bertopic.transform([doc])

    assert len(topics_test) == 1

    # Test topics over time
    timestamps = [i % 10 for i in range(1000)]
    topics_over_time = base_bertopic.topics_over_time(newsgroup_docs, topics, timestamps)

    assert topics_over_time.Frequency.sum() == 1000
    assert len(topics_over_time.Topic.unique()) == len(set(topics))

    # Test find topic
    similar_topics, similarity = base_bertopic.find_topics("query", top_n=2)
    assert len(similar_topics) == 2
    assert len(similarity) == 2
    assert max(similarity) <= 1

    # Test update topics
    topic = base_bertopic.get_topic(1)[:10]
    base_bertopic.update_topics(newsgroup_docs, topics, n_gram_range=(2, 2))
    updated_topic = base_bertopic.get_topic(1)[:10]
    base_bertopic.update_topics(newsgroup_docs, topics)
    original_topic = base_bertopic.get_topic(1)[:10]

    assert topic != updated_topic
    assert topic == original_topic

    # Test topic reduction
    nr_topics = 2
    new_topics, new_probs = base_bertopic.reduce_topics(newsgroup_docs, topics, probs, nr_topics=nr_topics)

    assert len(base_bertopic.get_topic_freq()) == nr_topics + 1
    assert len(new_topics) == len(topics)
    assert len(new_probs) == len(probs)
