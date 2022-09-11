import copy
import pytest
import numpy as np
import pandas as pd


@pytest.mark.parametrize('model', [('kmeans_pca_topic_model'),
                                   ('base_topic_model'),
                                   ('custom_topic_model'),
                                   ('merged_topic_model'),
                                   ('reduced_topic_model'),
                                   ('online_topic_model')])
def test_get_topic(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    topics = [topic_model.get_topic(topic) for topic in set(topic_model.topics_)]
    unknown_topic = topic_model.get_topic(500)

    for topic in topics:
        assert topic is not False

    assert len(topics) == len(topic_model.get_topic_info())
    assert not unknown_topic

@pytest.mark.parametrize('model', [('kmeans_pca_topic_model'),
                                   ('base_topic_model'),
                                   ('custom_topic_model'),
                                   ('merged_topic_model'),
                                   ('reduced_topic_model'),
                                   ('online_topic_model')])
def test_get_topics(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    topics = topic_model.get_topics()

    assert topics == topic_model.topic_representations_
    assert len(topics.keys()) == len(set(topic_model.topics_))


@pytest.mark.parametrize('model', [('kmeans_pca_topic_model'),
                                   ('base_topic_model'),
                                   ('custom_topic_model'),
                                   ('merged_topic_model'),
                                   ('reduced_topic_model'),
                                   ('online_topic_model')])
def test_get_topic_freq(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    for topic in set(topic_model.topics_):
        assert not isinstance(topic_model.get_topic_freq(topic), pd.DataFrame)

    topic_freq = topic_model.get_topic_freq()
    unique_topics = set(topic_model.topics_)
    topics_in_mapper = set(np.array(topic_model.topic_mapper_.mappings_)[: ,-1])

    assert isinstance(topic_freq, pd.DataFrame)

    assert len(topic_freq) == len(set(topic_model.topics_))
    assert len(topics_in_mapper.difference(unique_topics)) == 0
    assert len(unique_topics.difference(topics_in_mapper)) == 0


@pytest.mark.parametrize('model', [('base_topic_model'),
                                   ('custom_topic_model'),
                                   ('merged_topic_model'),
                                   ('reduced_topic_model')])
def test_get_representative_docs(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    all_docs = topic_model.get_representative_docs()
    unique_topics = set(topic_model.topics_)
    topics_in_mapper = set(np.array(topic_model.topic_mapper_.mappings_)[:, -1])

    assert len(all_docs) == len(topic_model.topic_sizes_.keys()) - topic_model._outliers
    assert len(all_docs) == len(topics_in_mapper) - topic_model._outliers
    assert len(all_docs) == topic_model.c_tf_idf_.shape[0] - topic_model._outliers
    assert len(all_docs) == len(topic_model.topic_labels_) - topic_model._outliers

    if model == "merged_topic_model":
        assert len([True for docs in all_docs.values() if len(docs) == 6]) == 2
        assert len([True for docs in all_docs.values() if len(docs) == 9]) == 1
    elif model == "reduced_topic_model":
        assert any([True if len(docs) == 3 else False for docs in all_docs.values()])
    else:
        assert all([True if len(docs) == 3 else False for docs in all_docs.values()])

    topics = set(list(all_docs.keys()))

    assert len(topics.difference(unique_topics)) == 0
    assert len(topics.difference(topics_in_mapper)) == 0


@pytest.mark.parametrize('model', [('kmeans_pca_topic_model'),
                                   ('base_topic_model'),
                                   ('custom_topic_model'),
                                   ('merged_topic_model'),
                                   ('reduced_topic_model'),
                                   ('online_topic_model')])
def test_get_topic_info(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    info = topic_model.get_topic_info()

    assert all(info.columns == ['Topic', 'Count', 'Name'])

    if topic_model._outliers:
        assert info.iloc[0].Topic == -1
    else:
        assert info.iloc[0].Topic == 0

    for topic in set(topic_model.topics_):
        assert len(topic_model.get_topic_info(topic)) == 1

    assert len(topic_model.get_topic_info(200)) == 0
