import copy
import pytest


@pytest.mark.parametrize('model', [('kmeans_pca_topic_model'),
                                   ('base_topic_model'),
                                   ('custom_topic_model'),
                                   ('merged_topic_model'),
                                   ('reduced_topic_model')])
def test_term_rank(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    fig = topic_model.visualize_term_rank()

    assert len(fig.to_dict()["data"]) == len(set(topic_model.topics_)) - topic_model._outliers
