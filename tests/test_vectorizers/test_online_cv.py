import copy
import pytest
from bertopic.vectorizers import OnlineCountVectorizer


@pytest.mark.parametrize(
    "model",
    [
        ("kmeans_pca_topic_model"),
        ("custom_topic_model"),
        ("merged_topic_model"),
        ("reduced_topic_model"),
        ("online_topic_model"),
    ],
)
def test_online_cv(model, documents, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    vectorizer_model = OnlineCountVectorizer(stop_words="english", ngram_range=(2, 2))

    topics = [topic_model.get_topic(topic) for topic in set(topic_model.topics_)]
    topic_model.update_topics(documents, vectorizer_model=vectorizer_model)
    new_topics = [topic_model.get_topic(topic) for topic in set(topic_model.topics_)]

    for old_topic, new_topic in zip(topics, new_topics):
        if old_topic[0][0] != "":
            assert old_topic != new_topic


@pytest.mark.parametrize("model", [("online_topic_model")])
def test_clean_bow(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))

    original_shape = topic_model.vectorizer_model.X_.shape
    topic_model.vectorizer_model.delete_min_df = 2
    topic_model.vectorizer_model._clean_bow()

    assert original_shape[0] == topic_model.vectorizer_model.X_.shape[0]
    assert original_shape[1] > topic_model.vectorizer_model.X_.shape[1]


def test_refit_restarts_bow(documents):
    vectorizer_model = OnlineCountVectorizer()
    vectorizer_model.partial_fit(documents[:50]).update_bow(documents[:50])

    # A refit renumbers the vocabulary, so the kept bag-of-words must start over from it
    X = vectorizer_model.fit_transform(documents[50:100])
    assert vectorizer_model.X_.shape == X.shape and (vectorizer_model.X_ != X).nnz == 0
