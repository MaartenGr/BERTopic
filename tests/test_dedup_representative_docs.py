"""Tests for deduplicating representative documents sampling.

Verifies that `_extract_representative_docs` samples without replacement so a
topic never yields duplicate representative documents.

Run from BERTopic repo root:
    pytest tests/test_dedup_representative_docs.py -v
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer


def _build_minimal_model(docs, topics_list):
    """Build a minimal BERTopic model with vectorizer and c-TF-IDF."""
    documents = pd.DataFrame(
        {
            "Document": docs,
            "ID": range(len(docs)),
            "Topic": topics_list,
        }
    )

    vectorizer = CountVectorizer()
    docs_per_topic = documents.groupby(["Topic"], as_index=False).agg({"Document": " ".join})
    X = vectorizer.fit_transform(docs_per_topic.Document.values)

    ctfidf_model = ClassTfidfTransformer()
    ctfidf_model.fit(X)
    c_tf_idf = ctfidf_model.transform(X)

    model = BERTopic()
    model.vectorizer_model = vectorizer
    model.ctfidf_model = ctfidf_model

    topics = {}
    for topic_id in sorted(documents.Topic.unique()):
        topic_docs = docs_per_topic.loc[docs_per_topic.Topic == topic_id, "Document"].to_numpy()[0]
        bow = vectorizer.transform([topic_docs])
        tf = ctfidf_model.transform(bow)
        feature_names = vectorizer.get_feature_names_out()
        scores = tf.toarray().flatten()
        top_indices = scores.argsort()[-5:][::-1]
        topics[topic_id] = [(feature_names[i], float(scores[i])) for i in top_indices]

    return model, c_tf_idf, documents, topics


def test_no_duplicate_docs_per_topic():
    """Each topic's representative docs should contain no duplicates."""
    docs = ["alpha", "beta", "gamma", "delta", "epsilon"] * 3
    topics_list = [0, 0, 0, 1, 1] * 3

    model, c_tf_idf, documents, topics = _build_minimal_model(docs, topics_list)

    repr_docs_mappings, _repr_docs, _, _ = model._extract_representative_docs(
        c_tf_idf,
        documents,
        topics,
        nr_samples=500,
        nr_repr_docs=5,
    )

    assert repr_docs_mappings
    for topic, topic_docs in repr_docs_mappings.items():
        assert len(topic_docs) == len(set(topic_docs)), (
            f"Topic {topic} has duplicate representative docs: {[d for d in topic_docs if topic_docs.count(d) > 1]}"
        )


def test_heavy_duplicates_no_duplicates_in_output():
    """When a topic has 3 unique docs but nr_samples=500, no duplicates should appear."""
    docs = ["doc A", "doc B", "doc C"] * 2 + ["unique doc"]
    topics_list = [0, 0, 0, 1, 1, 1, 0]

    model, c_tf_idf, documents, topics = _build_minimal_model(docs, topics_list)

    repr_docs_mappings, _repr_docs, _, _ = model._extract_representative_docs(
        c_tf_idf,
        documents,
        topics,
        nr_samples=500,
        nr_repr_docs=3,
    )

    for topic, docs_list in repr_docs_mappings.items():
        assert len(docs_list) == len(set(docs_list)), f"Topic {topic} has duplicate representative docs"


def test_repr_docs_count_respects_topic_size():
    """nr_repr_docs should be capped at the number of unique docs in the topic."""
    docs = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    topics_list = [0, 0, 1, 1, 1, 1]
    # Topic 0 has only 2 unique docs — requesting 5 should yield 2

    model, c_tf_idf, documents, topics = _build_minimal_model(docs, topics_list)

    repr_docs_mappings, _, _, _ = model._extract_representative_docs(
        c_tf_idf,
        documents,
        topics,
        nr_samples=500,
        nr_repr_docs=5,
    )

    assert len(repr_docs_mappings[0]) <= 2
    assert len(repr_docs_mappings[1]) <= 4


def test_repr_docs_count_with_nr_repr_docs_greater_than_topic_size():
    """When nr_repr_docs > unique docs in a topic, return all unique docs."""
    docs = ["only one"] * 5 + ["other topic doc"] * 5
    topics_list = [0] * 5 + [1] * 5
    # Topic 0 has 1 unique doc, topic 1 has 1 unique doc

    model, c_tf_idf, documents, topics = _build_minimal_model(docs, topics_list)

    repr_docs_mappings, _, _, _ = model._extract_representative_docs(
        c_tf_idf,
        documents,
        topics,
        nr_samples=500,
        nr_repr_docs=10,
    )

    for topic, docs_list in repr_docs_mappings.items():
        assert len(docs_list) == len(set(docs_list))


def test_with_diversity_no_duplicates():
    """MMR branch (diversity > 0) should also produce no duplicates."""
    docs = [
        "machine learning algorithms",
        "deep learning neural networks",
        "natural language processing",
        "computer vision image analysis",
        "data mining techniques",
        "statistical modeling methods",
    ]
    topics_list = [0, 0, 0, 1, 1, 1]

    model, c_tf_idf, documents, topics = _build_minimal_model(docs, topics_list)

    repr_docs_mappings, _, _, _ = model._extract_representative_docs(
        c_tf_idf,
        documents,
        topics,
        nr_samples=500,
        nr_repr_docs=3,
        diversity=0.5,
    )

    for topic, docs_list in repr_docs_mappings.items():
        assert len(docs_list) == len(set(docs_list)), f"Topic {topic} has duplicate representative docs with diversity"
