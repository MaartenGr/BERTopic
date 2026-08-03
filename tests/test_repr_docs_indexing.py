"""Tests for positional indexing in `_extract_representative_docs`.

Verifies that representative documents map back to the correct topic when the
same text appears in multiple topics, instead of matching by text membership.

Run from BERTopic repo root:
    pytest tests/test_repr_docs_indexing.py -v
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


def test_duplicate_text_across_topics():
    """Documents with identical text in different topics get correct doc_ids."""
    # "shared text" appears in both topic 0 and topic 1
    docs = [
        "shared text",
        "unique topic zero content",
        "shared text",
        "unique topic one content",
    ]
    topics_list = [0, 0, 1, 1]

    model, c_tf_idf, documents, topics = _build_minimal_model(docs, topics_list)

    _repr_docs_mappings, _repr_docs, _repr_docs_indices, repr_docs_ids = model._extract_representative_docs(
        c_tf_idf, documents, topics, nr_samples=500, nr_repr_docs=3
    )

    # Verify each topic's representative doc_ids point to documents
    # that actually belong to that topic
    for topic_id, doc_ids in zip(topics.keys(), repr_docs_ids):
        for doc_id in doc_ids:
            actual_topic = documents.loc[doc_id, "Topic"]
            assert actual_topic == topic_id, (
                f"doc_id {doc_id} has topic {actual_topic} but was assigned as representative of topic {topic_id}"
            )


def test_all_identical_docs():
    """When all docs are identical, doc_ids should still be correct per topic."""
    docs = ["same text"] * 6
    topics_list = [0, 0, 0, 1, 1, 1]

    model, c_tf_idf, documents, topics = _build_minimal_model(docs, topics_list)

    _repr_docs_mappings, _repr_docs, _repr_docs_indices, repr_docs_ids = model._extract_representative_docs(
        c_tf_idf, documents, topics, nr_samples=500, nr_repr_docs=2
    )

    for topic_id, doc_ids in zip(topics.keys(), repr_docs_ids):
        for doc_id in doc_ids:
            actual_topic = documents.loc[doc_id, "Topic"]
            assert actual_topic == topic_id, (
                f"doc_id {doc_id} mapped to topic {actual_topic}, expected topic {topic_id}"
            )


def test_no_cross_topic_contamination():
    """Representative docs for a topic should not contain docs from another topic."""
    docs = [
        "alpha beta gamma",
        "alpha beta delta",
        "epsilon zeta eta",
        "epsilon zeta theta",
    ]
    topics_list = [0, 0, 1, 1]

    model, c_tf_idf, documents, topics = _build_minimal_model(docs, topics_list)

    repr_docs_mappings, _, _, _repr_docs_ids = model._extract_representative_docs(
        c_tf_idf, documents, topics, nr_samples=500, nr_repr_docs=2
    )

    for topic_id in topics.keys():
        repr_doc_texts = repr_docs_mappings[topic_id]
        topic_doc_texts = documents.loc[documents.Topic == topic_id, "Document"].tolist()
        for doc in repr_doc_texts:
            assert doc in topic_doc_texts, (
                f"Representative doc '{doc}' for topic {topic_id} not found in that topic's documents"
            )


def test_selected_indices_variable_used():
    """doc_ids count should match nr_repr_docs per topic."""
    docs = [
        "doc alpha one",
        "doc beta two",
        "doc gamma three",
        "doc delta four",
        "doc epsilon five",
        "doc zeta six",
    ]
    topics_list = [0, 0, 0, 1, 1, 1]

    model, c_tf_idf, documents, topics = _build_minimal_model(docs, topics_list)

    _, _, _, repr_docs_ids = model._extract_representative_docs(
        c_tf_idf, documents, topics, nr_samples=500, nr_repr_docs=2
    )

    for topic_id, doc_ids in zip(topics.keys(), repr_docs_ids):
        assert len(doc_ids) == 2, f"Topic {topic_id} should have 2 doc_ids, got {len(doc_ids)}"


def test_doc_ids_are_valid_dataframe_indices():
    """All returned doc_ids should be valid indices into the original DataFrame."""
    docs = [
        "shared text",
        "unique topic zero content",
        "shared text",
        "unique topic one content",
        "more topic one docs",
    ]
    topics_list = [0, 0, 1, 1, 1]

    model, c_tf_idf, documents, topics = _build_minimal_model(docs, topics_list)

    _, _, _, repr_docs_ids = model._extract_representative_docs(
        c_tf_idf, documents, topics, nr_samples=500, nr_repr_docs=3
    )

    valid_indices = set(documents.index.tolist())
    for doc_ids in repr_docs_ids:
        for doc_id in doc_ids:
            assert doc_id in valid_indices, f"doc_id {doc_id} not in DataFrame index"


def test_duplicate_text_with_diversity():
    """MMR branch should also map doc_ids correctly with duplicate text."""
    docs = [
        "machine learning algorithms applied",
        "machine learning methods used",
        "natural language processing tasks",
        "natural language understanding models",
    ]
    topics_list = [0, 0, 1, 1]

    model, c_tf_idf, documents, topics = _build_minimal_model(docs, topics_list)

    _, _, _, repr_docs_ids = model._extract_representative_docs(
        c_tf_idf,
        documents,
        topics,
        nr_samples=500,
        nr_repr_docs=2,
        diversity=0.5,
    )

    for topic_id, doc_ids in zip(topics.keys(), repr_docs_ids):
        for doc_id in doc_ids:
            actual_topic = documents.loc[doc_id, "Topic"]
            assert actual_topic == topic_id, (
                f"doc_id {doc_id} has topic {actual_topic} but assigned to topic {topic_id}"
            )

    for topic_id, doc_ids in zip(topics.keys(), repr_docs_ids):
        # With positional indexing, doc_ids count should equal nr_repr_docs
        # (or fewer if topic has fewer docs)
        assert len(doc_ids) == 2, f"Topic {topic_id} should have 2 doc_ids, got {len(doc_ids)}"
