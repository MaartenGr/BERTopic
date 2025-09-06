"""
Tests for zero-shot topic modeling functionality.

This module tests various aspects of zero-shot topic modeling, including
edge cases with the nr_topics parameter and topic_sizes_ consistency.
"""

from bertopic import BERTopic
from umap import UMAP


def test_zeroshot_with_nr_topics():
    """Test zero-shot topic modeling with nr_topics parameter."""
    docs = [
        "This is about machine learning and artificial intelligence",
        "Deep learning neural networks are powerful",
        "Python programming for data science",
        "Machine learning algorithms and models",
        "Artificial intelligence and deep learning",
        "Data science with Python programming",
        "Neural networks and machine learning",
        "Programming in Python for AI",
        "Deep learning models and algorithms",
        "Artificial intelligence programming",
    ]

    zeroshot_topics = ["Technology and Programming"]

    topic_model = BERTopic(
        zeroshot_topic_list=zeroshot_topics, zeroshot_min_similarity=0.1, nr_topics=2, min_topic_size=2
    )

    topics, probs = topic_model.fit_transform(docs)

    # Verify topic_sizes_ is properly populated
    assert topic_model.topic_sizes_ is not None
    assert len(topic_model.topic_sizes_) > 0

    # Verify total document count matches
    total_in_sizes = sum(topic_model.topic_sizes_.values())
    assert total_in_sizes == len(docs)

    # Verify all topics are accounted for
    for topic in set(topics):
        assert topic in topic_model.topic_sizes_


def test_zeroshot_all_documents_assigned():
    """Test edge case where all documents are assigned to zero-shot topics."""
    docs = [
        "Technology is advancing rapidly",
        "Software development is important",
        "Programming languages are evolving",
        "Computer science research continues",
        "Digital transformation is happening",
        "Innovation in technology sector",
        "Software engineering best practices",
        "Modern programming techniques",
        "Computer systems and architecture",
        "Digital solutions and platforms",
        "Technology trends and developments",
        "Software design patterns",
        "Programming paradigms evolution",
        "Computing infrastructure advances",
        "Digital innovation strategies",
    ]

    zeroshot_topics = ["Technology"]
    umap_model = UMAP(n_neighbors=5, n_components=2, min_dist=0.0, metric="cosine", random_state=42)

    topic_model = BERTopic(
        zeroshot_topic_list=zeroshot_topics,
        zeroshot_min_similarity=0.05,
        nr_topics=2,
        min_topic_size=1,
        umap_model=umap_model,
    )

    topics, probs = topic_model.fit_transform(docs)

    # Verify all documents are accounted for
    total_in_sizes = sum(topic_model.topic_sizes_.values())
    assert total_in_sizes == len(docs)
    assert topic_model.topic_sizes_ is not None


def test_zeroshot_topic_info_consistency():
    """Test consistency between topic_sizes_ and get_topic_info()."""
    docs = [
        "AI and machine learning research",
        "Deep learning neural networks",
        "Neural network architectures",
        "Machine learning algorithms",
        "Artificial intelligence systems",
        "Deep learning models training",
        "Neural network optimization",
        "Machine learning applications",
        "AI research and development",
        "Deep learning frameworks",
    ]
    zeroshot_topics = ["Artificial Intelligence"]
    umap_model = UMAP(n_neighbors=5, n_components=2, min_dist=0.0, metric="cosine", random_state=42)

    topic_model = BERTopic(
        zeroshot_topic_list=zeroshot_topics,
        zeroshot_min_similarity=0.1,
        nr_topics=2,
        min_topic_size=1,
        umap_model=umap_model,
    )

    topics, probs = topic_model.fit_transform(docs)

    # Verify topic info consistency
    topic_info = topic_model.get_topic_info()
    assert not topic_info.empty
    assert topic_info.shape[0] > 0

    # Verify topic_sizes_ and topic_info are consistent
    topic_info_counts = dict(zip(topic_info.Topic, topic_info.Count))
    for topic_id, count in topic_model.topic_sizes_.items():
        assert topic_id in topic_info_counts
        assert topic_info_counts[topic_id] == count


def test_github_issue_2384_reproduction():
    """Test exact reproduction case from GitHub issue #2384."""
    # Exact reproduction case from GitHub issue #2384
    docs = ["I need help with my voucher", "Gift card not working", "Customer service was poor"] * 50
    zeroshot_topics = ["Voucher inquiries", "Gift card issues", "Customer service feedback"]

    model = BERTopic(
        zeroshot_topic_list=zeroshot_topics,
        zeroshot_min_similarity=-1,
        nr_topics=4,
    )

    topics, _ = model.fit_transform(docs)

    # Verify the fix
    assert model.topic_sizes_ is not None
    assert len(model.topic_sizes_) > 0

    # Verify get_topic_info() works
    topic_info = model.get_topic_info()
    assert not topic_info.empty
    assert topic_info.shape[0] > 0

    # Verify total document count matches
    total_docs_in_sizes = sum(model.topic_sizes_.values())
    assert total_docs_in_sizes == len(docs)

    # Verify topic_representations_ still works (no regression)
    assert model.topic_representations_ is not None
    assert len(model.topic_representations_) > 0


def test_zeroshot_nr_topics_consistency():
    """Test consistency between using nr_topics and not using it."""
    docs = ["I need help with my voucher", "Gift card not working", "Customer service was poor"] * 20
    zeroshot_topics = ["Voucher inquiries", "Gift card issues", "Customer service feedback"]

    # Test without nr_topics
    model_without = BERTopic(zeroshot_topic_list=zeroshot_topics, zeroshot_min_similarity=-1)
    topics_without, _ = model_without.fit_transform(docs)

    # Test with nr_topics
    model_with = BERTopic(zeroshot_topic_list=zeroshot_topics, zeroshot_min_similarity=-1, nr_topics=4)
    topics_with, _ = model_with.fit_transform(docs)

    # Both should have properly populated topic_sizes_
    assert model_without.topic_sizes_ is not None
    assert model_with.topic_sizes_ is not None

    # Both should have same total document count
    total_without = sum(model_without.topic_sizes_.values())
    total_with = sum(model_with.topic_sizes_.values())
    assert total_without == len(docs)
    assert total_with == len(docs)

    # Both should have working get_topic_info()
    info_without = model_without.get_topic_info()
    info_with = model_with.get_topic_info()
    assert not info_without.empty
    assert not info_with.empty
