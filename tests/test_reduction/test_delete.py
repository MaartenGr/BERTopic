import copy
import pytest


@pytest.mark.parametrize(
    "model",
    [
        ("kmeans_pca_topic_model"),
        ("base_topic_model"),
        ("custom_topic_model"),
        ("merged_topic_model"),
        ("reduced_topic_model"),
        ("online_topic_model"),
    ],
)
def test_delete(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    nr_topics = len(set(topic_model.topics_))
    length_documents = len(topic_model.topics_)
    
    print("\n" + "="*50)
    print(f"Testing model: {model}")
    print(f"Initial number of topics: {nr_topics}")
    print(f"Initial topics: {sorted(list(set(topic_model.topics_)))}")
    print(f"Number of documents: {length_documents}")
    print("="*50)

    # First deletion
    topics_to_delete = [1, 2]
    print(f"\nFirst deletion - attempting to delete topics: {topics_to_delete}")
    topic_model.delete_topics(topics_to_delete)
    
    print(f"Topics after first deletion: {sorted(list(set(topic_model.topics_)))}")
    print(f"Number of topics after first deletion: {len(set(topic_model.topics_))}")
    
    mappings = topic_model.topic_mapper_.get_mappings(list(topic_model.hdbscan_model.labels_))
    print(f"Topic mappings after first deletion: {mappings}")
    
    mapped_labels = [mappings[label] for label in topic_model.hdbscan_model.labels_]
    print(f"First 10 mapped labels: {mapped_labels[:10]}")
    print(f"First 10 model topics: {topic_model.topics_[:10]}")

    print("\nFirst deletion - Assertions:")
    print(f"Expected topics: {nr_topics - 2}, Actual topics: {len(set(topic_model.topics_))}")
    print(f"Expected documents: {length_documents}, Actual documents: {topic_model.get_topic_info().Count.sum()}")
    
    assert nr_topics == len(set(topic_model.topics_)) + 2
    assert topic_model.get_topic_info().Count.sum() == length_documents
    if model == "online_topic_model":
        assert mapped_labels == topic_model.topics_[950:]
    else:
        assert mapped_labels == topic_model.topics_

    # Find two existing topics for second deletion
    remaining_topics = sorted(list(set(topic_model.topics_)))
    remaining_topics = [t for t in remaining_topics if t != -1]  # Exclude outlier topic
    topics_to_delete = remaining_topics[:2]  # Take first two remaining topics
    
    print(f"\nSecond deletion - attempting to delete topics: {topics_to_delete}")
    print(f"All remaining topics before second deletion: {remaining_topics}")
    
    # Second deletion
    topic_model.delete_topics(topics_to_delete)
    
    print(f"Topics after second deletion: {sorted(list(set(topic_model.topics_)))}")
    print(f"Number of topics after second deletion: {len(set(topic_model.topics_))}")
    
    mappings = topic_model.topic_mapper_.get_mappings(list(topic_model.hdbscan_model.labels_))
    print(f"Topic mappings after second deletion: {mappings}")
    
    mapped_labels = [mappings[label] for label in topic_model.hdbscan_model.labels_]
    print(f"First 10 mapped labels: {mapped_labels[:10]}")
    print(f"First 10 model topics: {topic_model.topics_[:10]}")

    print("\nSecond deletion - Assertions:")
    print(f"Expected topics: {nr_topics - 4}, Actual topics: {len(set(topic_model.topics_))}")
    print(f"Expected documents: {length_documents}, Actual documents: {topic_model.get_topic_info().Count.sum()}")
    
    assert nr_topics == len(set(topic_model.topics_)) + 4
    assert topic_model.get_topic_info().Count.sum() == length_documents
    if model == "online_topic_model":
        assert mapped_labels == topic_model.topics_[950:]
    else:
        assert mapped_labels == topic_model.topics_
