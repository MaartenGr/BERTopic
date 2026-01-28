import numpy as np

from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.cluster import hierarchy as sch
from typing import Callable, TYPE_CHECKING
from sklearn.metrics.pairwise import cosine_similarity

# BERTopic
from bertopic._utils import (
    check_documents_type,
    validate_distance_matrix,
    select_topic_representation,
    get_unique_distances,
)
from bertopic._topics import TopicHierarchy, Topic, Keywords
from bertopic._corpus import Corpus

if TYPE_CHECKING:
    from bertopic import BERTopic


def hierarchical_topics(
    topic_model: "BERTopic",
    docs: list[str],
    use_ctfidf: bool = True,
    linkage_function: Callable[[csr_matrix], np.ndarray] | None = None,
    distance_function: Callable[[csr_matrix], csr_matrix] | None = None,
    use_representation_model: bool | str = False,
) -> TopicHierarchy:
    """Create a hierarchy of topics.

    To create this hierarchy, BERTopic needs to be already fitted once.
    Then, a hierarchy is calculated on the distance matrix of the c-TF-IDF or topic embeddings
    representation using `scipy.cluster.hierarchy.linkage`.

    Based on that hierarchy, we calculate the topic representation at each
    merged step. This is a local representation, as we only assume that the
    chosen step is merged and not all others which typically improves the
    topic representation.

    Arguments:
        topic_model: The BERTopic instance
        docs: The documents you used when calling either `fit` or `fit_transform`
        use_ctfidf: Whether to calculate distances between topics based on c-TF-IDF embeddings. If False, the
                    embeddings from the embedding model are used.
        linkage_function: The linkage function to use. Default is:
                            `lambda x: sch.linkage(x, 'ward', optimal_ordering=True)`
        distance_function: The distance function to use on the c-TF-IDF matrix. Default is:
                            `lambda x: 1 - cosine_similarity(x)`.
                            You can pass any function that returns either a square matrix of
                            shape (n_samples, n_samples) with zeros on the diagonal and
                            non-negative values or condensed distance matrix of shape
                            (n_samples * (n_samples - 1) / 2,) containing the upper
                            triangular of the distance matrix.
        use_representation_model: Whether to use the representation model to extract topic words.
                                  This will make the process slower but can yield better topic representations.
                                  If a string is provided, the representation model with that name will be used.

    Returns:
        hierarchy: A TopicHierarchy containing all topics (leaves and merged)
                   that can be cut at any level to get a Topics object.

    Examples:
    ```python
    from bertopic import BERTopic
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(docs)
    hierarchy = topic_model.hierarchical_topics(docs)

    # Get topics at a specific level
    topics_at_10 = hierarchy.get_topics(nr_topics=10)
    ```

    A custom linkage function can be used as follows:

    ```python
    from scipy.cluster import hierarchy as sch
    from bertopic import BERTopic
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(docs)

    # Hierarchical topics
    linkage_function = lambda x: sch.linkage(x, 'ward', optimal_ordering=True)
    hierarchy = topic_model.hierarchical_topics(docs, linkage_function=linkage_function)
    ```
    """
    check_documents_type(docs)
    if distance_function is None:
        distance_function = lambda x: 1 - cosine_similarity(x)

    if linkage_function is None:
        linkage_function = lambda x: sch.linkage(x, "ward", optimal_ordering=True)

    # Determine representation model parameters
    if isinstance(use_representation_model, str):
        aspect_to_return, fine_tune, calculate_aspects = use_representation_model, False, True
    elif use_representation_model is True:
        aspect_to_return, fine_tune, calculate_aspects = None, True, False
    else:
        aspect_to_return, fine_tune, calculate_aspects = None, False, False

    # Calculate distance matrix and linkage matrix (excluding outlier topic)
    embeddings = select_topic_representation(
        topic_model.c_tf_idf_, topic_model.topic_embeddings_, use_ctfidf
    )[0][topic_model._outliers :]
    X = distance_function(embeddings)
    X = validate_distance_matrix(X, embeddings.shape[0])
    Z = linkage_function(X)

    # Ensure distances are unique for correct fcluster behavior
    if len(Z[:, 2]) != len(np.unique(Z[:, 2])):
        Z[:, 2] = get_unique_distances(Z[:, 2])

    # Get leaf topic IDs (excluding outlier)
    leaf_topic_ids = [tid for tid in sorted(topic_model.topic_sizes_.keys()) if tid != -1]
    n_leaves = len(leaf_topic_ids)

    # Bag-of-words
    documents_per_topic = Corpus(
        documents=docs, topics=np.array(topic_model.topics_)
    ).group_documents_by_topic()
    clean_documents = topic_model._preprocess_text([documents_per_topic[tid] for tid in leaf_topic_ids])
    words = topic_model.vectorizer_model.get_feature_names_out()
    bow = topic_model.vectorizer_model.transform(clean_documents)

    # Initialize hierarchy with leaf topics
    hierarchy = TopicHierarchy(
        linkage_matrix=Z,
        n_leaves=n_leaves,
        _original_predictions=np.array(topic_model.topics_),
        _original_probabilities=topic_model.probabilities_,
    )

    # Add outlier topic if it exists
    if -1 in topic_model.topic_sizes_:
        embedding = (
            topic_model.topic_embeddings_[0] if topic_model.topic_embeddings_ is not None else np.array([])
        )
        hierarchy.outlier_topic = Topic(
            id=-1,
            representations={"Main": Keywords(data=topic_model.get_topic(-1))},
            c_tf_idf=topic_model.c_tf_idf_[0] if topic_model._outliers else csr_matrix([]),
            embedding=embedding,
            nr_documents=topic_model.topic_sizes_[-1],
        )

    # Add leaf topics (IDs 0 to n_leaves-1)
    for topic_id in leaf_topic_ids:
        embedding = (
            topic_model.topic_embeddings_[topic_id + topic_model._outliers]
            if topic_model.topic_embeddings_ is not None
            else np.array([])
        )
        hierarchy.nodes[topic_id] = Topic(
            id=topic_id,
            representations={"Main": Keywords(data=topic_model.get_topic(topic_id))},
            c_tf_idf=topic_model.c_tf_idf_[topic_id + topic_model._outliers],
            embedding=embedding,
            nr_documents=topic_model.topic_sizes_[topic_id],
            leaf_topic_ids=[topic_id],
        )

    # Build merged topics from linkage matrix
    for index in tqdm(range(len(Z))):
        left_child_id = int(Z[index][0])
        right_child_id = int(Z[index][1])
        merge_distance = Z[index][2]
        new_id = n_leaves + index

        # Get child topics and merged topic
        left_child = hierarchy.nodes[left_child_id]
        right_child = hierarchy.nodes[right_child_id]
        merged_leaf_ids = left_child.leaf_topic_ids + right_child.leaf_topic_ids

        # Compute c-TF-IDF for merged topic
        grouped_bow = csr_matrix(bow[merged_leaf_ids].sum(axis=0))
        c_tf_idf = topic_model.ctfidf_model.transform(grouped_bow)

        # Select documents belonging to any of the merged leaf topics
        merged_leaf_set = set(merged_leaf_ids)
        topics_array = np.array(topic_model.topics_)
        doc_indices = np.where(np.isin(topics_array, list(merged_leaf_set)))[0]
        merged_docs = [docs[index] for index in doc_indices]

        # Create corpus with merged documents (all assigned to topic 0)
        merged_corpus = Corpus(documents=merged_docs, topics=np.zeros(len(merged_docs), dtype=int))
        words_per_topic = topic_model._extract_words_per_topic(
            words=words,
            corpus=merged_corpus,
            c_tf_idf=c_tf_idf,
            fine_tune=fine_tune,
            calculate_aspects=calculate_aspects,
            aspect_to_return=aspect_to_return,
        )

        # Compute weighted average embedding
        total_docs = left_child.nr_documents + right_child.nr_documents
        embedding = (
            left_child.embedding * left_child.nr_documents + right_child.embedding * right_child.nr_documents
        ) / total_docs

        # Create merged topic
        hierarchy.nodes[new_id] = Topic(
            id=new_id,
            representations={"Main": words_per_topic.get(0, Keywords())},
            c_tf_idf=c_tf_idf,
            embedding=embedding,
            nr_documents=total_docs,
            parent_id=None,  # Will be set if this gets merged later
            child_ids=(left_child_id, right_child_id),
            merge_distance=merge_distance,
            leaf_topic_ids=merged_leaf_ids,
        )

        # Set parent_id on children
        left_child.parent_id = new_id
        right_child.parent_id = new_id

    return hierarchy
