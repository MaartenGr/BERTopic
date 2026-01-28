# ruff: noqa: E402
import yaml
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    yaml._warnings_enabled["YAMLLoadWarning"] = False
except (KeyError, AttributeError, TypeError):
    pass

import numpy as np
import pandas as pd

from tqdm import tqdm
from packaging import version
from scipy.sparse import csr_matrix
from scipy.cluster import hierarchy as sch

from typing import List, Callable


from sklearn import __version__ as sklearn_version
from sklearn.metrics.pairwise import cosine_similarity

# BERTopic
from bertopic._utils import (
    check_documents_type,
    validate_distance_matrix,
    select_topic_representation,
    get_unique_distances,
)


def hierarchical_topics(
    self,
    docs: List[str],
    use_ctfidf: bool = True,
    linkage_function: Callable[[csr_matrix], np.ndarray] | None = None,
    distance_function: Callable[[csr_matrix], csr_matrix] | None = None,
) -> pd.DataFrame:
    """Create a hierarchy of topics.

    To create this hierarchy, BERTopic needs to be already fitted once.
    Then, a hierarchy is calculated on the distance matrix of the c-TF-IDF or topic embeddings
    representation using `scipy.cluster.hierarchy.linkage`.

    Based on that hierarchy, we calculate the topic representation at each
    merged step. This is a local representation, as we only assume that the
    chosen step is merged and not all others which typically improves the
    topic representation.

    Arguments:
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

    Returns:
        hierarchical_topics: A dataframe that contains a hierarchy of topics
                                represented by their parents and their children

    Examples:
    ```python
    from bertopic import BERTopic
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(docs)
    hierarchical_topics = topic_model.hierarchical_topics(docs)
    ```

    A custom linkage function can be used as follows:

    ```python
    from scipy.cluster import hierarchy as sch
    from bertopic import BERTopic
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(docs)

    # Hierarchical topics
    linkage_function = lambda x: sch.linkage(x, 'ward', optimal_ordering=True)
    hierarchical_topics = topic_model.hierarchical_topics(docs, linkage_function=linkage_function)
    ```
    """
    check_documents_type(docs)
    if distance_function is None:
        distance_function = lambda x: 1 - cosine_similarity(x)

    if linkage_function is None:
        linkage_function = lambda x: sch.linkage(x, "ward", optimal_ordering=True)

    # Calculate distance
    embeddings = select_topic_representation(self.c_tf_idf_, self.topic_embeddings_, use_ctfidf)[0][
        self._outliers :
    ]
    X = distance_function(embeddings)
    X = validate_distance_matrix(X, embeddings.shape[0])

    # Use the 1-D condensed distance matrix as an input instead of the raw distance matrix
    Z = linkage_function(X)

    # Ensuring that the distances between clusters are unique otherwise the flatting of the hierarchy with
    # `sch.fcluster(...)` would produce incorrect values for "Topics" for these clusters
    if len(Z[:, 2]) != len(np.unique(Z[:, 2])):
        Z[:, 2] = get_unique_distances(Z[:, 2])

    # Calculate basic bag-of-words to be iteratively merged later
    documents = pd.DataFrame({"Document": docs, "ID": range(len(docs)), "Topic": self.topics_})
    documents_per_topic = documents.groupby(["Topic"], as_index=False).agg({"Document": " ".join})
    documents_per_topic = documents_per_topic.loc[documents_per_topic.Topic != -1, :]
    clean_documents = self._preprocess_text(documents_per_topic.Document.values)

    # Scikit-Learn Deprecation: get_feature_names is deprecated in 1.0
    # and will be removed in 1.2. Please use get_feature_names_out instead.
    if version.parse(sklearn_version) >= version.parse("1.0.0"):
        words = self.vectorizer_model.get_feature_names_out()
    else:
        words = self.vectorizer_model.get_feature_names()

    bow = self.vectorizer_model.transform(clean_documents)

    # Extract clusters
    hier_topics = pd.DataFrame(
        columns=[
            "Parent_ID",
            "Parent_Name",
            "Topics",
            "Child_Left_ID",
            "Child_Left_Name",
            "Child_Right_ID",
            "Child_Right_Name",
        ]
    )
    for index in tqdm(range(len(Z))):
        # Find clustered documents
        clusters = sch.fcluster(Z, t=Z[index][2], criterion="distance") - self._outliers
        nr_clusters = len(clusters)

        # Extract first topic we find to get the set of topics in a merged topic
        topic = None
        val = Z[index][0]
        while topic is None:
            if val - len(clusters) < 0:
                topic = int(val)
            else:
                val = Z[int(val - len(clusters))][0]
        clustered_topics = [i for i, x in enumerate(clusters) if x == clusters[topic]]

        # Group bow per cluster, calculate c-TF-IDF and extract words
        grouped = csr_matrix(bow[clustered_topics].sum(axis=0))
        c_tf_idf = self.ctfidf_model.transform(grouped)
        selection = documents.loc[documents.Topic.isin(clustered_topics), :]
        selection.Topic = 0
        words_per_topic = self._extract_words_per_topic(words, selection, c_tf_idf, calculate_aspects=False)

        # Extract parent's name and ID
        parent_id = index + len(clusters)
        parent_name = "_".join([x[0] for x in words_per_topic[0]][:5])

        # Extract child's name and ID
        Z_id = Z[index][0]
        child_left_id = Z_id if Z_id - nr_clusters < 0 else Z_id - nr_clusters

        if Z_id - nr_clusters < 0:
            child_left_name = "_".join([x[0] for x in self.get_topic(Z_id)][:5])
        else:
            child_left_name = hier_topics.iloc[int(child_left_id)].Parent_Name

        # Extract child's name and ID
        Z_id = Z[index][1]
        child_right_id = Z_id if Z_id - nr_clusters < 0 else Z_id - nr_clusters

        if Z_id - nr_clusters < 0:
            child_right_name = "_".join([x[0] for x in self.get_topic(Z_id)][:5])
        else:
            child_right_name = hier_topics.iloc[int(child_right_id)].Parent_Name

        # Save results
        hier_topics.loc[len(hier_topics), :] = [
            parent_id,
            parent_name,
            clustered_topics,
            int(Z[index][0]),
            child_left_name,
            int(Z[index][1]),
            child_right_name,
        ]

    hier_topics["Distance"] = Z[:, 2]
    hier_topics = hier_topics.sort_values("Parent_ID", ascending=False)
    hier_topics[["Parent_ID", "Child_Left_ID", "Child_Right_ID"]] = hier_topics[
        ["Parent_ID", "Child_Left_ID", "Child_Right_ID"]
    ].astype(str)

    return hier_topics
