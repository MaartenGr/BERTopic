from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import polars as pl
from scipy.sparse import csr_matrix, vstack
from scipy.cluster.hierarchy import fcluster
import numpy as np
from collections import defaultdict


@dataclass
class TopicRepresentation(ABC):
    """Base class for all topic representations."""

    data: Any = None

    def __str__(self) -> str:
        """String representation of the topic representation."""
        return str(self.data)


@dataclass
class Keywords(TopicRepresentation):
    """Weighted keywords representing a topic."""

    data: list[tuple[str, float]] = field(default_factory=lambda: [("", 1e-05)])

    def top_n(self, n: int = 10) -> list[tuple[str, float]]:
        """Get the top N keywords by score."""
        return sorted(self.data, key=lambda x: x[1], reverse=True)[:n]

    @property
    def words(self) -> list[str]:
        """Get just the keyword strings."""
        return [word for word, _ in self.data]

    @property
    def scores(self) -> list[float]:
        """Get just the keyword scores."""
        return [score for _, score in self.data]

    def __str__(self) -> str:
        """String representation of the top 5 keywords."""
        return ", ".join([word for word, _ in self.top_n(5)])


@dataclass
class Label(TopicRepresentation):
    """A single descriptive label for a topic."""

    data: str = ""


@dataclass
class StructuredJSON(TopicRepresentation):
    """A structured JSON representation for a topic."""

    data: dict[str, str] = field(default_factory=dict)


@dataclass
class Metadata(TopicRepresentation):
    """Arbitrary metadata for a topic (sentiment, scores, etc.)."""

    data: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """Get metadata value by key."""
        return self.data.get(key)

    def __setitem__(self, key: str, value: Any):
        """Set metadata value by key."""
        self.data[key] = value


class TopicAction(str, Enum):
    """Enumeration of possible lifecycle actions for topics.
    Inheriting from str allows for easy printing and JSON serialization.
    """

    INITIALIZED = "initialized"
    MERGED = "merged"
    OUTLIERS_REDUCED = "outliers_reduced"
    HIERARCHICAL = "hierarchical_calculation"
    UPDATED = "updated"
    DELETED = "deleted"
    REDUCED = "reduced"


class TopicType(str, Enum):
    """Enumeration of topic types."""

    NORMAL = "normal"
    ZERO_SHOT = "zero_shot"
    OUTLIER = "outlier"


@dataclass
class TopicMapping:
    """Tracks cumulative topic ID transformations from original to current state.
    Also keeps track of the most recent mapping applied.
    """

    _mapping: dict[int, int] = field(default_factory=dict)
    _recent_mapping: dict[int, int] = field(default_factory=dict)

    def apply(self, new_mapping: dict[int, int]) -> None:
        """Compose a new mapping: original -> current becomes original -> new_current."""
        if not self._mapping:
            self._mapping = new_mapping.copy()
            self._recent_mapping = new_mapping.copy()
        else:
            self._mapping = {original: new_mapping[current] for original, current in self._mapping.items()}
            self._recent_mapping = new_mapping.copy()

    def map(self, topic_id: int, from_original: bool = True) -> int:
        """Map an ID to its current ID.

        Arguments:
            topic_id: The topic ID to map.
            from_original: If True, map from the original ID to current.
                           If False, map from the last applied mapping to current.
        """
        # Map from the original IDs
        if from_original:
            try:
                new_id = self._mapping.get(topic_id, topic_id)
                return new_id
            except KeyError:
                return topic_id

        # Map from the last applied mapping
        else:
            try:
                new_id = self._recent_mapping.get(topic_id, topic_id)
                return new_id
            except KeyError:
                return topic_id

    def map_predictions(self, predictions: list[int], from_original: bool = True) -> list[int]:
        """Map a list of original predictions to current IDs."""
        if from_original:
            return [self.map(prediction, from_original=True) for prediction in predictions]
        else:
            return [self.map(prediction, from_original=False) for prediction in predictions]

    def map_probabilities(self, probabilities: np.ndarray, from_original: bool = True) -> np.ndarray:
        """Map a 2D array of probabilities to current IDs."""
        if from_original:
            mapped_probs = np.zeros((probabilities.shape[0], len(set(self._mapping.values()))))
            for original_id, current_id in self._mapping.items():
                if original_id >= 0 and current_id >= 0:
                    mapped_probs[:, current_id] = probabilities[:, original_id]
            return mapped_probs
        else:
            mapped_probs = np.zeros((probabilities.shape[0], len(set(self._recent_mapping.values()))))
            for last_id, current_id in self._recent_mapping.items():
                if last_id >= 0 and current_id >= 0:
                    mapped_probs[:, current_id] = probabilities[:, last_id]
            return mapped_probs

    def reset(self) -> None:
        """Clear the mapping (e.g., after re-fitting)."""
        self._mapping.clear()
        self._recent_mapping.clear()


@dataclass
class Topic:
    """A topic with multiple representations from different sources.
    Sources are strings identifying the method used (e.g., 'c-tf-idf', 'gemma3').

    Attributes:
        id: Unique identifier for the topic.
        representations: A dictionary mapping source names to TopicRepresentation instances.
        representative_documents: A list of representative documents for the topic.
        representative_images: An array of representative images for the topic.
        label: A human-readable label for the topic.
        embedding: The embedding vector representing the topic.
        c_tf_idf: The c-TF-IDF vector representing the topic.
        topic_type: The type of topic (NORMAL, ZERO_SHOT, OUTLIER).
        nr_documents: The number of documents assigned to this topic.
        parent_id: The ID of the parent topic in a hierarchy (None for root).
        child_ids: A tuple of (left_child_id, right_child_id) for merged topics (None for leaves).
        merge_distance: The distance at which this topic was merged (None for leaves).
        leaf_topic_ids: A list of original leaf topic IDs contained in this node.
    """

    id: int

    # Representations
    representations: dict[str, TopicRepresentation] = field(default_factory=dict)
    representative_documents: list[str] | None = field(default_factory=list)
    representative_images: np.ndarray | None = field(default_factory=lambda: np.array([]))
    _label: str | None = None

    # Data matrices
    embedding: np.ndarray = field(default_factory=lambda: np.array([]))
    c_tf_idf: csr_matrix = field(default_factory=lambda: csr_matrix([]))

    # Metadata
    topic_type: TopicType = TopicType.NORMAL
    nr_documents: int = 0

    # Hierarchy fields
    parent_id: int | None = None
    child_ids: tuple[int, int] | None = None
    merge_distance: float | None = None
    leaf_topic_ids: list[int] = field(default_factory=list)

    def __post_init__(self):
        # Set outlier
        if self.id == -1:
            self.topic_type = TopicType.OUTLIER

        # Leaves contain only themselves (ignoring the outlier)
        if not self.leaf_topic_ids and self.child_ids is None and self.id >= 0:
            self.leaf_topic_ids = [self.id]

    @property
    def is_leaf(self) -> bool:
        """Check if this topic is a leaf (not merged from other topics)."""
        return self.child_ids is None

    @property
    def label(self) -> str | None:
        """Get the label of the topic."""
        # Set label representation if no is provided
        if self._label is None:
            representation = self.representations.get("Main")
            if isinstance(representation, Label):
                self._label = representation.data
            elif isinstance(representation, Keywords):
                self._label = "_".join(representation.words[:4])
            else:
                self._label = "No label could be created"

        return self._label

    def __getitem__(self, source: str) -> TopicRepresentation:
        """Get representation for a specific source, or default if not found."""
        return self.representations.get(source, TopicRepresentation())

    def __setitem__(self, source: str, rep: TopicRepresentation):
        """Set representation for a specific source."""
        self.representations[source] = rep

    def to_dict(self) -> dict:
        """Serialize topic info to a flat dictionary for tabular output."""
        info = {"Topic": self.id, "Count": self.nr_documents, "Name": self.label}
        info["Representation"] = str(self.representations.get("Main"))

        # Extract all other representations
        for name, rep in self.representations.items():
            if name != "Main":
                info[name] = str(rep)

        # Representative documents and images
        info["Representative_Docs"] = self.representative_documents if self.representative_documents else [""]
        if self.representative_images is not None and self.representative_images.size > 0:
            info["Representative_Images"] = self.representative_images

        return info

    def __str__(self) -> str:
        """Pretty print all representations of the topic."""
        lines = [f"Topic {self.id} Representations:"]
        lines.append("=" * 50)
        lines.append("\n")
        for source, rep in self.representations.items():
            lines.append(f"  {source}: {rep}")
        return "\n".join(lines)


@dataclass
class Topics:
    """A collection of topics with history tracking."""

    # Topics metadata
    topics: dict[int, Topic] = field(default_factory=dict)
    mapping: TopicMapping = field(default_factory=TopicMapping)

    # Document metadata (optional)
    # NOTE: These are the original predictions/probabilities from the
    # clustering algorithm before any remapping.
    _original_predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    _original_probabilities: np.ndarray | None = None

    # Zero-shot probabilities (optional)
    _zeroshot_probabilities: np.ndarray | None = None

    # History of actions applied to this collection
    actions: list[TopicAction] = field(default_factory=list)

    def frequencies(self) -> dict[int, int]:
        """Get the number of documents for each topic."""
        return {topic.id: topic.nr_documents for topic in self.topics.values()}

    @property
    def labels(self) -> dict[int, str]:
        """Get labels for all topics."""
        labels = {topic.id: topic.label if topic.label is not None else "" for topic in self.topics.values()}
        return dict(sorted(labels.items()))

    @property
    def predictions(self) -> np.ndarray | None:
        """Get original predictions."""
        return self.mapping.map_predictions(self._original_predictions.tolist(), from_original=True)

    @property
    def probabilities(self) -> np.ndarray | None:
        """Get current probabilities."""
        if self._zeroshot_probabilities is not None:
            return self._zeroshot_probabilities
        elif self._original_probabilities is not None:
            if self._original_probabilities.ndim == 1:
                return self._original_probabilities
            elif self._original_probabilities.ndim == 2:
                return self.mapping.map_probabilities(self._original_probabilities, from_original=True)

    @property
    def c_tf_idf(self) -> csr_matrix:
        """Get c-TF-IDF matrix for all topics in ascending topic ID order
        without casting to dense format.
        """
        sorted_ids = sorted(self.topic_ids())
        return vstack([self.topics[topic_id].c_tf_idf for topic_id in sorted_ids])

    @property
    def embeddings(self) -> np.ndarray:
        """Get embeddings matrix for all topics in ascending topic ID order."""
        sorted_ids = sorted(self.topic_ids())
        return np.array([self.topics[topic_id].embedding for topic_id in sorted_ids])

    @property
    def unique_ids(self) -> list[int]:
        """Get list of unique topic IDs."""
        return sorted(list(self.topics.keys()))

    def topic_ids(self, outliers: bool = True) -> list[int]:
        """Get a list of all topic IDs in the collection."""
        if outliers:
            return sorted(list(self.topics.keys()))
        else:
            return sorted([topic_id for topic_id in self.topics.keys() if topic_id != -1])

    def __getitem__(self, topic_id: int) -> Topic:
        """Get topic by ID."""
        if topic_id not in self.topics:
            raise KeyError(f"Topic ID {topic_id} not found.")
        return self.topics[topic_id]

    def __iter__(self):
        """Iterate over all topics."""
        sorted_ids = sorted(self.topics.keys())
        for topic_id in sorted_ids:
            yield self.topics[topic_id]

    def __len__(self):
        """Get number of topics."""
        return len(self.topics)

    def __str__(self) -> str:
        """Pretty print all topics."""
        lines = [f"Topics (total: {len(self.topics)})"]
        lines.append("=" * 50)

        # # Show pipeline history
        # if self.actions:
        #     lines.append(f"Pipeline: {' -> '.join([h.value for h in self.actions])}")

        # Print each topic
        for topic_id, topic in sorted(self.topics.items()):
            lines.append(f"\nTopic {topic_id}:")
            for source, rep in topic.representations.items():
                lines.append(f"  {source}: {rep}")

        return "\n".join(lines)

    def get(self, topic) -> Topic:
        """Get topic by ID, raising KeyError if not found."""
        if self.topics.get(topic):
            return self.topics[topic]
        else:
            raise KeyError(f"Topic ID {topic} not found.")

    def initialize(
        self,
        predictions: list[int] | np.ndarray = None,
        zeroshot_labels: list[str] | None = None,
        topic_type: TopicType = TopicType.NORMAL,
    ):
        """Initialize topics from clustering predictions."""
        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()

        for topic_id in set(predictions):
            if zeroshot_labels and topic_id in range(len(zeroshot_labels)):
                topic_type_to_use = TopicType.ZERO_SHOT
                label = zeroshot_labels[topic_id]
            else:
                topic_type_to_use = topic_type
                label = None

            # Create Topic instance
            self.topics[topic_id] = Topic(
                id=topic_id,
                topic_type=topic_type_to_use,
                nr_documents=predictions.count(topic_id),
                _label=label,
            )

        # Set original predictions/probabilities
        self._original_predictions = np.array(predictions)

        # Log the initialization
        self.add_action(TopicAction.INITIALIZED)

        return self

    def add_action(self, action: TopicAction):
        """Register a completed action in the history log."""
        self.actions.append(action)

    def has_action(self, action: TopicAction) -> bool:
        """Check if a specific action has ever been applied."""
        return action in self.actions

    @property
    def outlier_exists(self) -> bool:
        """Check if outlier topic (-1) exists."""
        return True if -1 in self.topics else False

    def set_data(
        self,
        embeddings: np.ndarray | None = None,
        c_tf_idf: csr_matrix | None = None,
        representations: dict[str, dict[int, TopicRepresentation]] | None = None,
        representative_documents: dict[int, list[str]] | None = None,
    ) -> None:
        """Bulk set attributes for topics. Matrices are in ascending topic ID order."""
        sorted_ids = sorted(self.topic_ids())

        # Set embeddings
        if embeddings is not None:
            for idx, topic_id in enumerate(sorted_ids):
                self.topics[topic_id].embedding = embeddings[idx]

        # Set c-TF-IDF vectors
        if c_tf_idf is not None:
            for idx, topic_id in enumerate(sorted_ids):
                self.topics[topic_id].c_tf_idf = c_tf_idf[idx]

        # Set representations
        if representations is not None:
            for source, reps in representations.items():
                for topic_id, rep in reps.items():
                    if topic_id in self.topics:
                        self.topics[topic_id][source] = rep

        # Set representative documents
        if representative_documents is not None:
            for topic_id, docs in representative_documents.items():
                if topic_id in self.topics:
                    self.topics[topic_id].representative_documents = docs

    def sort_by_frequency(self) -> "Topics":
        """Sort topics by frequency: outlier (-1) -> zero-shot (original order) -> normal (freq desc)."""
        frequencies = self.frequencies()

        # Partition topics by type (OUTLIER, ZERO_SHOT, NORMAL)
        outlier = None
        zeroshot, normal = [], []
        for topic in self:
            if topic.topic_type == TopicType.OUTLIER or topic.id == -1:
                outlier = topic
            elif topic.topic_type == TopicType.ZERO_SHOT:
                zeroshot.append(topic)
            else:
                normal.append(topic)
        normal.sort(key=lambda t: frequencies.get(t.id, 0), reverse=True)

        # Build old -> new mapping
        old_to_new = {outlier.id: -1} if outlier else {}
        for new_id, topic in enumerate(zeroshot + normal):
            old_to_new[topic.id] = new_id

        self.remap(old_to_new)

        return self

    def add_topics(self, other: "Topics") -> None:
        """Combine another Topics collection into this one.

        The other's topic IDs are remapped to start after this collection's max ID.
        Outlier topics (-1) are merged (other's outlier predictions join this outlier).
        """
        # Find the next available ID (after max, excluding -1)
        next_id = max((self.topic_ids(outliers=False)), default=-1) + 1

        # Build mapping from other's IDs to new IDs
        old_to_new = {}
        for old_topic in other.topics.values():
            if old_topic.id == -1:
                old_to_new[-1] = -1
                # Merge outlier if we already have one, otherwise add it
                if -1 not in self.topics:
                    self.topics[-1] = old_topic
            else:
                old_to_new[old_topic.id] = next_id
                old_topic.id = next_id
                self.topics[next_id] = old_topic
                next_id += 1

        # Sort topics
        self.topics = dict(sorted(self.topics.items()))
        self.sort_by_frequency()
        self.mapping.reset()

        # Register the merge action
        self.add_action(TopicAction.MERGED)

    def get_topics(self, topic_ids: list[int]) -> "Topics":
        """Get a subset of topics by their IDs as a new Topics collection."""
        return Topics(
            topics={tid: self.topics[tid] for tid in topic_ids if tid in self.topics},
            actions=self.actions.copy(),
        )

    def remap(self, old_to_new: dict[int, int]) -> None:
        """Apply an ID remapping to topics and update cumulative mapping.
        This is expected to be a one to one mapping.
        """
        for topic in self.topics.values():
            topic.id = old_to_new[topic.id]
        self.topics = {topic.id: topic for topic in self.topics.values()}
        self.mapping.apply(old_to_new)

    def merge(self, old_to_new: dict[int, int]) -> None:
        """Merge topics and update cumulative mapping.
        This is expected to be a many to one mapping (e.g., merging topics).

        This will do the following:

            * Combine topic embeddings by weighted average based on nr_documents
            * Combine c-TF-IDF vectors by weighted average based on nr_documents
            * Sum nr_documents for merged topics
            * Keep the representations of the most prominent topic (highest nr_documents)
            * Combine representative documents from all merged topics
            * Make TopicType NORMAL if any merged topic is NORMAL
            * Map original IDs to new IDs in the cumulative mapping

        Arguments:
            old_to_new: A dictionary mapping old topic IDs to new topic IDs.
                        New topics are fewer than old topics.
        """
        # Group old topic IDs by their new target ID
        new_to_old: dict[int, list[int]] = defaultdict(list)
        for old_id, new_id in old_to_new.items():
            new_to_old[new_id].append(old_id)

        merged_topics = {}
        for new_id, old_ids in new_to_old.items():
            old_topics = [self.topics[old_id] for old_id in old_ids]
            total_docs = sum(t.nr_documents for t in old_topics)

            # Calculate weights (handle zero documents edge case)
            weights = np.array([t.nr_documents / total_docs for t in old_topics])

            # Weighted average of embeddings
            if old_topics[0].embedding.size > 0:
                embedding = np.average([t.embedding for t in old_topics], weights=weights, axis=0)

            # Weighted average of c-TF-IDF vectors (sparse)
            c_tf_idf = sum(t.c_tf_idf * w for t, w in zip(old_topics, weights))

            # Keep representations from most prominent topic
            prominent = max(old_topics, key=lambda t: t.nr_documents)

            # Combine representative documents from all topics
            rep_docs = [doc for t in old_topics for doc in (t.representative_documents or [])]

            # NORMAL if any merged topic is NORMAL
            # TODO: Think about improving this for MERGED/ZEROSHOT instances.
            topic_type = (
                TopicType.NORMAL
                if any(t.topic_type == TopicType.NORMAL for t in old_topics)
                else prominent.topic_type
            )

            merged_topics[new_id] = Topic(
                id=new_id,
                representations=prominent.representations.copy(),
                representative_documents=rep_docs,
                representative_images=prominent.representative_images,
                embedding=embedding,
                c_tf_idf=c_tf_idf,
                topic_type=topic_type,
                nr_documents=total_docs,
            )

        self.topics = merged_topics
        self.mapping.apply(old_to_new)
        self.add_action(TopicAction.MERGED)

    def delete(self, topics: list[int] | int) -> None:
        """Delete topics by mapping them to the outlier topic (-1).

        This will:
            * Create an outlier topic if it doesn't exist
            * Map deleted topic predictions to -1
            * Remove deleted Topic objects from the collection
            * Update outlier's nr_documents with the sum of deleted topic counts

        Arguments:
            topics: List of topic IDs to delete or a single topic ID.
        """
        if isinstance(topics, int):
            topics = {topics}
        else:
            topics = set(topics)

        topics = {topics} if isinstance(topics, int) else set(topics)

        # Calculate total documents being moved to outlier
        deleted_doc_count = sum(
            self.topics[topic_id].nr_documents for topic_id in topics if topic_id in self.topics
        )

        # Create outlier topic if it doesn't exist
        if -1 not in self.topics:
            sample_topic = next(iter(self.topics.values()))

            # If only one topic is deleted, we use its data for the outlier
            if len(topics) == 1:
                selected_topic = self.topics[next(iter(topics))]
                embedding = selected_topic.embedding
                c_tf_idf = selected_topic.c_tf_idf
                representative_documents = selected_topic.representative_documents
                representations = selected_topic.representations
            else:
                embedding_dim = sample_topic.embedding.shape[0]
                embedding = np.zeros(embedding_dim) if embedding_dim > 0 else np.array([])
                c_tf_idf_dim = sample_topic.c_tf_idf.shape[1]
                c_tf_idf = csr_matrix((1, c_tf_idf_dim)) if c_tf_idf_dim > 0 else csr_matrix([])
                representative_documents = [""]
                representations = {name: type(rep)() for name, rep in sample_topic.representations.items()}

            # Build outlier topic
            self.topics[-1] = Topic(
                id=-1,
                representations=representations,
                representative_documents=representative_documents,
                embedding=embedding,
                c_tf_idf=c_tf_idf,
                topic_type=TopicType.OUTLIER,
                nr_documents=deleted_doc_count,
            )
        else:
            self.topics[-1].nr_documents += deleted_doc_count

        # Build mapping: deleted -> -1, others -> themselves
        old_to_new = {topic_id: -1 if topic_id in topics else topic_id for topic_id in self.topics.keys()}
        old_to_new[-1] = -1

        # Remove deleted topics
        for topic_id in topics:
            if topic_id in self.topics:
                del self.topics[topic_id]

        self.mapping.apply(old_to_new)
        self.add_action(TopicAction.DELETED)

    def map_predictions(self, predictions: list[int], from_original: bool) -> list[int]:
        """Map a list of original predictions to current IDs.

        Arguments:
            predictions: List of topic IDs to map.
            from_original: If True, map from original IDs to current.
                           If False, map from last applied mapping to current.
        """
        return [self.mapping.map(prediction, from_original=from_original) for prediction in predictions]

    def map_probabilities(self, probabilities: np.ndarray, from_original: bool) -> np.ndarray:
        """Map a 2D array of probabilities to current IDs.

        Arguments:
            probabilities: 2D array of shape (n_samples, n_topics) to map.
            from_original: If True, map from original IDs to current.
                           If False, map from last applied mapping to current.
        """
        if probabilities is not None:
            return self.mapping.map_probabilities(probabilities, from_original=from_original)
        else:
            return None

    def get_mappings(self, from_original: bool = True) -> dict[int, int]:
        """Get the current topic ID mappings.

        Arguments:
            from_original: If True, get mapping from original IDs to current.
                           If False, get mapping from last applied mapping to current.
        """
        if from_original:
            return self.mapping._mapping.copy()
        else:
            return self.mapping._recent_mapping.copy()

    def to_polars(self, topic: int | None = None) -> pl.DataFrame:
        """Convert topic info to a polars DataFrame."""
        if topic is not None:
            selected_topic = self.topics.get(topic)
            rows = [selected_topic.to_dict()] if selected_topic else []
        else:
            rows = [topic.to_dict() for topic in self]

        columns = list(rows[0].keys())
        data = {col: [row.get(col) for row in rows] for col in columns}
        return pl.DataFrame(data)


@dataclass
class TopicHierarchy:
    """Binary tree of topics from hierarchical clustering.

    Node IDs follow scipy's linkage convention. Example with 5 leaf topics:

    ```
    Leaf IDs:   0   1   2   3   4      (IDs 0 to n_leaves-1)
                │   │   │   │   │
                └─┬─┘   │   └─┬─┘
                  │     │     │
    Merge 0:      5     │     │        (creates ID n_leaves = 5)
                  │     │     │
                  └──┬──┘     │
                     │        │
    Merge 1:         6        │        (creates ID n_leaves+1 = 6)
                     │        │
                     └───┬────┘
                         │
    Merge 2:             7             (creates ID n_leaves+2 = 7)
                         │
    Merge 3:             8             (root = ID 2*n_leaves-2 = 8)
    ```

    The outlier topic (-1) is stored separately and is not part of the tree.

    Attributes:
        nodes: A dictionary mapping node IDs to Topic instances.
        linkage_matrix: The scipy linkage matrix used to create the hierarchy.
        n_leaves: The number of leaf topics (excluding the outlier topic).
        outlier_topic: The outlier topic (if any). Not part of the hierarchy tree.
        _original_predictions: Original document-to-leaf-topic predictions from fit.
        _original_probabilities: Original document-to-leaf-topic probabilities from fit.
    """

    # Hierarchy structure
    nodes: dict[int, Topic] = field(default_factory=dict)
    linkage_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    n_leaves: int = 0

    # The outlier topic is not part of the hierarchy
    outlier_topic: Topic | None = None

    # Original values from fitting
    _original_predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    _original_probabilities: np.ndarray | None = None

    @property
    def root(self) -> Topic:
        """Root node (everything merged into one) excluding the outlier Topic."""
        root_id = 2 * self.n_leaves - 2
        return self.nodes[root_id]

    @property
    def leaves(self) -> list[Topic]:
        """Leaf topics (most granular level) excluding the outlier Topic."""
        return [self.nodes[i] for i in range(self.n_leaves)]

    def get_topics(self, nr_topics: int) -> Topics:
        """Cut the hierarchy to get a Topics object with `nr_topics`.

        Arguments:
            nr_topics: Desired number of topics (excluding the outlier topic).

        Returns:
            A Topics object at this granularity.
        """
        nr_topics = min(nr_topics, self.n_leaves)

        if nr_topics >= self.n_leaves:
            selected_ids = list(range(self.n_leaves))
        else:
            # fcluster assigns each leaf to a cluster (1-indexed)
            clusters = fcluster(self.linkage_matrix, t=nr_topics, criterion="maxclust")

            # Group leaves by cluster assignment
            cluster_to_leaves: dict[int, list[int]] = {}
            for leaf_id, cluster in enumerate(clusters):
                cluster_to_leaves.setdefault(cluster, []).append(leaf_id)

            # For each cluster, find the appropriate node
            selected_ids = []
            for leaf_ids in cluster_to_leaves.values():
                if len(leaf_ids) == 1:
                    selected_ids.append(leaf_ids[0])
                else:
                    lca = self._find_lowest_common_ancestor(leaf_ids)
                    selected_ids.append(lca)

        return self._build_topics(selected_ids)

    def _find_lowest_common_ancestor(self, leaf_ids: list[int]) -> int:
        """Find the lowest common ancestor of a set of leaves."""
        # Get all ancestors of first leaf
        ancestors = set()
        node_id = leaf_ids[0]
        while node_id is not None:
            ancestors.add(node_id)
            node_id = self.nodes[node_id].parent_id

        # Intersect with ancestors of other leaves
        for leaf_id in leaf_ids[1:]:
            leaf_ancestors = set()
            node_id = leaf_id
            while node_id is not None:
                leaf_ancestors.add(node_id)
                node_id = self.nodes[node_id].parent_id
            ancestors &= leaf_ancestors

        # lowest common ancestor is the ancestor with highest ID (most recent merge)
        return max(ancestors)

    def _build_topics(self, selected_ids: list[int]) -> Topics:
        """Build a Topics object selected IDs."""
        # Sort selected topics by document count (descending)
        selected_topics = [self.nodes[id] for id in selected_ids]
        selected_topics.sort(key=lambda t: t.nr_documents, reverse=True)

        # Build mapping: original_leaf_id -> new_topic_id
        old_to_new = {-1: -1}
        for new_id, topic in enumerate(selected_topics):
            for leaf_id in topic.leaf_topic_ids:
                old_to_new[leaf_id] = new_id

        # Create Topics
        topics = Topics()
        topics._original_predictions = self._original_predictions.copy()
        topics._original_probabilities = (
            self._original_probabilities.copy() if self._original_probabilities is not None else None
        )
        topics.mapping.apply(old_to_new)

        # Add outlier topic
        if self.outlier_topic is not None:
            topics.topics[-1] = self._create_topic_copy(self.outlier_topic, -1)

        # Add selected topics with new IDs
        for new_id, topic in enumerate(selected_topics):
            topics.topics[new_id] = self._create_topic_copy(topic, new_id)

        topics.add_action(TopicAction.HIERARCHICAL)
        return topics

    def _create_topic_copy(self, topic: Topic, new_id: int) -> Topic:
        """Create a copy of a topic with a new ID (without hierarchy fields)."""
        return Topic(
            id=new_id,
            representations={k: v for k, v in topic.representations.items()},
            representative_documents=list(topic.representative_documents or []),
            embedding=topic.embedding.copy() if topic.embedding.size else np.array([]),
            c_tf_idf=topic.c_tf_idf.copy(),
            topic_type=topic.topic_type,
            nr_documents=topic.nr_documents,
            _label=topic._label,
        )

    def to_polars(self) -> "pl.DataFrame":
        """Convert hierarchy to a polars DataFrame."""
        rows = []
        for node_id in sorted(self.nodes.keys(), reverse=True):
            topic = self.nodes[node_id]
            if topic.child_ids is not None:
                left_id, right_id = topic.child_ids
                rows.append(
                    {
                        "Parent_ID": str(node_id),
                        "Parent_Name": topic.label,
                        "Topics": topic.leaf_topic_ids,
                        "Child_Left_ID": str(left_id),
                        "Child_Left_Name": self.nodes[left_id].label,
                        "Child_Right_ID": str(right_id),
                        "Child_Right_Name": self.nodes[right_id].label,
                        "Distance": topic.merge_distance,
                    }
                )

        return pl.DataFrame(rows)
