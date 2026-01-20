from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from scipy.sparse import csr_matrix, vstack
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

    data: list[tuple[str, float]] = field(default_factory=list)

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

    def reset(self) -> None:
        """Clear the mapping (e.g., after re-fitting)."""
        self._mapping.clear()
        self._recent_mapping.clear()


@dataclass
class Topic:
    """A topic with multiple representations from different sources.
    Sources are strings identifying the method used (e.g., 'c-tf-idf', 'gemma3').
    """

    id: int

    # Representations
    representations: dict[str, TopicRepresentation] = field(default_factory=dict)
    representative_documents: list[str] | None = field(default_factory=list)
    representative_images: np.ndarray | None = field(default_factory=lambda: np.array([]))
    label: str | None = None

    # Data matrices
    embedding: np.ndarray = field(default_factory=lambda: np.array([]))
    c_tf_idf: csr_matrix = field(default_factory=lambda: csr_matrix([]))

    # Metadata
    topic_type: TopicType = TopicType.NORMAL
    nr_documents: int = 0

    def __post_init__(self):
        """Auto-assign OUTLIER type if id is -1."""
        if self.id == -1:
            self.topic_type = TopicType.OUTLIER

    def __getitem__(self, source: str) -> TopicRepresentation:
        """Get representation for a specific source, or default if not found."""
        return self.representations.get(source, TopicRepresentation())

    def __setitem__(self, source: str, rep: TopicRepresentation):
        """Set representation for a specific source."""
        self.representations[source] = rep

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
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    probabilities: np.ndarray | None = None

    # History of actions applied to this collection
    actions: list[TopicAction] = field(default_factory=list)

    # TODO: For hierarchical topics, indicates the level in the hierarchy
    level: int = 0

    def frequencies(self) -> dict[int, int]:
        """Get the number of documents for each topic."""
        return {topic.id: topic.nr_documents for topic in self.topics.values()}

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
        return iter(self.topics.values())

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
                label=label,
            )

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

    def map_predictions(self, predictions: list[int], from_original: bool) -> list[int]:
        """Map a list of original predictions to current IDs.

        Arguments:
            predictions: List of topic IDs to map.
            from_original: If True, map from original IDs to current.
                           If False, map from last applied mapping to current.
        """
        return [self.mapping.map(prediction, from_original=from_original) for prediction in predictions]

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
