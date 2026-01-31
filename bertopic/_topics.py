from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from importlib.metadata import version

import polars as pl
from scipy.sparse import csr_matrix, vstack
from scipy.cluster.hierarchy import fcluster
import numpy as np
from collections import defaultdict

BERTOPIC_VERSION = version("bertopic")


@dataclass
class TopicRepresentation:
    """Base class for all topic representations."""

    data: Any = None

    def __str__(self) -> str:
        """String representation of the topic representation."""
        return str(self.data)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {"type": "base", "data": self.data}

    @classmethod
    def from_dict(cls, data: dict) -> "TopicRepresentation":
        """Deserialize from dictionary."""
        return cls(data=data.get("data"))


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

    def to_dict(self) -> dict:
        return {"type": "keywords", "data": [list(item) for item in self.data]}

    @classmethod
    def from_dict(cls, data: dict) -> "Keywords":
        return cls(data=[tuple(item) for item in data["data"]])


@dataclass
class Label(TopicRepresentation):
    """A single descriptive label for a topic."""

    data: str = ""

    def to_dict(self) -> dict:
        return {"type": "label", "data": self.data}

    @classmethod
    def from_dict(cls, data: dict) -> "Label":
        return cls(data=data["data"])


@dataclass
class StructuredJSON(TopicRepresentation):
    """A structured JSON representation for a topic."""

    data: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"type": "structured_json", "data": self.data}

    @classmethod
    def from_dict(cls, data: dict) -> "StructuredJSON":
        return cls(data=data["data"])


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

    def to_dict(self) -> dict:
        return {"type": "metadata", "data": self.data}

    @classmethod
    def from_dict(cls, data: dict) -> "Metadata":
        return cls(data=data["data"])


def representation_from_dict(d: dict) -> TopicRepresentation:
    """Deserialize a TopicRepresentation from a dictionary based on its type field."""
    type_map = {
        "keywords": Keywords,
        "label": Label,
        "structured_json": StructuredJSON,
        "metadata": Metadata,
        "base": TopicRepresentation,
    }
    rep_type = d.get("type", "base")
    cls = type_map.get(rep_type, TopicRepresentation)
    return cls.from_dict(d)


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
        if from_original:
            return self._mapping.get(topic_id, topic_id)
        else:
            return self._recent_mapping.get(topic_id, topic_id)

    def map_predictions(self, predictions: list[int], from_original: bool = True) -> list[int]:
        """Map a list of original predictions to current IDs."""
        return [self.map(p, from_original) for p in predictions]

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

    def add_new_topics(self, new_mappings: dict[int, int]) -> None:
        """Add mappings for newly discovered topics in online learning.

        Arguments:
            new_mappings: Mapping from new cluster IDs to new topic IDs
        """
        self._mapping.update(new_mappings)
        self._recent_mapping.update(new_mappings)

    def reset(self) -> None:
        """Clear the mapping (e.g., after re-fitting)."""
        self._mapping.clear()
        self._recent_mapping.clear()

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "mapping": {str(k): v for k, v in self._mapping.items()},
            "recent_mapping": {str(k): v for k, v in self._recent_mapping.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TopicMapping":
        """Deserialize from dictionary."""
        mapping = cls()
        mapping._mapping = {int(k): v for k, v in data.get("mapping", {}).items()}
        mapping._recent_mapping = {int(k): v for k, v in data.get("recent_mapping", {}).items()}
        return mapping

    def copy(self) -> "TopicMapping":
        """Create a copy of this mapping."""
        return TopicMapping.from_dict(self.to_dict())


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
    representative_documents: list[str] = field(default_factory=list)
    representative_images: np.ndarray = field(default_factory=lambda: np.array([]))
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
    def label(self) -> str:
        """Get the label of the topic.

        If a custom label was set via `_label`, return it.
        Otherwise, generate a label from the Main representation.
        """
        if self._label is not None:
            return self._label

        # Generate label from representation
        representation = self.representations.get("Main")
        if isinstance(representation, Label):
            return representation.data
        elif isinstance(representation, Keywords):
            return "_".join(representation.words[:4])
        else:
            return "No label could be created"

    def __getitem__(self, source: str) -> TopicRepresentation:
        """Get representation for a specific source, or default if not found."""
        return self.representations.get(source, TopicRepresentation())

    def __setitem__(self, source: str, rep: TopicRepresentation):
        """Set representation for a specific source."""
        self.representations[source] = rep

    def to_info_dict(self) -> dict:
        """Serialize topic info to a flat dictionary for tabular output."""
        info = {"Topic": self.id, "Count": self.nr_documents, "Name": self.label}
        info["Representation"] = str(self.representations.get("Main"))

        # Extract all other representations
        for name, rep in self.representations.items():
            if name != "Main":
                info[name] = str(rep)

        # Representative documents and images
        info["Representative_Docs"] = self.representative_documents if self.representative_documents else [""]
        if self.representative_images.size > 0:
            info["Representative_Images"] = self.representative_images

        return info

    def to_dict(self, full: bool = False) -> dict:
        """Serialize topic for storage.

        Arguments:
            full: If True, include embeddings and c_tf_idf (for in-memory copy).
                  If False, exclude large arrays (for disk serialization).
        """
        data = {
            "id": self.id,
            "label": self._label,
            "nr_documents": self.nr_documents,
            "topic_type": self.topic_type.value,
            "representations": {name: rep.to_dict() for name, rep in self.representations.items()},
            "representative_documents": self.representative_documents,
        }

        if full:
            data["embedding"] = self.embedding.tolist() if self.embedding.size else []
            if self.c_tf_idf.nnz > 0:
                data["c_tf_idf"] = {
                    "data": self.c_tf_idf.data.tolist(),
                    "indices": self.c_tf_idf.indices.tolist(),
                    "indptr": self.c_tf_idf.indptr.tolist(),
                    "shape": list(self.c_tf_idf.shape),
                }
            if self.representative_images.size:
                data["representative_images"] = self.representative_images.tolist()

        # Hierarchy fields (only if set)
        if self.parent_id is not None:
            data["parent_id"] = self.parent_id
        if self.child_ids is not None:
            data["child_ids"] = list(self.child_ids)
        if self.merge_distance is not None:
            data["merge_distance"] = self.merge_distance
        if self.leaf_topic_ids:
            data["leaf_topic_ids"] = self.leaf_topic_ids
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Topic":
        """Deserialize topic from storage."""
        representations = {
            name: representation_from_dict(rep_dict)
            for name, rep_dict in data.get("representations", {}).items()
        }

        # Handle full format fields
        embedding = np.array(data["embedding"]) if "embedding" in data else np.array([])
        c_tf_idf_data = data.get("c_tf_idf")
        if c_tf_idf_data:
            c_tf_idf = csr_matrix(
                (c_tf_idf_data["data"], c_tf_idf_data["indices"], c_tf_idf_data["indptr"]),
                shape=tuple(c_tf_idf_data["shape"]),
            )
        else:
            c_tf_idf = csr_matrix([])

        representative_images = (
            np.array(data["representative_images"]) if "representative_images" in data else np.array([])
        )

        return cls(
            id=data["id"],
            _label=data.get("label"),
            nr_documents=data.get("nr_documents", 0),
            topic_type=TopicType(data.get("topic_type", "normal")),
            representations=representations,
            representative_documents=data.get("representative_documents", []),
            embedding=embedding,
            c_tf_idf=c_tf_idf,
            representative_images=representative_images,
            parent_id=data.get("parent_id"),
            child_ids=tuple(data["child_ids"]) if data.get("child_ids") else None,
            merge_distance=data.get("merge_distance"),
            leaf_topic_ids=data.get("leaf_topic_ids", []),
        )

    def copy(self, new_id: int | None = None) -> "Topic":
        """Create a copy of this topic, optionally with a new ID."""
        copied = Topic.from_dict(self.to_dict(full=True))
        if new_id is not None:
            copied.id = new_id
        return copied

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
    def predictions(self) -> list[int]:
        """Get current predictions (mapped from original)."""
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

    def get(self, topic_id: int) -> Topic | None:
        """Get topic by ID, returning None if not found."""
        return self.topics.get(topic_id)

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
        return [int(self.mapping.map(prediction, from_original=from_original)) for prediction in predictions]

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
            rows = [selected_topic.to_info_dict()] if selected_topic else []
        else:
            rows = [t.to_info_dict() for t in self]

        columns = list(rows[0].keys())
        data = {col: [row.get(col) for row in rows] for col in columns}
        return pl.DataFrame(data)

    def to_dict(self, full: bool = False) -> dict:
        """Serialize Topics for storage.

        Arguments:
            full: If True, include embeddings and probabilities (for in-memory copy).
                  If False, exclude large arrays (for disk serialization).
        """
        data = {
            "bertopic_version": BERTOPIC_VERSION,
            "topics": {str(tid): topic.to_dict(full=full) for tid, topic in self.topics.items()},
            "mapping": self.mapping.to_dict(),
            "predictions": self._original_predictions.tolist() if self._original_predictions.size > 0 else [],
            "actions": [a.value for a in self.actions],
        }

        if full:
            if self._original_probabilities is not None:
                data["original_probabilities"] = self._original_probabilities.tolist()
            if self._zeroshot_probabilities is not None:
                data["zeroshot_probabilities"] = self._zeroshot_probabilities.tolist()

        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Topics":
        """Deserialize Topics from storage."""
        topics = cls()
        topics.topics = {int(tid): Topic.from_dict(td) for tid, td in data.get("topics", {}).items()}
        topics.mapping = TopicMapping.from_dict(data.get("mapping", {}))
        topics._original_predictions = np.array(data.get("predictions", []))
        topics.actions = [TopicAction(a) for a in data.get("actions", [])]

        # Handle full format fields
        if "original_probabilities" in data:
            topics._original_probabilities = np.array(data["original_probabilities"])
        if "zeroshot_probabilities" in data:
            topics._zeroshot_probabilities = np.array(data["zeroshot_probabilities"])

        return topics

    def copy(self) -> "Topics":
        """Create a deep copy of this Topics collection."""
        return Topics.from_dict(self.to_dict(full=True))

    def merge_similar(self, other: "Topics", min_similarity: float = 0.7) -> "Topics":
        """Merge another Topics collection based on embedding similarity.

        Topics from `other` are compared against topics in `self`. Those with
        cosine similarity >= min_similarity are deduplicated (their document
        predictions map to the existing similar topic). Dissimilar topics are
        added as new topics with new IDs.

        After merging:
        - New topics are added to self.topics
        - Predictions from other are remapped and appended to self
        - Document counts are recalculated
        - The mapping is reset to identity

        Arguments:
            other: Another Topics collection to merge into this one.
            min_similarity: Minimum cosine similarity to consider topics as duplicates.

        Returns:
            self (for method chaining)
        """
        from sklearn.metrics.pairwise import cosine_similarity
        from collections import Counter

        self_ids = self.topic_ids(outliers=False)
        other_ids = other.topic_ids(outliers=False)

        # Handle edge cases
        if not other_ids:
            return self

        # Build ID mapping: other_id -> self_id
        id_mapping = {-1: -1}

        if not self_ids:
            # Self has no real topics, just add all from other
            for other_id in other_ids:
                id_mapping[other_id] = other_id
                self.topics[other_id] = other.topics[other_id].copy()
        else:
            # Compute similarity
            self_emb = np.array([self.topics[tid].embedding for tid in self_ids])
            other_emb = np.array([other.topics[tid].embedding for tid in other_ids])
            sim_matrix = cosine_similarity(other_emb, self_emb)

            max_sims = np.max(sim_matrix, axis=1)
            best_matches = np.argmax(sim_matrix, axis=1)
            next_id = max(self_ids) + 1

            for i, other_id in enumerate(other_ids):
                if max_sims[i] >= min_similarity:
                    id_mapping[other_id] = self_ids[best_matches[i]]
                else:
                    id_mapping[other_id] = next_id
                    self.topics[next_id] = other.topics[other_id].copy(new_id=next_id)
                    next_id += 1

        # Ensure outlier exists
        if -1 in other.topics and -1 not in self.topics:
            self.topics[-1] = other.topics[-1].copy()

        # Merge predictions: get current, remap other's, concatenate
        current_preds = list(self.predictions)
        other_preds = [id_mapping[p] for p in other.predictions]
        all_preds = current_preds + other_preds

        # Store as new "original" with identity mapping
        self._original_predictions = np.array(all_preds)
        self.mapping.reset()

        # Recalculate document counts
        counts = Counter(all_preds)
        for topic in self.topics.values():
            topic.nr_documents = counts.get(topic.id, 0)

        self.add_action(TopicAction.MERGED)
        return self


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
        copied = topic.copy(new_id=new_id)
        # Clear hierarchy fields
        copied.parent_id = None
        copied.child_ids = None
        copied.merge_distance = None
        copied.leaf_topic_ids = []
        return copied

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

    def to_dict(self) -> dict:
        """Serialize hierarchy for storage."""
        return {
            "bertopic_version": BERTOPIC_VERSION,
            "nodes": {str(nid): node.to_dict() for nid, node in self.nodes.items()},
            "linkage_matrix": self.linkage_matrix.tolist() if self.linkage_matrix.size > 0 else [],
            "n_leaves": self.n_leaves,
            "outlier_topic": self.outlier_topic.to_dict() if self.outlier_topic else None,
            "predictions": self._original_predictions.tolist() if self._original_predictions.size > 0 else [],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TopicHierarchy":
        """Deserialize hierarchy from storage."""
        hierarchy = cls()
        hierarchy.nodes = {int(nid): Topic.from_dict(nd) for nid, nd in data.get("nodes", {}).items()}
        hierarchy.linkage_matrix = np.array(data.get("linkage_matrix", []))
        hierarchy.n_leaves = data.get("n_leaves", 0)
        hierarchy.outlier_topic = (
            Topic.from_dict(data["outlier_topic"]) if data.get("outlier_topic") else None
        )
        hierarchy._original_predictions = np.array(data.get("predictions", []))
        return hierarchy
