from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from importlib.metadata import version

import narwhals.stable.v2 as nw
from narwhals.stable.v2.typing import IntoDataFrame

from bertopic._config import get_output
from bertopic._corpus import Modality
from scipy.sparse import csr_matrix, vstack
from scipy.cluster.hierarchy import fcluster
import numpy as np
from collections import Counter, defaultdict

BERTOPIC_VERSION = version("bertopic")


@dataclass
class TopicRepresentation:
    """Base class for all topic representations."""

    data: Any = None

    @property
    def words(self) -> list[str]:
        """The text view of this representation, which is empty when it has none.

        `data` is a payload of any modality, so a representation made of images or
        audio legitimately has no words until something converts it to text.
        """
        return []

    @property
    def scores(self) -> list[float]:
        """One score per word, weighting every word equally.

        Only `Keywords` ranks the words within a topic. Everything else names the topic
        instead of weighting what is in it, so no word is more representative than another.
        """
        return [1.0] * len(self.words)

    @property
    def word_scores(self) -> list[tuple[str, float]]:
        """Words paired with their scores, which is how `get_topic` returns a topic."""
        return list(zip(self.words, self.scores))

    @property
    def label(self) -> str:
        """A short name for the topic, which only some representations can give."""
        return "No label could be created"

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

    @property
    def label(self) -> str:
        """The top four keywords joined, which is how a topic is named by default."""
        return "_".join(self.words[:4])

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

    @property
    def words(self) -> list[str]:
        """Return the label as a single-element list."""
        return [self.data] if self.data else []

    @property
    def label(self) -> str:
        """The label itself."""
        return self.data

    def to_dict(self) -> dict:
        return {"type": "label", "data": self.data}

    @classmethod
    def from_dict(cls, data: dict) -> "Label":
        return cls(data=data["data"])


@dataclass
class StructuredJSON(TopicRepresentation):
    """A structured JSON representation for a topic."""

    data: dict[str, str] = field(default_factory=dict)

    @property
    def words(self) -> list[str]:
        """Return the values of the structured JSON as a list of strings."""
        return [str(v) for v in self.data.values()] if self.data else []

    @property
    def label(self) -> str:
        """The `topic_label` or `label` field, or the whole JSON when it has neither."""
        return self.data.get("topic_label", self.data.get("label", str(self.data)))

    def to_dict(self) -> dict:
        return {"type": "structured_json", "data": self.data}

    @classmethod
    def from_dict(cls, data: dict) -> "StructuredJSON":
        return cls(data=data["data"])


@dataclass
class Media(TopicRepresentation):
    """Representative media for a topic, described once a converter has run.

    A topic may hold several modalities at once, so `items` is keyed by modality rather
    than being one list per kind: a topic of photographs and voice notes is one
    representation carrying both. `summaries` holds one object per modality that stands for
    the whole topic, the way a label does for text: the images tiled into a collage, the
    middle frame of each video tiled into a frame sheet, and the opening of each clip joined
    into a montage.

    Captions are serialized, and so are items given as paths. Media held in memory is
    regenerated by re-running the representation model, and summaries are written beside
    the model as files, so persisting either here would bloat the model on disk.
    """

    items: dict[Modality, list] = field(default_factory=dict)
    summaries: dict[Modality, Any] = field(default_factory=dict)
    captions: list[str] = field(default_factory=list)

    @property
    def words(self) -> list[str]:
        """The captions, which stay empty until a converter has produced them."""
        return self.captions

    @property
    def images(self) -> list:
        """The representative images, for topics that have any."""
        return self.items.get(Modality.IMAGE, [])

    @property
    def collage(self) -> Any:
        """The representative images tiled into one picture, for topics that have any."""
        return self.summaries.get(Modality.IMAGE)

    def __str__(self) -> str:
        """String representation of the captions, or the media count."""
        return ", ".join(self.captions) if self.captions else "No captions available"

    def to_dict(self) -> dict:
        # Paths are worth writing down, since the media they point to is already on disk
        paths = {
            modality.value: items
            for modality, items in self.items.items()
            if all(isinstance(item, str) for item in items)
        }
        return {"type": "media", "captions": self.captions, "items": paths}

    @classmethod
    def from_dict(cls, data: dict) -> "Media":
        items = {Modality(modality): paths for modality, paths in data.get("items", {}).items()}
        return cls(items=items, captions=data.get("captions", []))


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
        "media": Media,
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
    """Tracks how a cluster model's original labels map to the current topic IDs.

    Only `transform` needs this. A cluster model hands back the labels it invented while
    fitting, which have to be translated into whatever the topics were renumbered to
    since. Rows that were part of the fit keep their assignments in `Topics.predictions`
    instead, so nothing here describes the fitted corpus.
    """

    _mapping: dict[int, int] = field(default_factory=dict)

    def apply(self, old_to_new: dict[int, int]) -> None:
        """Compose a renumbering, keeping the mapping original -> current."""
        if not self._mapping:
            self._mapping = old_to_new.copy()
            return

        # Outliers remain as -1 and do not need to be included in the mapping
        missing = sorted(set(self._mapping.values()) - set(old_to_new) - {-1})
        if missing:
            raise ValueError(
                f"The mapping is missing topics {missing}. Every current topic must appear "
                "in a new mapping, including topics that hold no documents."
            )
        self._mapping = {
            original: old_to_new.get(current, current) for original, current in self._mapping.items()
        }

    def map(self, topic_id: int) -> int:
        """Translate one original cluster label into its current topic ID."""
        return self._mapping.get(topic_id, topic_id)

    def map_predictions(self, predictions: list[int] | np.ndarray) -> list[int]:
        """Translate a cluster model's labels into current topic IDs."""
        return [self.map(int(prediction)) for prediction in predictions]

    def reset(self) -> None:
        """Clear the mapping, so current IDs become the originals."""
        self._mapping.clear()

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {"mapping": {str(original): current for original, current in self._mapping.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> "TopicMapping":
        """Deserialize from dictionary."""
        mapping = cls()
        mapping._mapping = {int(original): current for original, current in data.get("mapping", {}).items()}
        return mapping


@dataclass
class Topic:
    """A topic with multiple representations from different sources.
    Sources are strings identifying the method used (e.g., 'c-tf-idf', 'gemma3').

    Attributes:
        id: Unique identifier for the topic.
        representations: A dictionary mapping source names to TopicRepresentation instances.
        representative_documents: A list of representative documents for the topic.
        custom_label: A label the user set with `set_topic_labels`, shown where custom labels are asked for.
        zeroshot_label: The label a zero-shot topic was given in `zeroshot_topic_list`.
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
    custom_label: str | None = None
    zeroshot_label: str | None = None

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
        """The topic's zero-shot label, or else the label its Main representation gives."""
        if self.zeroshot_label is not None:
            return self.zeroshot_label
        representation = self.representations.get("Main")
        return representation.label if representation else "No label could be created"

    @property
    def name(self) -> str:
        """How the topic is named in `get_topic_info` and `topic_labels_`.

        The topic's ID followed by its label, which is what makes two topics with similar
        keywords tellable apart. A zero-shot topic keeps its zero-shot label as it is.
        """
        return self.zeroshot_label if self.zeroshot_label is not None else f"{self.id}_{self.label}"

    @property
    def media(self) -> Media | None:
        """The representation holding this topic's media, under whichever aspect it was given."""
        return next((rep for rep in self.representations.values() if isinstance(rep, Media)), None)

    @property
    def representative_items(self) -> list:
        """The media that represent this topic, one modality after another."""
        if self.media is None:
            return []
        return [item for items in self.media.items.values() for item in items]

    def __getitem__(self, source: str) -> TopicRepresentation:
        """Get representation for a specific source, or default if not found."""
        return self.representations.get(source, TopicRepresentation())

    def __setitem__(self, source: str, rep: TopicRepresentation):
        """Set representation for a specific source."""
        self.representations[source] = rep

    def to_info_dict(self, custom_labels: bool = False) -> dict:
        """Serialize topic info to a flat dictionary for tabular output.

        With `custom_labels`, a `CustomName` column follows `Name`.

        Media is shown as the items themselves, under `Representative_Items` rather than
        its aspect's name, so the columns stay the same whatever modalities a topic holds.
        """
        info = {"Topic": self.id, "Count": self.nr_documents, "Name": self.name}
        if custom_labels:
            info["CustomName"] = self.custom_label or self.name
        info["Representation"] = self["Main"].words

        # Extract all other representations
        for name, rep in self.representations.items():
            if name != "Main" and not isinstance(rep, Media):
                info[name] = rep.words

        info["Representative_Docs"] = self.representative_documents or [""]
        if self.media is not None:
            info["Representative_Items"] = self.representative_items

        return info

    def to_dict(self, full: bool = False) -> dict:
        """Serialize topic for storage.

        Arguments:
            full: If True, include embeddings and c_tf_idf (for in-memory copy).
                  If False, exclude large arrays (for disk serialization).
        """
        data = {
            "id": self.id,
            "custom_label": self.custom_label,
            "zeroshot_label": self.zeroshot_label,
            "nr_documents": self.nr_documents,
            "topic_type": self.topic_type.value,
            "representations": {name: rep.to_dict() for name, rep in self.representations.items()},
            "representative_documents": self.representative_documents,
        }

        if full:
            data["embedding"] = self.embedding.tolist() if self.embedding.size else []
            data["c_tf_idf"] = {
                "data": self.c_tf_idf.data.tolist(),
                "indices": self.c_tf_idf.indices.tolist(),
                "indptr": self.c_tf_idf.indptr.tolist(),
                "shape": list(self.c_tf_idf.shape),
            }

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

        return cls(
            id=data["id"],
            custom_label=data.get("custom_label"),
            zeroshot_label=data.get("zeroshot_label"),
            nr_documents=data.get("nr_documents", 0),
            topic_type=TopicType(data.get("topic_type", "normal")),
            representations=representations,
            representative_documents=data.get("representative_documents", []),
            embedding=embedding,
            c_tf_idf=c_tf_idf,
            parent_id=data.get("parent_id"),
            child_ids=tuple(data["child_ids"]) if data.get("child_ids") else None,
            merge_distance=data.get("merge_distance"),
            leaf_topic_ids=data.get("leaf_topic_ids", []),
        )

    def copy(self, new_id: int | None = None) -> "Topic":
        """Create a copy of this topic, optionally with a new ID."""
        copied = deepcopy(self)
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

    # One assignment per row, always in current topic IDs. Rows are whatever was fitted,
    # which is not necessarily the user's documents.
    predictions: list[int] = field(default_factory=list)
    probabilities: np.ndarray | None = None

    # History of actions applied to this collection
    actions: list[TopicAction] = field(default_factory=list)

    def frequencies(self) -> dict[int, int]:
        """Get the number of documents for each topic."""
        return {topic.id: topic.nr_documents for topic in self.topics.values()}

    @property
    def names(self) -> dict[int, str]:
        """The name of every topic, by topic ID."""
        return {topic.id: topic.name for topic in self}

    @property
    def custom_labels(self) -> list[str] | None:
        """The labels a user set, with every other topic's name, or None while no topic has one."""
        if all(topic.custom_label is None for topic in self):
            return None
        return [topic.custom_label or topic.name for topic in self]

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

    def check(self) -> None:
        """Raise a ValueError naming every rule these topics break.

        These are the rules every operation must leave intact. Topic IDs may have gaps and a
        topic may hold no documents, since custom topics and `partial_fit` produce both.
        """
        topic_ids = self.topic_ids()
        counts = Counter(self.predictions)
        mapped = set(self.get_mappings().values())
        rules = {
            "topic IDs are plain ints": all(type(topic_id) is int for topic_id in [*topic_ids, *counts]),
            "every prediction is a known topic": set(counts) <= set(topic_ids),
            "topic sizes match the predictions": all(
                topic.nr_documents == counts[topic.id] for topic in self
            ),
            "only topic -1 is the outlier": all(
                (topic.id == -1) == (topic.topic_type == TopicType.OUTLIER) for topic in self
            ),
            "the cluster mapping points at known topics": mapped <= {*topic_ids, -1},
            "c-TF-IDF rows share one width": len({topic.c_tf_idf.shape[1] for topic in self}) <= 1,
            "embeddings share one shape": len({topic.embedding.shape for topic in self}) <= 1,
        }
        if self.probabilities is not None:
            rules["one probability row per prediction"] = len(self.probabilities) == len(self.predictions)
            if self.probabilities.ndim == 2:
                rules["one probability column per topic"] = self.probabilities.shape[1] == len(topic_ids)

        broken = [rule for rule, holds in rules.items() if not holds]
        if broken:
            raise ValueError(f"The topics break {len(broken)} rule(s): {'; '.join(broken)}.")

    def initialize(
        self,
        predictions: list[int] | np.ndarray,
        zeroshot_labels: list[str] | None = None,
        probabilities: np.ndarray | None = None,
    ):
        """Initialize topics from a cluster model's labels; the first topics take the zero-shot labels."""
        self.assign(predictions, probabilities)
        for topic in self:
            if zeroshot_labels and topic.id in range(len(zeroshot_labels)):
                topic.topic_type, topic.zeroshot_label = TopicType.ZERO_SHOT, zeroshot_labels[topic.id]

        self.add_action(TopicAction.INITIALIZED)
        return self

    def assign(self, predictions: list[int] | np.ndarray, probabilities: np.ndarray | None = None) -> None:
        """Point every row at a topic, adding a topic for each new ID and removing those left without rows.

        The clusters of a removed topic count as outliers from then on.

        Arguments:
            predictions: The topic ID of every row.
            probabilities: The probabilities of every row, which replace those stored, also when None.
        """
        predictions = [int(prediction) for prediction in predictions]
        sizes = Counter(predictions)
        self.mapping.apply({topic_id: topic_id if topic_id in sizes else -1 for topic_id in self.topics})

        for topic_id in self.topics.keys() - sizes.keys():
            del self.topics[topic_id]
        for topic_id in sizes.keys() - self.topics.keys():
            self.topics[topic_id] = Topic(id=topic_id)
        for topic in self:
            topic.nr_documents = sizes[topic.id]

        self.predictions, self.probabilities = predictions, probabilities

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

    def _apply_to_rows(self, old_to_new: dict[int, int], column_order: list[int]) -> None:
        """Move every row's assignment and probability column to the new topic IDs.

        Columns are summed rather than overwritten, so this covers merging and deleting
        as well as renumbering: topics that collapse into one add their mass together,
        and deleted topics map to -1 and so add theirs to the outlier.
        """
        self.predictions = [old_to_new[prediction] for prediction in self.predictions]

        if self.probabilities is None or self.probabilities.ndim != 2:
            return

        new_ids = sorted(set(old_to_new.values()))
        remapped = np.zeros((self.probabilities.shape[0], len(new_ids)))
        for position, old_id in enumerate(column_order):
            remapped[:, new_ids.index(old_to_new[old_id])] += self.probabilities[:, position]
        self.probabilities = remapped

    def remap(self, old_to_new: dict[int, int]) -> None:
        """Move topics to new IDs, and the rows, probability columns and cluster mapping with them.

        Renumbering, sorting, merging and deleting all come down to this:

            * A topic that moves alone keeps everything it holds, its labels included
            * The outlier topic keeps its own data while it stays -1, and absorbs the documents moved onto it
            * Topics moved onto the same ID are merged into one, as `merge` describes, and so is a topic
              that becomes the outlier or stops being it

        Arguments:
            old_to_new: Every current topic ID, mapped to its new ID.
        """
        column_order = sorted(self.topics)
        new_to_old: dict[int, list[Topic]] = defaultdict(list)
        for old_id, new_id in old_to_new.items():
            new_to_old[new_id].append(self.topics[old_id])

        self.topics = {new_id: self._combine(new_id, old_topics) for new_id, old_topics in new_to_old.items()}
        self._apply_to_rows(old_to_new, column_order)
        self.mapping.apply(old_to_new)

    @staticmethod
    def _combine(new_id: int, old_topics: list[Topic]) -> Topic:
        """The topic that `old_topics` become once they are all moved onto `new_id`."""
        nr_documents = sum(topic.nr_documents for topic in old_topics)
        outliers = [topic for topic in old_topics if topic.id == -1]

        # The outlier topic keeps what it holds while it stays the outlier, and so does a topic that moves
        # alone, unless it becomes the outlier or stops being it
        if (outliers and new_id == -1) or (len(old_topics) == 1 and not outliers and new_id != -1):
            kept = outliers[0] if outliers else old_topics[0]
            kept.id, kept.nr_documents = new_id, nr_documents
            return kept

        # Weight by document count, falling back to equal weights when no topic holds documents
        if nr_documents:
            weights = np.array([topic.nr_documents / nr_documents for topic in old_topics])
        else:
            weights = np.full(len(old_topics), 1 / len(old_topics))

        # Average the embeddings only when every topic has one, since they may never have been computed
        embeddings = [topic.embedding for topic in old_topics if topic.embedding.size > 0]
        if len(embeddings) == len(old_topics):
            embedding = np.average(embeddings, weights=weights, axis=0)
        else:
            embedding = np.array([])

        # The result is zero-shot only if all of its parts are, and the outlier topic never is
        zeroshot = new_id != -1 and all(topic.topic_type == TopicType.ZERO_SHOT for topic in old_topics)
        prominent = max(old_topics, key=lambda topic: topic.nr_documents)
        return Topic(
            id=new_id,
            representations=prominent.representations.copy(),
            representative_documents=[doc for topic in old_topics for doc in topic.representative_documents],
            zeroshot_label=prominent.zeroshot_label if zeroshot else None,
            embedding=embedding,
            c_tf_idf=sum(topic.c_tf_idf * weight for topic, weight in zip(old_topics, weights)),
            topic_type=TopicType.ZERO_SHOT if zeroshot else TopicType.NORMAL,
            nr_documents=nr_documents,
        )

    def merge(self, old_to_new: dict[int, int]) -> None:
        """Merge the topics mapped onto the same ID.

        This will do the following:

            * Combine topic embeddings by weighted average based on nr_documents
            * Combine c-TF-IDF vectors by weighted average based on nr_documents
            * Sum nr_documents for merged topics
            * Keep the representations of the most prominent topic (highest nr_documents)
            * Combine representative documents from all merged topics
            * Keep a zero-shot label only if every merged topic is zero-shot
            * Leave a topic that is not merged as it is

        Arguments:
            old_to_new: Every current topic ID, mapped to its new ID.
        """
        self.remap(old_to_new)
        self.add_action(TopicAction.MERGED)

    def delete(self, topics: list[int] | int) -> None:
        """Delete topics by moving their documents to the outlier topic (-1), which is created if needed.

        Arguments:
            topics: List of topic IDs to delete or a single topic ID.
        """
        topics = {topics} if isinstance(topics, int) else set(topics)
        self.remap({topic_id: -1 if topic_id in topics else topic_id for topic_id in self.topics})
        self.add_action(TopicAction.DELETED)

    def map_predictions(self, predictions: list[int] | np.ndarray) -> list[int]:
        """Translate a cluster model's labels into current topic IDs, for new rows."""
        return self.mapping.map_predictions(predictions)

    def align_probabilities(self, probabilities: np.ndarray | None) -> np.ndarray | None:
        """Turn a cluster model's raw membership matrix into one column per current topic.

        Cluster models emit a column per cluster in their own label order and none for
        outliers, whose share is therefore whatever is left over. Columns are summed so
        that clusters which have since been merged land on the same topic.
        """
        if probabilities is None or probabilities.ndim != 2:
            return probabilities

        topic_ids = self.topic_ids()
        aligned = np.zeros((probabilities.shape[0], len(topic_ids)))
        for label in range(probabilities.shape[1]):
            topic_id = self.mapping.map(label)
            if topic_id in topic_ids:
                aligned[:, topic_ids.index(topic_id)] += probabilities[:, label]

        if -1 in topic_ids:
            aligned[:, 0] = 1 - aligned.sum(axis=1)
        return aligned

    def get_mappings(self) -> dict[int, int]:
        """Get the mapping from the cluster model's original labels to current topic IDs."""
        return self.mapping._mapping.copy()

    def to_dataframe(self, topics: int | list[int] | None = None) -> IntoDataFrame:
        """Convert topic info to a dataframe, a row per topic or for each of `topics`, repeats included."""
        # Each topic's row is built once, since a topic can be asked for once per document
        labelled = self.custom_labels is not None
        info = {topic.id: topic.to_info_dict(custom_labels=labelled) for topic in self}
        if topics is None:
            topics = list(info)
        elif not isinstance(topics, list):
            topics = [topics]
        rows = [info[topic] for topic in topics if topic in info]

        if not rows:
            return nw.from_dict({}, backend=get_output()).to_native()

        # Columns come from every topic, since the first may lack an aspect the others have
        columns = dict.fromkeys(column for row in rows for column in row)
        data = {column: [row.get(column) for row in rows] for column in columns}

        # Media is kept as the objects it is, since polars has no type for a list of images
        items = data.pop("Representative_Items", None)
        frame = nw.from_dict(data, backend=get_output())
        if items is not None:
            frame = frame.with_columns(
                nw.new_series("Representative_Items", items, dtype=nw.Object(), backend=get_output())
            )
        return frame.to_native()

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
            "predictions": list(self.predictions),
            "actions": [a.value for a in self.actions],
        }

        if full and self.probabilities is not None:
            data["probabilities"] = self.probabilities.tolist()

        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Topics":
        """Deserialize Topics from storage."""
        topics = cls()
        topics.topics = {int(tid): Topic.from_dict(td) for tid, td in data.get("topics", {}).items()}
        topics.mapping = TopicMapping.from_dict(data.get("mapping", {}))
        topics.predictions = list(data.get("predictions", []))
        topics.actions = [TopicAction(a) for a in data.get("actions", [])]

        if "probabilities" in data:
            topics.probabilities = np.array(data["probabilities"])

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
        - Document counts are recalculated, and probabilities dropped
        - The mapping is reset to identity

        Arguments:
            other: Another Topics collection to merge into this one.
            min_similarity: Minimum cosine similarity to consider topics as duplicates.

        Returns:
            self (for method chaining)
        """
        from sklearn.metrics.pairwise import cosine_similarity

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

        # Add the other's rows, whose probabilities describe neither collection's topics
        self.assign(self.predictions + [id_mapping[prediction] for prediction in other.predictions])

        # These rows are now the reference point, so the cluster mapping starts over
        self.mapping.reset()

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
        predictions: Row-to-leaf-topic assignments from the fit this hierarchy was built on.
        probabilities: Row-to-leaf-topic probabilities from that same fit.
    """

    # Hierarchy structure
    nodes: dict[int, Topic] = field(default_factory=dict)
    linkage_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    n_leaves: int = 0

    # The outlier topic is not part of the hierarchy
    outlier_topic: Topic | None = None

    # Original values from fitting
    predictions: list[int] = field(default_factory=list)
    probabilities: np.ndarray | None = None

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

        # Create Topics, moving the rows onto the topic IDs at this level of the tree
        topics = Topics()
        topics.predictions = list(self.predictions)
        topics.probabilities = self.probabilities.copy() if self.probabilities is not None else None
        topics._apply_to_rows(old_to_new, sorted(old_to_new))
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

    def to_dataframe(self) -> IntoDataFrame:
        """Convert hierarchy to a dataframe."""
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

        data = {column: [row.get(column) for row in rows] for column in rows[0]}
        return nw.from_dict(data, backend=get_output()).to_native()

    def to_dict(self) -> dict:
        """Serialize hierarchy for storage."""
        return {
            "bertopic_version": BERTOPIC_VERSION,
            "nodes": {str(nid): node.to_dict(full=True) for nid, node in self.nodes.items()},
            "linkage_matrix": self.linkage_matrix.tolist() if self.linkage_matrix.size > 0 else [],
            "n_leaves": self.n_leaves,
            "outlier_topic": self.outlier_topic.to_dict(full=True) if self.outlier_topic else None,
            "predictions": list(self.predictions),
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
        hierarchy.predictions = list(data.get("predictions", []))
        return hierarchy
