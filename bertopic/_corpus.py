from dataclasses import dataclass, field
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict

from bertopic._topics import Topics


@dataclass
class Corpus:
    """Temporary container used to track the input and generated data during fitting."""

    # Input data
    documents: list[str] | np.ndarray = field(default_factory=list)
    topics: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    embeddings: np.ndarray | None = None
    images: list[str] | None = None
    timestamps: list[str] | list[int] | np.ndarray | None = None
    classes: list[str] | list[int] | np.ndarray | None = None

    # Generated data
    umap_embeddings: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = None

    # For tracking ID mappings (e.g., during zero-shot)
    original_indices: np.ndarray = None

    # Zero-shot topic labels
    _zeroshot_labels: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.original_indices is None:
            self.original_indices = np.arange(len(self.documents))

        # For inference where a single document is passed as a string
        if isinstance(self.documents, str):
            self.documents = [self.documents]

        if isinstance(self.documents, np.ndarray):
            self.documents = self.documents.tolist()

        if isinstance(self.classes, list):
            self.classes = np.array(self.classes)

        if self.timestamps is not None:
            self.timestamps = np.array(self.timestamps, dtype="datetime64[ns]")

        check_documents_type(self.documents)
        check_embeddings_shape(self.embeddings, self.documents)

    @property
    def has_only_images(self) -> bool:
        """Check whether only images are provided."""
        return self.images is not None and self.documents is None

    @property
    def has_documents(self) -> bool:
        """Check whether documents are provided."""
        return self.documents is not None

    @property
    def has_outliers(self) -> bool:
        """Check whether outlier topic (-1) is present."""
        return -1 in self.topics

    @property
    def has_zeroshot_labels(self) -> bool:
        """Check whether zero-shot topic labels are provided."""
        return bool(self._zeroshot_labels)

    @property
    def _outliers(self) -> np.ndarray:
        """Returns a boolean array indicating outlier documents."""
        return 1 if self.has_outliers else 0

    def topic_ids(self, outliers: bool = True) -> list[int]:
        """Returns the unique topics in the data."""
        if self.topics is None:
            return []
        elif not outliers and self.has_outliers:
            topics = [int(topic) for topic in set(self.topics) if topic != -1]
            return sorted(topics)
        else:
            topics = [int(topic) for topic in set(self.topics)] if self.topics is not None else []
            return sorted(topics)

    def nr_topics(self, include_outliers: bool = True) -> int:
        """Returns the number of unique topics in the data."""
        if self.topics is None:
            return 0
        elif include_outliers:
            return len(set(self.topics))
        else:
            return len(set(self.topics)) - (1 if -1 in self.topics else 0)

    def group_documents_by_topic(self) -> dict[int, str]:
        """Groups documents by their assigned topic."""
        # Group documents by topic
        grouped = defaultdict(list)
        for doc, topic in zip(self.documents, self.topics):
            grouped[topic].append(doc)

        # Aggregate documents per topic into a single string
        aggregated = {topic: " ".join(docs) for topic, docs in grouped.items()}
        aggregated_sorted = dict(sorted(aggregated.items()))
        return aggregated_sorted

    def average_embeddings_by_topic(self) -> dict[int, np.ndarray]:
        """Averages embeddings by their assigned topic."""
        if self.embeddings is None:
            raise ValueError("Embeddings are not available to average by topic.")

        # Group embeddings by topic
        grouped = defaultdict(list)
        for embedding, topic in zip(self.embeddings, self.topics):
            grouped[topic].append(embedding)

        # Average embeddings per topic
        averaged = {topic: np.mean(embs, axis=0) for topic, embs in grouped.items()}
        averaged_sorted = dict(sorted(averaged.items()))
        return averaged_sorted

    def map_topics_and_probabilities(self, topics: Topics, from_original: bool = False) -> None:
        """Map both topics and probabilities to the reduced topics using the provided Topics object.

        Arguments:
            topics: A Topics object containing the mapping information.
            from_original: Whether to map from the original topics to the current ones.
        """
        self.map_topics(topics, from_original=from_original)

        # Only map probabilities if they are 2-dimensional since only then they
        # correspond to topic probabilities which might be reduced or reordered.
        if self.probabilities is not None:
            if len(self.probabilities.shape) > 1:
                self.map_probabilities(topics, from_original=from_original)

    def map_topics(self, topics: Topics, from_original: bool = False) -> None:
        """Map the topics to the reduced topics using the provided Topics object.

        Arguments:
            topics: A Topics object containing the mapping information.
            from_original: Whether to map from the original topics to the current ones.
        """
        self.topics = [
            topics.mapping.map(prediction, from_original=from_original) for prediction in self.topics
        ]

    def map_probabilities(self, topics: Topics, from_original: bool = False) -> None:
        """Map the (2-dimensional) probabilities to the reduced topics.

        There are two scenarios based on the mappings in the Topics object:
        * The order of topics has changed (e.g., after sorting by frequency).
          In this case, the probabilities are simply reordered.
        * Some topics have been merged. In this case, the probabilities
          of the merged topics are summed together and assigned to the new topic.

        Note that the outlier topic (-1), if present, is skipped during this process.
        If it is present, it is always at the zero-th index of the initial probabilities matrix
        and should remain so after mapping.

        Arguments:
            topics: A Topics object containing the mapping information.
            from_original: If True, mappings are obtained from the original topics.
                           If False, mappings are obtained from the most recent topics.
        """
        # Check scenario based on mappings
        mappings = topics.get_mappings(from_original=from_original)

        # Scenario 1: Reordering
        if len(set(mappings.values())) == len(mappings):
            nr_topics = len(set(mappings.values()))
            new_order = [0] * nr_topics
            for old_topic, new_topic in mappings.items():
                if old_topic == -1:
                    continue  # Skip outlier topic
                new_topic_idx = new_topic + self._outliers
                old_topic_idx = old_topic + self._outliers
                new_order[new_topic_idx] = old_topic_idx

            self.probabilities = self.probabilities[:, new_order]

        # Scenario 2: Merging
        else:
            nr_new_topics = len(set(mappings.values()))
            new_probabilities = np.zeros((self.probabilities.shape[0], nr_new_topics))

            for old_topic, new_topic in mappings.items():
                if old_topic == -1:
                    continue  # Skip outlier topic
                new_topic_idx = new_topic + self._outliers
                old_topic_idx = old_topic + self._outliers
                new_probabilities[:, new_topic_idx] += self.probabilities[:, old_topic_idx]

            self.probabilities = new_probabilities

    def sort_topics_by_frequency(self) -> "Corpus":
        """Maps the label of each topic to its frequency rank with
        the outlier topic (-1) always being the -1 topic.
        """
        unique, counts = np.unique(self.topics, return_counts=True)
        topic_freq = dict(zip(unique, counts))

        # Sort topics by frequency, excluding outlier topic (-1)
        sorted_topics = sorted(
            [t for t in topic_freq.keys() if t != -1],
            key=lambda x: topic_freq[x],
            reverse=True,
        )

        # Create mapping from old topic to new topic
        topic_mapping = {old_topic: new_topic for new_topic, old_topic in enumerate(sorted_topics)}
        topic_mapping[-1] = -1  # Keep outlier topic as -1

        # Remap topics
        self.topics = np.array([topic_mapping[t] for t in self.topics])

        # Reorder zero-shot labels if they exist
        # The zero-shot labels always start at index 0 and go up to the number of zero-shot topics - 1
        # The resulting zero-labels should therefore be the same size as the original zero-shot labels
        # Thus, we can create a new list of zero-shot labels based on the new topic mapping
        if self._zeroshot_labels:
            nr_zeroshot = len(self._zeroshot_labels)
            new_zeroshot_labels = [""] * nr_zeroshot
            for old_topic, new_topic in topic_mapping.items():
                if 0 <= old_topic < nr_zeroshot:
                    new_zeroshot_labels[new_topic] = self._zeroshot_labels[old_topic]
            self._zeroshot_labels = new_zeroshot_labels

        return self

    def sort_by_timestamps(self) -> "Corpus":
        """Sorts the corpus by timestamps in ascending order."""
        if self.timestamps is None:
            raise ValueError("Timestamps are not available to sort the corpus.")

        sort_order = np.argsort(self.timestamps)

        self.documents = [self.documents[i] for i in sort_order]
        self.topics = np.array(self.topics)[sort_order] if self.topics is not None else None
        self.probabilities = self.probabilities[sort_order] if self.probabilities is not None else None
        self.embeddings = self.embeddings[sort_order] if self.embeddings is not None else None
        self.images = list(np.array(self.images)[sort_order]) if self.images is not None else None
        self.timestamps = self.timestamps[sort_order]
        self.classes = np.array(self.classes)[sort_order] if self.classes is not None else None
        self.umap_embeddings = (
            self.umap_embeddings[sort_order] if self.umap_embeddings.size > 0 else self.umap_embeddings
        )
        self.y = np.array(self.y)[sort_order] if self.y is not None else None
        self.original_indices = (
            np.array(self.original_indices)[sort_order] if self.original_indices is not None else None
        )

        return self

    def get_documents_by_indices(self, indices: list[int]) -> list[str]:
        """Returns documents corresponding to the provided indices."""
        return [self.documents[index] for index in indices]

    def get_corpus_by_indices(self, indices: list[int]) -> "Corpus":
        """Returns a Corpus object corresponding to the provided indices."""
        sorted_indices = sorted(indices)
        selected_documents = [self.documents[index] for index in sorted_indices]
        selected_topics = (
            [self.topics[index] for index in sorted_indices] if self.topics is not None else None
        )
        selected_embeddings = self.embeddings[sorted_indices] if self.embeddings is not None else None
        selected_images = (
            [self.images[index] for index in sorted_indices] if self.images is not None else None
        )
        selected_original_indices = [self.original_indices[index] for index in sorted_indices]
        return Corpus(
            documents=selected_documents,
            topics=selected_topics,
            embeddings=selected_embeddings,
            images=selected_images,
            original_indices=selected_original_indices,
            _zeroshot_labels=self._zeroshot_labels,
        )

    def get_topic(self, topic_id: int, nr_samples: int | None = None) -> "Corpus":
        """Return a Corpus object containing only documents of the specified topic.

        Arguments:
            topic_id: The topic ID to filter by.
            nr_samples: The number of documents to randomly sample for the topic.
        """
        # Filter documents by topic
        filtered_docs = [doc for doc, topic in zip(self.documents, self.topics) if topic == topic_id]
        filtered_indices = [i for i, topic in enumerate(self.topics) if topic == topic_id]

        # Sample documents if nr_samples is specified
        if nr_samples is not None and len(filtered_docs) > nr_samples:
            sampled_indices = np.random.choice(len(filtered_docs), size=nr_samples, replace=False)
            filtered_docs = [filtered_docs[i] for i in sampled_indices]
            filtered_indices = [filtered_indices[i] for i in sampled_indices]

        # Filter embeddings if they exist
        if self.embeddings is not None:
            filtered_embeddings = self.embeddings[filtered_indices]
        else:
            filtered_embeddings = None

        return Corpus(
            documents=filtered_docs,
            topics=[topic_id] * len(filtered_docs),
            embeddings=filtered_embeddings,
            original_indices=filtered_indices,
        )

    def __add__(self, other: "Corpus") -> "Corpus":
        """Combine two Documents objects by concatenating their attributes
        based on the original indices.
        """
        # Combine indices from both datasets
        combined_indices = self.original_indices + other.original_indices

        # Get the sorting order to restore original order
        sort_order = np.argsort(combined_indices)

        # Combine and reorder documents
        combined_documents = self.documents + other.documents
        sorted_documents = [combined_documents[i] for i in sort_order]

        # Combine and reorder topics
        if other.has_zeroshot_labels:
            # The "other" Data contains zero-shot topics which
            # should be be after any outlier topics in "self"
            # but before the regular topics.
            nr_zeroshot = len(set(other.topics))
            self.topics = np.array([topic + nr_zeroshot if topic != -1 else -1 for topic in self.topics])

        # Combine topics (two flat numpy arrays) and sort them
        combined_topics = np.concatenate([self.topics, other.topics])
        sorted_topics = [combined_topics[i] for i in sort_order]

        # Combine and reorder embeddings (numpy array)
        combined_embeddings = np.vstack([self.embeddings, other.embeddings])
        sorted_embeddings = combined_embeddings[sort_order]

        # Combine and reorder original_indices
        sorted_indices = [combined_indices[i] for i in sort_order]

        return Corpus(
            documents=sorted_documents,
            topics=sorted_topics,
            embeddings=sorted_embeddings,
            original_indices=sorted_indices,
            _zeroshot_labels=other._zeroshot_labels if other.has_zeroshot_labels else None,
        )

    def __len__(self):
        return len(self.documents)

    def _validate_length(self, name: str, value) -> None:
        """Checks that the length of the value matches the number of documents."""
        if value is not None and len(value) != len(self.documents):
            raise ValueError(
                f"Length of {name} ({len(value)}) does not match number of documents ({len(self.documents)})"
            )

    def __setattr__(self, name: str, value) -> None:
        """Whenever we update embeddings, images, or topics, validate their length."""
        if name in ("embeddings", "images", "topics") and hasattr(self, "documents"):
            self._validate_length(name, value)
        super().__setattr__(name, value)


def check_documents_type(documents):
    """Check whether the input documents are indeed a list of strings."""
    if not isinstance(documents, list):
        raise TypeError("The input documents needs to be a flat list of strings.")
    else:
        if len(documents) > 0 and not isinstance(documents[0], str):
            raise TypeError("Make sure that each document is a valid string.")


def check_embeddings_shape(embeddings, docs):
    """Check if the embeddings have the correct shape."""
    if embeddings is not None:
        if not any([isinstance(embeddings, np.ndarray), isinstance(embeddings, csr_matrix)]):
            raise ValueError(
                "Make sure to input embeddings as a numpy array or scipy.sparse.csr.csr_matrix. "
            )
        else:
            if embeddings.shape[0] != len(docs):
                raise ValueError(
                    "Make sure that the embeddings are a numpy array with shape: "
                    "(len(docs), vector_dim) where vector_dim is the dimensionality "
                    "of the vector embeddings. "
                )
