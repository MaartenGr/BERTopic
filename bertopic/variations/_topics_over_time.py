"""Topics over time (dynamic topic modeling) variation for BERTopic.
See: https://maartengr.github.io/BERTopic/getting_started/topicsovertime/topicsovertime.html.
"""

import numpy as np
from typing import TYPE_CHECKING
from sklearn.preprocessing import normalize
from tqdm import tqdm

from bertopic._utils import check_is_fitted, MyLogger
from bertopic._corpus import Corpus
from bertopic._topics import Topics

if TYPE_CHECKING:
    from bertopic import BERTopic

logger = MyLogger()
logger.configure("WARNING")


def topics_over_time(
    topic_model: "BERTopic",
    docs: list[str],
    timestamps: list[str] | list[int],
    topics: list[int] | None = None,
    nr_bins: int | None = None,
    evolution_tuning: bool = True,
    global_tuning: bool = True,
) -> dict[str, Topics]:
    """Create topics over time.

    To create the topics over time, BERTopic needs to be already fitted once.
    From the fitted models, the c-TF-IDF representations are calculate at
    each timestamp t. Then, the c-TF-IDF representations at timestamp t are
    averaged with the global c-TF-IDF representations in order to fine-tune the
    local representations.

    Note:
        Make sure to use a limited number of unique timestamps (<100) as the
        c-TF-IDF representation will be calculated at each single unique timestamp.
        Having a large number of unique timestamps can take some time to be calculated.
        Moreover, there aren't many use-cases where you would like to see the difference
        in topic representations over more than 100 different timestamps.

    Arguments:
        topic_model: The BERTopic instance
        docs: The documents you used when calling either `fit` or `fit_transform`
        timestamps: The timestamp of each document. This can be either a list of strings or ints.
                    If it is a list of strings, then the datetime format will be automatically
                    inferred. If it is a list of ints, then the documents will be ordered in
                    ascending order.
        topics: A list of topics where each topic is related to a document in `docs` and
                a timestamp in `timestamps`. You can use this to apply topics_over_time on
                a subset of the data. Make sure that `docs`, `timestamps`, and `topics`
                all correspond to one another and have the same size.
        nr_bins: The number of bins you want to create for the timestamps. The left interval will
                    be chosen as the timestamp. An additional column will be created with the
                    entire interval.
        evolution_tuning: Fine-tune each topic representation at timestamp *t* by averaging its
                            c-TF-IDF matrix with the c-TF-IDF matrix at timestamp *t-1*. This creates
                            evolutionary topic representations.
        global_tuning: Fine-tune each topic representation at timestamp *t* by averaging its c-TF-IDF matrix
                    with the global c-TF-IDF matrix. Turn this off if you want to prevent words in
                    topic representations that could not be found in the documents at timestamp *t*.

    Returns:
        topics_over_time: A dictionary where each key is a timestamp and each value
                          is a Topics object containing the topics at that timestamp.

    Examples:
    The timestamps variable represents the timestamp of each document. If you have over
    100 unique timestamps, it is advised to bin the timestamps as shown below:

    ```python
    from bertopic import BERTopic
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(docs)
    topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=20)
    ```
    """
    check_is_fitted(topic_model)

    corpus = Corpus(
        documents=docs, topics=topics if topics else topic_model.topics_.predictions, timestamps=timestamps
    )
    selected_topics = topics if topics else topic_model.topics_.predictions
    global_c_tf_idf = normalize(topic_model.c_tf_idf_, axis=1, norm="l1", copy=False)
    corpus.timestamps = bin_timestamps(corpus, nr_bins=nr_bins) if nr_bins else corpus.timestamps
    corpus.sort_by_timestamps()

    if len(set(corpus.timestamps)) > 100:
        logger.warning(
            f"There are more than 100 unique timestamps (i.e., {len(set(corpus.timestamps))}) "
            "which significantly slows down the application. Consider setting `nr_bins` "
            "to a value lower than 100 to speed up calculation. "
        )

    topics = {}
    for index, timestamp in tqdm(enumerate(set(corpus.timestamps))):
        timestamp_indices = np.where(corpus.timestamps == timestamp)[0]
        selected_corpus = corpus.get_corpus_by_indices(indices=timestamp_indices)
        documents_per_topic = selected_corpus.group_documents_by_topic()
        c_tf_idf, words = topic_model._c_tf_idf(documents_per_topic, fit=False)
        if global_tuning or evolution_tuning:
            c_tf_idf = normalize(c_tf_idf, axis=1, norm="l1", copy=False)

        # Fine-tune the c-TF-IDF matrix at timestamp t by averaging it with the c-TF-IDF
        # matrix at timestamp t-1
        if evolution_tuning and index != 0:
            current_topics = sorted(list(documents_per_topic.keys()))
            overlapping_topics = sorted(
                list(set(previous_topics).intersection(set(current_topics)))  # noqa: F821
            )

            current_overlap_idx = [current_topics.index(topic) for topic in overlapping_topics]
            previous_overlap_idx = [
                previous_topics.index(topic)  # noqa: F821
                for topic in overlapping_topics
            ]

            c_tf_idf.tolil()[current_overlap_idx] = (
                (
                    c_tf_idf[current_overlap_idx] + previous_c_tf_idf[previous_overlap_idx]  # noqa: F821
                )
                / 2.0
            ).tolil()

        # Fine-tune the timestamp c-TF-IDF representation based on the global c-TF-IDF representation
        # by simply taking the average of the two
        if global_tuning:
            topic_indices = {topic: index for index, topic in enumerate(selected_corpus.topic_labels)}
            selected_topics = [topic_indices[topic] for topic in documents_per_topic.keys()]
            c_tf_idf = (global_c_tf_idf[selected_topics] + c_tf_idf) / 2.0

        if evolution_tuning:
            previous_topics = sorted(list(documents_per_topic.keys()))  # noqa: F841
            previous_c_tf_idf = c_tf_idf.copy()  # noqa: F841

        # Extract the words per topic
        topic_representations = topic_model._extract_words_per_topic(
            words, selected_corpus, c_tf_idf, calculate_aspects=False, fine_tune=False
        )
        new_topics = Topics().initialize(selected_corpus.topics)
        new_topics.set_data(representations={"Main": topic_representations})
        topics[timestamp] = new_topics

    return topics


def bin_timestamps(corpus: Corpus, nr_bins: int) -> None:
    """Bins timestamps into specified number of bins and updates the corpus timestamps.

    Args:
        corpus: The Corpus object containing timestamps.
        nr_bins: The number of bins to create.
    """
    # Convert to datetime64[ns] first, then view as int64
    timestamps = corpus.timestamps.view(np.int64)

    # Create bin edges as integers
    bin_edges = np.linspace(timestamps.min(), timestamps.max(), nr_bins + 1).astype(np.int64)

    # Assign each timestamp to a bin (returns bin indices) and last bin should include the max value
    bin_indices = np.digitize(timestamps, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, nr_bins - 1)

    return bin_edges[bin_indices].astype("datetime64[ns]")
