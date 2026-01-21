"""Topics per class variation for BERTopic.

# Methodology

Instead of running the topic model per class, we can simply create a topic model and then extract,
for each topic, its representation per class. This allows you to see how certain topics, calculated over all documents,
are represented for certain subgroups.

"""

import numpy as np

from typing import TYPE_CHECKING
from sklearn.preprocessing import normalize
from tqdm import tqdm

from bertopic._utils import MyLogger
from bertopic._corpus import Corpus
from bertopic._topics import Topics

if TYPE_CHECKING:
    from bertopic import BERTopic

logger = MyLogger()
logger.configure("WARNING")


def topics_per_class(
    topic_model: "BERTopic",
    docs: list[str],
    classes: list[int | str],
    global_tuning: bool = True,
) -> dict[str, Topics]:
    """Create topics per class.

    To create the topics per class, BERTopic needs to be already fitted once.
    From the fitted models, the c-TF-IDF representations are calculated at
    each class c. Then, the c-TF-IDF representations at class c are
    averaged with the global c-TF-IDF representations in order to fine-tune the
    local representations. This can be turned off if the pure representation is
    needed.

    Note:
        Make sure to use a limited number of unique classes (<100) as the
        c-TF-IDF representation will be calculated at each single unique class.
        Having a large number of unique classes can take some time to be calculated.

    Arguments:
        topic_model: The BERTopic instance
        docs: The documents you used when calling either `fit` or `fit_transform`
        classes: The class of each document. This can be either a list of strings or ints.
        global_tuning: Fine-tune each topic representation for class c by averaging its c-TF-IDF matrix
                       with the global c-TF-IDF matrix. Turn this off if you want to prevent words in
                       topic representations that could not be found in the documents for class c.

    Returns:
        topics: A dictionary where keys are the unique classes and values are Topics objects
                representing the topics for that class.

    Examples:
    ```python
    from bertopic import BERTopic
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(docs)
    topics_per_class = topic_model.topics_per_class(docs, classes)
    ```
    """
    corpus = Corpus(documents=docs, topics=topic_model.topics_.predictions, classes=classes)
    global_c_tf_idf = normalize(topic_model.c_tf_idf_, axis=1, norm="l1", copy=False)

    # For each unique class, create topic representations
    topics = {}
    for _, class_ in tqdm(enumerate(set(classes)), disable=not topic_model.verbose):
        # Calculate c-TF-IDF representation for a specific class
        class_indices = np.where(corpus.classes == class_)[0]
        selected_corpus = corpus.get_corpus_by_indices(indices=class_indices)
        documents_per_topic = selected_corpus.group_documents_by_topic()
        c_tf_idf, words = topic_model._c_tf_idf(documents_per_topic, fit=False)

        # Fine-tune the class c-TF-IDF representation based on the global c-TF-IDF representation
        # by simply taking the average of the two
        if global_tuning:
            topic_indices = np.array(list(documents_per_topic.keys())) + topic_model._outliers
            c_tf_idf = normalize(c_tf_idf, axis=1, norm="l1", copy=False)
            c_tf_idf = (global_c_tf_idf[topic_indices] + c_tf_idf) / 2.0

        # Extract the words per topic
        topic_representations = topic_model._extract_words_per_topic(
            words, selected_corpus, c_tf_idf, calculate_aspects=False, fine_tune=False
        )
        new_topics = Topics().initialize(selected_corpus.topics)
        new_topics.set_data(representations={"Main": topic_representations})
        topics[class_] = new_topics

    return topics
