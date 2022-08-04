import yaml
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    yaml._warnings_enabled["YAMLLoadWarning"] = False
except (KeyError, AttributeError, TypeError) as e:
    pass

import re
import joblib
import inspect
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.cluster import hierarchy as sch
from typing import List, Tuple, Union, Mapping, Any, Callable, Iterable

# Models
import hdbscan
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# BERTopic
from bertopic._ctfidf import ClassTFIDF
from bertopic._utils import MyLogger, check_documents_type, check_embeddings_shape, check_is_fitted
from bertopic._mmr import mmr
from bertopic.backend._utils import select_backend
from bertopic import plotting

# Visualization
import plotly.graph_objects as go

logger = MyLogger("WARNING")


class BERTopic:
    """BERTopic is a topic modeling technique that leverages BERT embeddings and
    c-TF-IDF to create dense clusters allowing for easily interpretable topics
    whilst keeping important words in the topic descriptions.

    The default embedding model is `all-MiniLM-L6-v2` when selecting `language="english"` 
    and `paraphrase-multilingual-MiniLM-L12-v2` when selecting `language="multilingual"`.

    Usage:

    ```python
    from bertopic import BERTopic
    from sklearn.datasets import fetch_20newsgroups

    docs = fetch_20newsgroups(subset='all')['data']
    topic_model = BERTopic()
    topics, probabilities = topic_model.fit_transform(docs)
    ```

    If you want to use your own embedding model, use it as follows:

    ```python
    from bertopic import BERTopic
    from sklearn.datasets import fetch_20newsgroups
    from sentence_transformers import SentenceTransformer

    docs = fetch_20newsgroups(subset='all')['data']
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model = BERTopic(embedding_model=sentence_model)
    ```

    Due to the stochastisch nature of UMAP, the results from BERTopic might differ
    and the quality can degrade. Using your own embeddings allows you to
    try out BERTopic several times until you find the topics that suit
    you best.
    """
    def __init__(self,
                 language: str = "english",
                 top_n_words: int = 10,
                 n_gram_range: Tuple[int, int] = (1, 1),
                 min_topic_size: int = 10,
                 nr_topics: Union[int, str] = None,
                 low_memory: bool = False,
                 calculate_probabilities: bool = False,
                 diversity: float = None,
                 seed_topic_list: List[List[str]] = None,
                 embedding_model=None,
                 umap_model: UMAP = None,
                 hdbscan_model: hdbscan.HDBSCAN = None,
                 vectorizer_model: CountVectorizer = None,
                 verbose: bool = False,
                 ):
        """BERTopic initialization

        Arguments:
            language: The main language used in your documents. The default sentence-transformers 
                      model for "english" is `all-MiniLM-L6-v2`. For a full overview of
                      supported languages see bertopic.backend.languages. Select
                      "multilingual" to load in the `paraphrase-multilingual-MiniLM-L12-v2`
                      sentence-tranformers model that supports 50+ languages.
            top_n_words: The number of words per topic to extract. Setting this
                         too high can negatively impact topic embeddings as topics
                         are typically best represented by at most 10 words.
            n_gram_range: The n-gram range for the CountVectorizer.
                          Advised to keep high values between 1 and 3.
                          More would likely lead to memory issues.
                          NOTE: This param will not be used if you pass in your own
                          CountVectorizer.
            min_topic_size: The minimum size of the topic. Increasing this value will lead
                            to a lower number of clusters/topics.
            nr_topics: Specifying the number of topics will reduce the initial
                       number of topics to the value specified. This reduction can take
                       a while as each reduction in topics (-1) activates a c-TF-IDF
                       calculation. If this is set to None, no reduction is applied. Use
                       "auto" to automatically reduce topics using HDBSCAN.
            low_memory: Sets UMAP low memory to True to make sure less memory is used.
                        NOTE: This is only used in UMAP. For example, if you use PCA instead of UMAP
                        this parameter will not be used.
            calculate_probabilities: Whether to calculate the probabilities of all topics
                                     per document instead of the probability of the assigned
                                     topic per document. This could slow down the extraction
                                     of topics if you have many documents (> 100_000). Set this
                                     only to True if you have a low amount of documents or if
                                     you do not mind more computation time.
                                     NOTE: If false you cannot use the corresponding
                                     visualization method `visualize_probabilities`.
            diversity: Whether to use MMR to diversify the resulting topic representations.
                       If set to None, MMR will not be used. Accepted values lie between
                       0 and 1 with 0 being not at all diverse and 1 being very diverse.
            seed_topic_list: A list of seed words per topic to converge around
            verbose: Changes the verbosity of the model, Set to True if you want
                     to track the stages of the model.
            embedding_model: Use a custom embedding model.
                             The following backends are currently supported
                               * SentenceTransformers
                               * Flair
                               * Spacy
                               * Gensim
                               * USE (TF-Hub)
                             You can also pass in a string that points to one of the following
                             sentence-transformers models:
                               * https://www.sbert.net/docs/pretrained_models.html
            umap_model: Pass in a UMAP model to be used instead of the default.
                        NOTE: You can also pass in any dimensionality reduction algorithm as long
                        as it has `.fit` and `.transform` functions.
            hdbscan_model: Pass in a hdbscan.HDBSCAN model to be used instead of the default
                           NOTE: You can also pass in any clustering algorithm as long as it has
                           `.fit` and `.predict` functions along with the `.labels_` variable.
            vectorizer_model: Pass in a CountVectorizer instead of the default
        """
        # Topic-based parameters
        if top_n_words > 30:
            raise ValueError("top_n_words should be lower or equal to 30. The preferred value is 10.")
        self.top_n_words = top_n_words
        self.min_topic_size = min_topic_size
        self.nr_topics = nr_topics
        self.low_memory = low_memory
        self.calculate_probabilities = calculate_probabilities
        self.diversity = diversity
        self.verbose = verbose
        self.seed_topic_list = seed_topic_list

        # Embedding model
        self.language = language if not embedding_model else None
        self.embedding_model = embedding_model

        # Vectorizer
        self.n_gram_range = n_gram_range
        self.vectorizer_model = vectorizer_model or CountVectorizer(ngram_range=self.n_gram_range)

        # UMAP or another algorithm that has .fit and .transform functions
        self.umap_model = umap_model or UMAP(n_neighbors=15,
                                             n_components=5,
                                             min_dist=0.0,
                                             metric='cosine',
                                             low_memory=self.low_memory)

        # HDBSCAN or another clustering algorithm that has .fit and .predict functions and
        # the .labels_ variable to extract the labels
        self.hdbscan_model = hdbscan_model or hdbscan.HDBSCAN(min_cluster_size=self.min_topic_size,
                                                              metric='euclidean',
                                                              cluster_selection_method='eom',
                                                              prediction_data=True)

        # Attributes
        self.topics = None
        self.topic_mapper = None
        self.topic_sizes = None
        self.merged_topics = None
        self.custom_labels = None
        self.topic_embeddings = None
        self.representative_docs = None
        self._outliers = 1

        if verbose:
            logger.set_level("DEBUG")

    def fit(self,
            documents: List[str],
            embeddings: np.ndarray = None,
            y: Union[List[int], np.ndarray] = None):
        """ Fit the models (Bert, UMAP, and, HDBSCAN) on a collection of documents and generate topics

        Arguments:
            documents: A list of documents to fit on
            embeddings: Pre-trained document embeddings. These can be used
                        instead of the sentence-transformer model
            y: The target class for (semi)-supervised modeling. Use -1 if no class for a
               specific instance is specified.

        Usage:

        ```python
        from bertopic import BERTopic
        from sklearn.datasets import fetch_20newsgroups

        docs = fetch_20newsgroups(subset='all')['data']
        topic_model = BERTopic().fit(docs)
        ```

        If you want to use your own embeddings, use it as follows:

        ```python
        from bertopic import BERTopic
        from sklearn.datasets import fetch_20newsgroups
        from sentence_transformers import SentenceTransformer

        # Create embeddings
        docs = fetch_20newsgroups(subset='all')['data']
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = sentence_model.encode(docs, show_progress_bar=True)

        # Create topic model
        topic_model = BERTopic().fit(docs, embeddings)
        ```
        """
        self.fit_transform(documents, embeddings, y)
        return self

    def fit_transform(self,
                      documents: List[str],
                      embeddings: np.ndarray = None,
                      y: Union[List[int], np.ndarray] = None) -> Tuple[List[int],
                                                                       Union[np.ndarray, None]]:
        """ Fit the models on a collection of documents, generate topics, and return the docs with topics

        Arguments:
            documents: A list of documents to fit on
            embeddings: Pre-trained document embeddings. These can be used
                        instead of the sentence-transformer model
            y: The target class for (semi)-supervised modeling. Use -1 if no class for a
               specific instance is specified.

        Returns:
            predictions: Topic predictions for each documents
            probabilities: The probability of the assigned topic per document.
                           If `calculate_probabilities` in BERTopic is set to True, then
                           it calculates the probabilities of all topics across all documents
                           instead of only the assigned topic. This, however, slows down
                           computation and may increase memory usage.

        Usage:

        ```python
        from bertopic import BERTopic
        from sklearn.datasets import fetch_20newsgroups

        docs = fetch_20newsgroups(subset='all')['data']
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(docs)
        ```

        If you want to use your own embeddings, use it as follows:

        ```python
        from bertopic import BERTopic
        from sklearn.datasets import fetch_20newsgroups
        from sentence_transformers import SentenceTransformer

        # Create embeddings
        docs = fetch_20newsgroups(subset='all')['data']
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = sentence_model.encode(docs, show_progress_bar=True)

        # Create topic model
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(docs, embeddings)
        ```
        """
        check_documents_type(documents)
        check_embeddings_shape(embeddings, documents)

        documents = pd.DataFrame({"Document": documents,
                                  "ID": range(len(documents)),
                                  "Topic": None})

        # Extract embeddings
        if embeddings is None:
            self.embedding_model = select_backend(self.embedding_model,
                                                  language=self.language)
            embeddings = self._extract_embeddings(documents.Document,
                                                  method="document",
                                                  verbose=self.verbose)
            logger.info("Transformed documents to Embeddings")
        else:
            if self.embedding_model is not None:
                self.embedding_model = select_backend(self.embedding_model,
                                                      language=self.language)

        # Reduce dimensionality
        if self.seed_topic_list is not None and self.embedding_model is not None:
            y, embeddings = self._guided_topic_modeling(embeddings)
        umap_embeddings = self._reduce_dimensionality(embeddings, y)

        # Cluster reduced embeddings
        documents, probabilities = self._cluster_embeddings(umap_embeddings, documents)

        # Sort and Map Topic IDs by their frequency
        if not self.nr_topics:
            documents = self._sort_mappings_by_frequency(documents)

        # Extract topics by calculating c-TF-IDF
        self._extract_topics(documents)

        # Reduce topics
        if self.nr_topics:
            documents = self._reduce_topics(documents)

        self._map_representative_docs(original_topics=True)
        probabilities = self._map_probabilities(probabilities, original_topics=True)
        predictions = documents.Topic.to_list()

        return predictions, probabilities

    def transform(self,
                  documents: Union[str, List[str]],
                  embeddings: np.ndarray = None) -> Tuple[List[int], np.ndarray]:
        """ After having fit a model, use transform to predict new instances

        Arguments:
            documents: A single document or a list of documents to fit on
            embeddings: Pre-trained document embeddings. These can be used
                        instead of the sentence-transformer model.

        Returns:
            predictions: Topic predictions for each documents
            probabilities: The topic probability distribution which is returned by default.
                           If `calculate_probabilities` in BERTopic is set to False, then the
                           probabilities are not calculated to speed up computation and
                           decrease memory usage.

        Usage:

        ```python
        from bertopic import BERTopic
        from sklearn.datasets import fetch_20newsgroups

        docs = fetch_20newsgroups(subset='all')['data']
        topic_model = BERTopic().fit(docs)
        topics, probs = topic_model.transform(docs)
        ```

        If you want to use your own embeddings:

        ```python
        from bertopic import BERTopic
        from sklearn.datasets import fetch_20newsgroups
        from sentence_transformers import SentenceTransformer

        # Create embeddings
        docs = fetch_20newsgroups(subset='all')['data']
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = sentence_model.encode(docs, show_progress_bar=True)

        # Create topic model
        topic_model = BERTopic().fit(docs, embeddings)
        topics, probs = topic_model.transform(docs, embeddings)
        ```
        """
        check_is_fitted(self)
        check_embeddings_shape(embeddings, documents)

        if isinstance(documents, str):
            documents = [documents]

        if embeddings is None:
            embeddings = self._extract_embeddings(documents,
                                                  method="document",
                                                  verbose=self.verbose)

        umap_embeddings = self.umap_model.transform(embeddings)
        logger.info("Reduced dimensionality")

        # Extract predictions and probabilities if it is a HDBSCAN model
        if isinstance(self.hdbscan_model, hdbscan.HDBSCAN):
            predictions, probabilities = hdbscan.approximate_predict(self.hdbscan_model, umap_embeddings)

            # Calculate probabilities
            if self.calculate_probabilities:
                probabilities = hdbscan.membership_vector(self.hdbscan_model, umap_embeddings)
                logger.info("Calculated probabilities with HDBSCAN")

        else:
            predictions = self.hdbscan_model.predict(umap_embeddings)
            probabilities = None
        logger.info("Predicted clusters")

        # Map probabilities and predictions
        probabilities = self._map_probabilities(probabilities, original_topics=True)
        predictions = self._map_predictions(predictions)
        return predictions, probabilities

    def topics_over_time(self,
                         docs: List[str],
                         topics: List[int],
                         timestamps: Union[List[str],
                                           List[int]],
                         nr_bins: int = None,
                         datetime_format: str = None,
                         evolution_tuning: bool = True,
                         global_tuning: bool = True) -> pd.DataFrame:
        """ Create topics over time

        To create the topics over time, BERTopic needs to be already fitted once.
        From the fitted models, the c-TF-IDF representations are calculate at
        each timestamp t. Then, the c-TF-IDF representations at timestamp t are
        averaged with the global c-TF-IDF representations in order to fine-tune the
        local representations.

        NOTE:
            Make sure to use a limited number of unique timestamps (<100) as the
            c-TF-IDF representation will be calculated at each single unique timestamp.
            Having a large number of unique timestamps can take some time to be calculated.
            Moreover, there aren't many use-cased where you would like to see the difference
            in topic representations over more than 100 different timestamps.

        Arguments:
            docs: The documents you used when calling either `fit` or `fit_transform`
            topics: The topics that were returned when calling either `fit` or `fit_transform`
            timestamps: The timestamp of each document. This can be either a list of strings or ints.
                        If it is a list of strings, then the datetime format will be automatically
                        inferred. If it is a list of ints, then the documents will be ordered by
                        ascending order.
            nr_bins: The number of bins you want to create for the timestamps. The left interval will
                     be chosen as the timestamp. An additional column will be created with the
                     entire interval.
            datetime_format: The datetime format of the timestamps if they are strings, eg “%d/%m/%Y”.
                             Set this to None if you want to have it automatically detect the format.
                             See strftime documentation for more information on choices:
                             https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior.
            evolution_tuning: Fine-tune each topic representation at timestamp t by averaging its
                              c-TF-IDF matrix with the c-TF-IDF matrix at timestamp t-1. This creates
                              evolutionary topic representations.
            global_tuning: Fine-tune each topic representation at timestamp t by averaging its c-TF-IDF matrix
                       with the global c-TF-IDF matrix. Turn this off if you want to prevent words in
                       topic representations that could not be found in the documents at timestamp t.

        Returns:
            topics_over_time: A dataframe that contains the topic, words, and frequency of topic
                              at timestamp t.

        Usage:

        The timestamps variable represent the timestamp of each document. If you have over
        100 unique timestamps, it is advised to bin the timestamps as shown below:

        ```python
        from bertopic import BERTopic
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(docs)
        topics_over_time = topic_model.topics_over_time(docs, topics, timestamps, nr_bins=20)
        ```
        """
        check_is_fitted(self)
        check_documents_type(docs)
        documents = pd.DataFrame({"Document": docs, "Topic": topics, "Timestamps": timestamps})
        global_c_tf_idf = normalize(self.c_tf_idf, axis=1, norm='l1', copy=False)

        all_topics = sorted(list(documents.Topic.unique()))
        all_topics_indices = {topic: index for index, topic in enumerate(all_topics)}

        if isinstance(timestamps[0], str):
            infer_datetime_format = True if not datetime_format else False
            documents["Timestamps"] = pd.to_datetime(documents["Timestamps"],
                                                     infer_datetime_format=infer_datetime_format,
                                                     format=datetime_format)

        if nr_bins:
            documents["Bins"] = pd.cut(documents.Timestamps, bins=nr_bins)
            documents["Timestamps"] = documents.apply(lambda row: row.Bins.left, 1)

        # Sort documents in chronological order
        documents = documents.sort_values("Timestamps")
        timestamps = documents.Timestamps.unique()
        if len(timestamps) > 100:
            warnings.warn(f"There are more than 100 unique timestamps (i.e., {len(timestamps)}) "
                          "which significantly slows down the application. Consider setting `nr_bins` "
                          "to a value lower than 100 to speed up calculation. ")

        # For each unique timestamp, create topic representations
        topics_over_time = []
        for index, timestamp in tqdm(enumerate(timestamps), disable=not self.verbose):

            # Calculate c-TF-IDF representation for a specific timestamp
            selection = documents.loc[documents.Timestamps == timestamp, :]
            documents_per_topic = selection.groupby(['Topic'], as_index=False).agg({'Document': ' '.join,
                                                                                    "Timestamps": "count"})
            c_tf_idf, words = self._c_tf_idf(documents_per_topic, fit=False)

            if global_tuning or evolution_tuning:
                c_tf_idf = normalize(c_tf_idf, axis=1, norm='l1', copy=False)

            # Fine-tune the c-TF-IDF matrix at timestamp t by averaging it with the c-TF-IDF
            # matrix at timestamp t-1
            if evolution_tuning and index != 0:
                current_topics = sorted(list(documents_per_topic.Topic.values))
                overlapping_topics = sorted(list(set(previous_topics).intersection(set(current_topics))))

                current_overlap_idx = [current_topics.index(topic) for topic in overlapping_topics]
                previous_overlap_idx = [previous_topics.index(topic) for topic in overlapping_topics]

                c_tf_idf.tolil()[current_overlap_idx] = ((c_tf_idf[current_overlap_idx] +
                                                          previous_c_tf_idf[previous_overlap_idx]) / 2.0).tolil()

            # Fine-tune the timestamp c-TF-IDF representation based on the global c-TF-IDF representation
            # by simply taking the average of the two
            if global_tuning:
                selected_topics = [all_topics_indices[topic] for topic in documents_per_topic.Topic.values]
                c_tf_idf = (global_c_tf_idf[selected_topics] + c_tf_idf) / 2.0

            # Extract the words per topic
            labels = sorted(list(documents_per_topic.Topic.unique()))
            words_per_topic = self._extract_words_per_topic(words, c_tf_idf, labels)
            topic_frequency = pd.Series(documents_per_topic.Timestamps.values,
                                        index=documents_per_topic.Topic).to_dict()

            # Fill dataframe with results
            topics_at_timestamp = [(topic,
                                    ", ".join([words[0] for words in values][:5]),
                                    topic_frequency[topic],
                                    timestamp) for topic, values in words_per_topic.items()]
            topics_over_time.extend(topics_at_timestamp)

            if evolution_tuning:
                previous_topics = sorted(list(documents_per_topic.Topic.values))
                previous_c_tf_idf = c_tf_idf.copy()

        return pd.DataFrame(topics_over_time, columns=["Topic", "Words", "Frequency", "Timestamp"])

    def topics_per_class(self,
                         docs: List[str],
                         topics: List[int],
                         classes: Union[List[int], List[str]],
                         global_tuning: bool = True) -> pd.DataFrame:
        """ Create topics per class

        To create the topics per class, BERTopic needs to be already fitted once.
        From the fitted models, the c-TF-IDF representations are calculate at
        each class c. Then, the c-TF-IDF representations at class c are
        averaged with the global c-TF-IDF representations in order to fine-tune the
        local representations. This can be turned off if the pure representation is
        needed.

        NOTE:
            Make sure to use a limited number of unique classes (<100) as the
            c-TF-IDF representation will be calculated at each single unique class.
            Having a large number of unique classes can take some time to be calculated.

        Arguments:
            docs: The documents you used when calling either `fit` or `fit_transform`
            topics: The topics that were returned when calling either `fit` or `fit_transform`
            classes: The class of each document. This can be either a list of strings or ints.
            global_tuning: Fine-tune each topic representation at timestamp t by averaging its c-TF-IDF matrix
                       with the global c-TF-IDF matrix. Turn this off if you want to prevent words in
                       topic representations that could not be found in the documents at timestamp t.

        Returns:
            topics_per_class: A dataframe that contains the topic, words, and frequency of topics
                              for each class.

        Usage:

        ```python
        from bertopic import BERTopic
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(docs)
        topics_per_class = topic_model.topics_per_class(docs, topics, classes)
        ```
        """
        documents = pd.DataFrame({"Document": docs, "Topic": topics, "Class": classes})
        global_c_tf_idf = normalize(self.c_tf_idf, axis=1, norm='l1', copy=False)

        # For each unique timestamp, create topic representations
        topics_per_class = []
        for index, class_ in tqdm(enumerate(set(classes)), disable=not self.verbose):

            # Calculate c-TF-IDF representation for a specific timestamp
            selection = documents.loc[documents.Class == class_, :]
            documents_per_topic = selection.groupby(['Topic'], as_index=False).agg({'Document': ' '.join,
                                                                                    "Class": "count"})
            c_tf_idf, words = self._c_tf_idf(documents_per_topic, fit=False)

            # Fine-tune the timestamp c-TF-IDF representation based on the global c-TF-IDF representation
            # by simply taking the average of the two
            if global_tuning:
                c_tf_idf = normalize(c_tf_idf, axis=1, norm='l1', copy=False)
                c_tf_idf = (global_c_tf_idf[documents_per_topic.Topic.values + self._outliers] + c_tf_idf) / 2.0

            # Extract the words per topic
            labels = sorted(list(documents_per_topic.Topic.unique()))
            words_per_topic = self._extract_words_per_topic(words, c_tf_idf, labels)
            topic_frequency = pd.Series(documents_per_topic.Class.values,
                                        index=documents_per_topic.Topic).to_dict()

            # Fill dataframe with results
            topics_at_class = [(topic,
                                ", ".join([words[0] for words in values][:5]),
                                topic_frequency[topic],
                                class_) for topic, values in words_per_topic.items()]
            topics_per_class.extend(topics_at_class)

        topics_per_class = pd.DataFrame(topics_per_class, columns=["Topic", "Words", "Frequency", "Class"])

        return topics_per_class

    def hierarchical_topics(self,
                            docs: List[int],
                            topics: List[int],
                            linkage_function: Callable[[csr_matrix], np.ndarray] = None,
                            distance_function: Callable[[csr_matrix], csr_matrix] = None) -> pd.DataFrame:
        """ Create a hierarchy of topics

        To create this hierarchy, BERTopic needs to be already fitted once.
        Then, a hierarchy is calculated on the distance matrix of the c-TF-IDF
        representation using `scipy.cluster.hierarchy.linkage`.

        Based on that hierarchy, we calculate the topic representation at each
        merged step. This is a local representation, as we only assume that the
        chosen step is merged and not all others which typically improves the
        topic representation.

        Arguments:
            docs: The documents you used when calling either `fit` or `fit_transform`
            topics: The topics that were returned when calling either `fit` or `fit_transform`
            linkage_function: The linkage function to use. Default is:
                            `lambda x: sch.linkage(x, 'ward', optimal_ordering=True)`
            distance_function: The distance function to use on the c-TF-IDF matrix. Default is:
                                `lambda x: 1 - cosine_similarity(x)`

        Returns:
            hierarchical_topics: A dataframe that contains a hierarchy of topics
                                represented by their parents and their children

        Usage:

        ```python
        from bertopic import BERTopic
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(docs)
        hierarchical_topics = topic_model.hierarchical_topics(docs, topics)
        ```

        A custom linkage function can be used as follows:

        ```python
        from scipy.cluster import hierarchy as sch
        from bertopic import BERTopic
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(docs)

        # Hierarchical topics
        linkage_function = lambda x: sch.linkage(x, 'ward', optimal_ordering=True)
        hierarchical_topics = topic_model.hierarchical_topics(docs, topics, linkage_function=linkage_function)
        ```
        """
        if distance_function is None:
            distance_function = lambda x: 1 - cosine_similarity(x)

        if linkage_function is None:
            linkage_function = lambda x: sch.linkage(x, 'ward', optimal_ordering=True)

        # Calculate linkage
        embeddings = self.c_tf_idf[self._outliers:]
        X = distance_function(embeddings)
        Z = linkage_function(X)

        # Calculate basic bag-of-words to be iteratively merged later
        documents = pd.DataFrame({"Document": docs,
                                  "ID": range(len(docs)),
                                  "Topic": topics})
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        documents_per_topic = documents_per_topic.loc[documents_per_topic.Topic != -1, :]
        documents = self._preprocess_text(documents_per_topic.Document.values)
        words = self.vectorizer_model.get_feature_names()
        bow = self.vectorizer_model.transform(documents)

        # Extract clusters
        hier_topics = pd.DataFrame(columns=["Parent_ID", "Parent_Name", "Topics",
                                            "Child_Left_ID", "Child_Left_Name",
                                            "Child_Right_ID", "Child_Right_Name"])
        for index in tqdm(range(len(Z))):

            # Find clustered documents
            clusters = sch.fcluster(Z, t=Z[index][2], criterion='distance') - self._outliers
            cluster_df = pd.DataFrame({"Topic": range(len(clusters)), "Cluster": clusters})
            cluster_df = cluster_df.groupby("Cluster").agg({'Topic': lambda x: list(x)}).reset_index()
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
            c_tf_idf = self.transformer.transform(grouped)
            words_per_topic = self._extract_words_per_topic(words, c_tf_idf, labels=[0])

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
            hier_topics.loc[len(hier_topics), :] = [parent_id, parent_name,
                                                    clustered_topics,
                                                    int(Z[index][0]), child_left_name,
                                                    int(Z[index][1]), child_right_name]

        hier_topics["Distance"] = Z[:, 2]
        hier_topics = hier_topics.sort_values("Parent_ID", ascending=False)
        hier_topics[["Parent_ID", "Child_Left_ID", "Child_Right_ID"]] = hier_topics[["Parent_ID", "Child_Left_ID", "Child_Right_ID"]].astype(str)

        return hier_topics

    def find_topics(self,
                    search_term: str,
                    top_n: int = 5) -> Tuple[List[int], List[float]]:
        """ Find topics most similar to a search_term

        Creates an embedding for search_term and compares that with
        the topic embeddings. The most similar topics are returned
        along with their similarity values.

        The search_term can be of any size but since it compares
        with the topic representation it is advised to keep it
        below 5 words.

        Arguments:
            search_term: the term you want to use to search for topics
            top_n: the number of topics to return

        Returns:
            similar_topics: the most similar topics from high to low
            similarity: the similarity scores from high to low

        Usage:

        You can use the underlying embedding model to find topics that
        best represent the search term:

        ```python
        topics, similarity = topic_model.find_topics("sports", top_n=5)
        ```

        Note that the search query is typically more accurate if the
        search_term consists of a phrase or multiple words.
        """
        if self.embedding_model is None:
            raise Exception("This method can only be used if you did not use custom embeddings.")

        topic_list = list(self.topics.keys())
        topic_list.sort()

        # Extract search_term embeddings and compare with topic embeddings
        search_embedding = self._extract_embeddings([search_term],
                                                    method="word",
                                                    verbose=False).flatten()
        sims = cosine_similarity(search_embedding.reshape(1, -1), self.topic_embeddings).flatten()

        # Extract topics most similar to search_term
        ids = np.argsort(sims)[-top_n:]
        similarity = [sims[i] for i in ids][::-1]
        similar_topics = [topic_list[index] for index in ids][::-1]

        return similar_topics, similarity

    def update_topics(self,
                      docs: List[str],
                      topics: List[int],
                      n_gram_range: Tuple[int, int] = None,
                      vectorizer_model: CountVectorizer = None):
        """ Updates the topic representation by recalculating c-TF-IDF with the new
        parameters as defined in this function.

        When you have trained a model and viewed the topics and the words that represent them,
        you might not be satisfied with the representation. Perhaps you forgot to remove
        stop_words or you want to try out a different n_gram_range. This function allows you
        to update the topic representation after they have been formed.

        Arguments:
            docs: The documents you used when calling either `fit` or `fit_transform`
            topics: The topics that were returned when calling either `fit` or `fit_transform`
            n_gram_range: The n-gram range for the CountVectorizer.
            vectorizer_model: Pass in your own CountVectorizer from scikit-learn

        Usage:

        In order to update the topic representation, you will need to first fit the topic
        model and extract topics from them. Based on these, you can update the representation:

        ```python
        topic_model.update_topics(docs, topics, n_gram_range=(2, 3))
        ```

        YOu can also use a custom vectorizer to update the representation:

        ```python
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
        topic_model.update_topics(docs, topics, vectorizer_model=vectorizer_model)
        ```
        """
        check_is_fitted(self)
        if not n_gram_range:
            n_gram_range = self.n_gram_range

        self.vectorizer_model = vectorizer_model or CountVectorizer(ngram_range=n_gram_range)

        documents = pd.DataFrame({"Document": docs, "Topic": topics})
        self._extract_topics(documents)

    def get_topics(self) -> Mapping[str, Tuple[str, float]]:
        """ Return topics with top n words and their c-TF-IDF score

        Returns:
            self.topic: The top n words per topic and the corresponding c-TF-IDF score

        Usage:

        ```python
        all_topics = topic_model.get_topics()
        ```
        """
        check_is_fitted(self)
        return self.topics

    def get_topic(self, topic: int) -> Union[Mapping[str, Tuple[str, float]], bool]:
        """ Return top n words for a specific topic and their c-TF-IDF scores

        Arguments:
            topic: A specific topic for which you want its representation

        Returns:
            The top n words for a specific word and its respective c-TF-IDF scores

        Usage:

        ```python
        topic = topic_model.get_topic(12)
        ```
        """
        check_is_fitted(self)
        if topic in self.topics:
            return self.topics[topic]
        else:
            return False

    def get_topic_info(self, topic: int = None) -> pd.DataFrame:
        """ Get information about each topic including its ID, frequency, and name.

        Arguments:
            topic: A specific topic for which you want the frequency

        Returns:
            info: The information relating to either a single topic or all topics

        Usage:

        ```python
        info_df = topic_model.get_topic_info()
        ```
        """
        check_is_fitted(self)

        info = pd.DataFrame(self.topic_sizes.items(), columns=["Topic", "Count"]).sort_values("Topic")
        info["Name"] = info.Topic.map(self.topic_names)

        if self.custom_labels is not None:
            if len(self.custom_labels) == len(info):
                labels = {topic - self._outliers: label for topic, label in enumerate(self.custom_labels)}
                info["CustomName"] = info["Topic"].map(labels)

        if topic is not None:
            info = info.loc[info.Topic == topic, :]

        return info.reset_index(drop=True)

    def get_topic_freq(self, topic: int = None) -> Union[pd.DataFrame, int]:
        """ Return the the size of topics (descending order)

        Arguments:
            topic: A specific topic for which you want the frequency

        Returns:
            Either the frequency of a single topic or dataframe with
            the frequencies of all topics

        Usage:

        To extract the frequency of all topics:

        ```python
        frequency = topic_model.get_topic_freq()
        ```

        To get the frequency of a single topic:

        ```python
        frequency = topic_model.get_topic_freq(12)
        ```
        """
        check_is_fitted(self)
        if isinstance(topic, int):
            return self.topic_sizes[topic]
        else:
            return pd.DataFrame(self.topic_sizes.items(), columns=['Topic', 'Count']).sort_values("Count",
                                                                                                  ascending=False)

    def get_representative_docs(self, topic: int = None) -> List[str]:
        """ Extract representative documents per topic

        Arguments:
            topic: A specific topic for which you want
                   the representative documents

        Returns:
            Representative documents of the chosen topic

        Usage:

        To extract the representative docs of all topics:

        ```python
        representative_docs = topic_model.get_representative_docs()
        ```

        To get the representative docs of a single topic:

        ```python
        representative_docs = topic_model.get_representative_docs(12)
        ```
        """
        check_is_fitted(self)
        if isinstance(topic, int):
            return self.representative_docs[topic]
        else:
            return self.representative_docs

    @staticmethod
    def get_topic_tree(hier_topics: pd.DataFrame,
                       max_distance: float = None,
                       tight_layout: bool = False) -> str:
        """ Extract the topic tree such that it can be printed

        Arguments:
            hier_topics: A dataframe containing the structure of the topic tree.
                        This is the output of `topic_model.hierachical_topics()`
            max_distance: The maximum distance between two topics. This value is
                        based on the Distance column in `hier_topics`.
            tight_layout: Whether to use a tight layout (narrow width) for
                        easier readability if you have hundreds of topics.

        Returns:
            A tree that has the following structure when printed:
                .
                .
                └─health_medical_disease_patients_hiv
                    ├─patients_medical_disease_candida_health
                    │    ├─■──candida_yeast_infection_gonorrhea_infections ── Topic: 48
                    │    └─patients_disease_cancer_medical_doctor
                    │         ├─■──hiv_medical_cancer_patients_doctor ── Topic: 34
                    │         └─■──pain_drug_patients_disease_diet ── Topic: 26
                    └─■──health_newsgroup_tobacco_vote_votes ── Topic: 9

            The blocks (■) indicate that the topic is one you can directly access
            from `topic_model.get_topic`. In other words, they are the original un-grouped topics.

        Usage:

        ```python
        # Train model
        from bertopic import BERTopic
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(docs)
        hierarchical_topics = topic_model.hierarchical_topics(docs, topics)

        # Print topic tree
        tree = topic_model.get_topic_tree(hierarchical_topics)
        print(tree)
        ```
        """
        width = 1 if tight_layout else 4
        if max_distance is None:
            max_distance = hier_topics.Distance.max() + 1

        max_original_topic = hier_topics.Parent_ID.astype(int).min() - 1

        # Extract mapping from ID to name
        topic_to_name = dict(zip(hier_topics.Child_Left_ID, hier_topics.Child_Left_Name))
        topic_to_name.update(dict(zip(hier_topics.Child_Right_ID, hier_topics.Child_Right_Name)))
        topic_to_name = {topic: name[:100] for topic, name in topic_to_name.items()}

        # Create tree
        tree = {str(row[1].Parent_ID): [str(row[1].Child_Left_ID), str(row[1].Child_Right_ID)]
                for row in hier_topics.iterrows()}

        def get_tree(start, tree):
            """ Based on: https://stackoverflow.com/a/51920869/10532563 """

            def _tree(to_print, start, parent, tree, grandpa=None, indent=""):

                # Get distance between merged topics
                distance = hier_topics.loc[(hier_topics.Child_Left_ID == parent) |
                                           (hier_topics.Child_Right_ID == parent), "Distance"]
                distance = distance.values[0] if len(distance) > 0 else 10

                if parent != start:
                    if grandpa is None:
                        to_print += topic_to_name[parent]
                    else:
                        if int(parent) <= max_original_topic:

                            # Do not append topic ID if they are not merged
                            if distance < max_distance:
                                to_print += "■──" + topic_to_name[parent] + f" ── Topic: {parent}" + "\n"
                            else:
                                to_print += "O \n"
                        else:
                            to_print += topic_to_name[parent] + "\n"

                if parent not in tree:
                    return to_print

                for child in tree[parent][:-1]:
                    to_print += indent + "├" + "─"
                    to_print = _tree(to_print, start, child, tree, parent, indent + "│" + " " * width)

                child = tree[parent][-1]
                to_print += indent + "└" + "─"
                to_print = _tree(to_print, start, child, tree, parent, indent + " " * (width+1))

                return to_print

            to_print = "." + "\n"
            to_print = _tree(to_print, start, start, tree)
            return to_print

        start = str(hier_topics.Parent_ID.astype(int).max())
        return get_tree(start, tree)

    def set_topic_labels(self, topic_labels: Union[List[str], Mapping[int, str]]) -> None:
        """ Set custom topic labels in your fitted BERTopic model

        Arguments:
            topic_labels: If a list of topic labels, it should contain the same number
                        of labels as there are topics. This must be ordered
                        from the topic with the lowest ID to the highest ID,
                        including topic -1 if it exists.
                        If a dictionary of `topic ID`: `topic_label`, it can have
                        any number of topics as it will only map the topics found
                        in the dictionary.

        Usage:

        First, we define our topic labels with `.get_topic_labels` in which
        we can customize our topic labels:

        ```python
        topic_labels = topic_model.get_topic_labels(nr_words=2,
                                                    topic_prefix=True,
                                                    word_length=10,
                                                    separator=", ")
        ```

        Then, we pass these `topic_labels` to our topic model which
        can be accessed at any time with `.custom_labels`:

        ```python
        topic_model.set_topic_labels(topic_labels)
        topic_model.custom_labels
        ```

        You might want to change only a few topic labels instead of all of them.
        To do so, you can pass a dictionary where the keys are the topic IDs and
        its keys the topic labels:

        ```python
        topic_model.set_topic_labels({0: "Space", 1: "Sports", 2: "Medicine"})
        topic_model.custom_labels
        ```
        """
        unique_topics = sorted(set(self._map_predictions(self.hdbscan_model.labels_)))

        if isinstance(topic_labels, dict):
            if self.custom_labels is not None:
                original_labels = {topic: label for topic, label in zip(unique_topics, self.custom_labels)}
            else:
                info = self.get_topic_info()
                original_labels = dict(zip(info.Topic, info.Name))
            custom_labels = [topic_labels.get(topic) if topic_labels.get(topic) else original_labels[topic] for topic in unique_topics]

        elif isinstance(topic_labels, list):
            if len(topic_labels) == len(unique_topics):
                custom_labels = topic_labels
            else:
                raise ValueError("Make sure that `topic_labels` contains the same number "
                                 "of labels as that there are topics.")

        self.custom_labels = custom_labels

    def generate_topic_labels(self,
                              nr_words: int = 3,
                              topic_prefix: bool = True,
                              word_length: int = None,
                              separator: str = "_") -> List[str]:
        """ Get labels for each topic in a user-defined format

        Arguments:
            original_labels:
            nr_words: Top `n` words per topic to use
            topic_prefix: Whether to use the topic ID as a prefix.
                        If set to True, the topic ID will be separated
                        using the `separator`
            word_length: The maximum length of each word in the topic label.
                        Some words might be relatively long and setting this
                        value helps to make sure that all labels have relatively
                        similar lengths.
            separator: The string with which the words and topic prefix will be
                    separated. Underscores are the default but a nice alternative
                    is `", "`.

        Returns:
            topic_labels: A list of topic labels sorted from the lowest topic ID to the highest.
                        If the topic model was trained using HDBSCAN, the lowest topic ID is -1,
                        otherwise it is 0.

        Usage:

        To create our custom topic labels, usage is rather straightforward:

        ```python
        topic_labels = topic_model.get_topic_labels(nr_words=2, separator=", ")
        ```
        """
        unique_topics = sorted(set(self._map_predictions(self.hdbscan_model.labels_)))
        topic_labels = []
        for topic in unique_topics:
            words, _ = zip(*self.get_topic(topic))

            if word_length:
                words = [word[:word_length] for word in words][:nr_words]
            else:
                words = list(words)[:nr_words]

            if topic_prefix:
                topic_label = f"{topic}{separator}" + separator.join(words)
            else:
                topic_label = separator.join(words)

            topic_labels.append(topic_label)

        return topic_labels

    def merge_topics(self,
                     docs: List[str],
                     topics: List[int],
                     topics_to_merge: List[Union[Iterable[int], int]]) -> None:
        """
        Arguments:
            docs: The documents you used when calling either `fit` or `fit_transform`
            topics: The topics that were returned when calling either `fit` or `fit_transform`
            topics_to_merge: Either a list of topics or a list of list of topics
                            to merge. For example:
                                [1, 2, 3] will merge topics 1, 2 and 3
                                [[1, 2], [3, 4]] will merge topics 1 and 2, and
                                separately merge topics 3 and 4.

        Usage:

        If you want to merge topics 1, 2, and 3:

        ```python
        topics_to_merge = [1, 2, 3]
        topic_model.merge_topics(docs, topics, topics_to_merge)
        ```

        or if you want to merge topics 1 and 2, and separately
        merge topics 3 and 4:

        ```python
        topics_to_merge = [[1, 2]
                            [3, 4]]
        topic_model.merge_topics(docs, topics, topics_to_merge)
        ```
        """
        check_is_fitted(self)
        documents = pd.DataFrame({"Document": docs, "Topic": topics})

        mapping = {topic: topic for topic in set(topics)}
        if isinstance(topics_to_merge[0], int):
            for topic in sorted(topics_to_merge):
                mapping[topic] = topics_to_merge[0]
        elif isinstance(topics_to_merge[0], Iterable):
            for topic_group in sorted(topics_to_merge):
                for topic in topic_group:
                    mapping[topic] = topic_group[0]
        else:
            raise ValueError("Make sure that `topics_to_merge` is either"
                             "a list of topics or a list of list of topics.")

        documents.Topic = documents.Topic.map(mapping)
        documents = self._sort_mappings_by_frequency(documents)
        self._extract_topics(documents)
        self._update_topic_size(documents)

    def reduce_topics(self,
                      docs: List[str],
                      topics: List[int],
                      probabilities: np.ndarray = None,
                      nr_topics: int = 20) -> Tuple[List[int], np.ndarray]:
        """ Further reduce the number of topics to nr_topics.

        The number of topics is further reduced by calculating the c-TF-IDF matrix
        of the documents and then reducing them by iteratively merging the least
        frequent topic with the most similar one based on their c-TF-IDF matrices.
        The topics, their sizes, and representations are updated.

        The reasoning for putting `docs`, `topics`, and `probs` as parameters is that
        these values are not saved within BERTopic on purpose. If you were to have a
        million documents, it seems very inefficient to save those in BERTopic
        instead of a dedicated database.

        Arguments:
            docs: The docs you used when calling either `fit` or `fit_transform`
            topics: The topics that were returned when calling either `fit` or `fit_transform`
            probabilities: The probabilities that were returned when calling either `fit` or `fit_transform`
            nr_topics: The number of topics you want reduced to

        Returns:
            new_topics: Updated topics
            new_probabilities: Updated probabilities

        Usage:

        You can further reduce the topics by passing the documents with its
        topics and probabilities (if they were calculated):

        ```python
        new_topics, new_probs = topic_model.reduce_topics(docs, topics, probabilities, nr_topics=30)
        ```

        If probabilities were not calculated simply run the function without them:

        ```python
        new_topics, new_probs = topic_model.reduce_topics(docs, topics, nr_topics=30)
        ```
        """
        check_is_fitted(self)
        self.nr_topics = nr_topics
        documents = pd.DataFrame({"Document": docs, "Topic": topics})

        # Reduce number of topics
        documents = self._reduce_topics(documents)
        self.merged_topics = None
        self._map_representative_docs()

        # Extract topics and map probabilities
        new_topics = documents.Topic.to_list()
        new_probabilities = self._map_probabilities(probabilities)

        return new_topics, new_probabilities

    def visualize_topics(self,
                         topics: List[int] = None,
                         top_n_topics: int = None,
                         width: int = 650,
                         height: int = 650) -> go.Figure:
        """ Visualize topics, their sizes, and their corresponding words

        This visualization is highly inspired by LDAvis, a great visualization
        technique typically reserved for LDA.

        Arguments:
            topics: A selection of topics to visualize
            top_n_topics: Only select the top n most frequent topics
            width: The width of the figure.
            height: The height of the figure.

        Usage:

        To visualize the topics simply run:

        ```python
        topic_model.visualize_topics()
        ```

        Or if you want to save the resulting figure:

        ```python
        fig = topic_model.visualize_topics()
        fig.write_html("path/to/file.html")
        ```
        """
        check_is_fitted(self)
        return plotting.visualize_topics(self,
                                         topics=topics,
                                         top_n_topics=top_n_topics,
                                         width=width,
                                         height=height)

    def visualize_documents(self,
                            docs: List[str],
                            topics: List[int] = None,
                            embeddings: np.ndarray = None,
                            reduced_embeddings: np.ndarray = None,
                            sample: float = None,
                            hide_annotations: bool = False,
                            hide_document_hover: bool = False,
                            custom_labels: bool = False,
                            width: int = 1200,
                            height: int = 750) -> go.Figure:
        """ Visualize documents and their topics in 2D

        Arguments:
            topic_model: A fitted BERTopic instance.
            docs: The documents you used when calling either `fit` or `fit_transform`
            topics: A selection of topics to visualize.
                    Not to be confused with the topics that you get from `.fit_transform`.
                    For example, if you want to visualize only topics 1 through 5:
                    `topics = [1, 2, 3, 4, 5]`.
            embeddings: The embeddings of all documents in `docs`.
            reduced_embeddings: The 2D reduced embeddings of all documents in `docs`.
            sample: The percentage of documents in each topic that you would like to keep.
                    Value can be between 0 and 1. Setting this value to, for example,
                    0.1 (10% of documents in each topic) makes it easier to visualize
                    millions of documents as a subset is chosen.
            hide_annotations: Hide the names of the traces on top of each cluster.
            hide_document_hover: Hide the content of the documents when hovering over
                                specific points. Helps to speed up generation of visualization.
            custom_labels: Whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
            width: The width of the figure.
            height: The height of the figure.

        Usage:

        To visualize the topics simply run:

        ```python
        topic_model.visualize_documents(docs)
        ```

        Do note that this re-calculates the embeddings and reduces them to 2D.
        The advised and prefered pipeline for using this function is as follows:

        ```python
        from sklearn.datasets import fetch_20newsgroups
        from sentence_transformers import SentenceTransformer
        from bertopic import BERTopic
        from umap import UMAP

        # Prepare embeddings
        docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = sentence_model.encode(docs, show_progress_bar=False)

        # Train BERTopic
        topic_model = BERTopic().fit(docs, embeddings)

        # Reduce dimensionality of embeddings, this step is optional
        # reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)

        # Run the visualization with the original embeddings
        topic_model.visualize_documents(docs, embeddings=embeddings)

        # Or, if you have reduced the original embeddings already:
        topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings)
        ```

        Or if you want to save the resulting figure:

        ```python
        fig = topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings)
        fig.write_html("path/to/file.html")
        ```

        <iframe src="../../getting_started/visualization/documents.html"
        style="width:1000px; height: 800px; border: 0px;""></iframe>
        """
        check_is_fitted(self)
        return plotting.visualize_documents(self,
                                            docs=docs,
                                            topics=topics,
                                            embeddings=embeddings,
                                            reduced_embeddings=reduced_embeddings,
                                            sample=sample,
                                            hide_annotations=hide_annotations,
                                            hide_document_hover=hide_document_hover,
                                            custom_labels=custom_labels,
                                            width=width,
                                            height=height)

    def visualize_hierarchical_documents(self,
                                         docs: List[str],
                                         hierarchical_topics: pd.DataFrame,
                                         topics: List[int] = None,
                                         embeddings: np.ndarray = None,
                                         reduced_embeddings: np.ndarray = None,
                                         sample: Union[float, int] = None,
                                         hide_annotations: bool = False,
                                         hide_document_hover: bool = True,
                                         nr_levels: int = 10,
                                         custom_labels: bool = False,
                                         width: int = 1200,
                                         height: int = 750) -> go.Figure:
        """ Visualize documents and their topics in 2D at different levels of hierarchy

        Arguments:
            docs: The documents you used when calling either `fit` or `fit_transform`
            hierarchical_topics: A dataframe that contains a hierarchy of topics
                                represented by their parents and their children
            topics: A selection of topics to visualize.
                    Not to be confused with the topics that you get from `.fit_transform`.
                    For example, if you want to visualize only topics 1 through 5:
                    `topics = [1, 2, 3, 4, 5]`.
            embeddings: The embeddings of all documents in `docs`.
            reduced_embeddings: The 2D reduced embeddings of all documents in `docs`.
            sample: The percentage of documents in each topic that you would like to keep.
                    Value can be between 0 and 1. Setting this value to, for example,
                    0.1 (10% of documents in each topic) makes it easier to visualize
                    millions of documents as a subset is chosen.
            hide_annotations: Hide the names of the traces on top of each cluster.
            hide_document_hover: Hide the content of the documents when hovering over
                                specific points. Helps to speed up generation of visualizations.
            nr_levels: The number of levels to be visualized in the hierarchy. First, the distances
                    in `hierarchical_topics.Distance` are split in `nr_levels` lists of distances with
                    equal length. Then, for each list of distances, the merged topics are selected that
                    have a distance less or equal to the maximum distance of the selected list of distances.
                    NOTE: To get all possible merged steps, make sure that `nr_levels` is equal to
                    the length of `hierarchical_topics`.
            custom_labels: Whether to use custom topic labels that were defined using
                           `topic_model.set_topic_labels`.
                           NOTE: Custom labels are only generated for the original
                           un-merged topics.
            width: The width of the figure.
            height: The height of the figure.

        Usage:

        To visualize the topics simply run:

        ```python
        topic_model.visualize_hierarchical_documents(docs, hierarchical_topics)
        ```

        Do note that this re-calculates the embeddings and reduces them to 2D.
        The advised and prefered pipeline for using this function is as follows:

        ```python
        from sklearn.datasets import fetch_20newsgroups
        from sentence_transformers import SentenceTransformer
        from bertopic import BERTopic
        from umap import UMAP

        # Prepare embeddings
        docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = sentence_model.encode(docs, show_progress_bar=False)

        # Train BERTopic and extract hierarchical topics
        topic_model = BERTopic().fit(docs, embeddings)
        hierarchical_topics = topic_model.hierarchical_topics(docs, topics)

        # Reduce dimensionality of embeddings, this step is optional
        # reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)

        # Run the visualization with the original embeddings
        topic_model.visualize_hierarchical_documents(docs, hierarchical_topics, embeddings=embeddings)

        # Or, if you have reduced the original embeddings already:
        topic_model.visualize_hierarchical_documents(docs, hierarchical_topics, reduced_embeddings=reduced_embeddings)
        ```

        Or if you want to save the resulting figure:

        ```python
        fig = topic_model.visualize_hierarchical_documents(docs, hierarchical_topics, reduced_embeddings=reduced_embeddings)
        fig.write_html("path/to/file.html")
        ```

        <iframe src="../../getting_started/visualization/hierarchical_documents.html"
        style="width:1000px; height: 770px; border: 0px;""></iframe>
        """
        check_is_fitted(self)
        return plotting.visualize_hierarchical_documents(self,
                                                         docs=docs,
                                                         hierarchical_topics=hierarchical_topics,
                                                         topics=topics,
                                                         embeddings=embeddings,
                                                         reduced_embeddings=reduced_embeddings,
                                                         sample=sample,
                                                         hide_annotations=hide_annotations,
                                                         hide_document_hover=hide_document_hover,
                                                         nr_levels=nr_levels,
                                                         custom_labels=custom_labels,
                                                         width=width,
                                                         height=height)

    def visualize_term_rank(self,
                            topics: List[int] = None,
                            log_scale: bool = False,
                            custom_labels: bool = False,
                            width: int = 800,
                            height: int = 500) -> go.Figure:
        """ Visualize the ranks of all terms across all topics

        Each topic is represented by a set of words. These words, however,
        do not all equally represent the topic. This visualization shows
        how many words are needed to represent a topic and at which point
        the beneficial effect of adding words starts to decline.

        Arguments:
            topics: A selection of topics to visualize. These will be colored
                    red where all others will be colored black.
            log_scale: Whether to represent the ranking on a log scale
            custom_labels: Whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
            width: The width of the figure.
            height: The height of the figure.

        Returns:
            fig: A plotly figure

        Usage:

        To visualize the ranks of all words across
        all topics simply run:

        ```python
        topic_model.visualize_term_rank()
        ```

        Or if you want to save the resulting figure:

        ```python
        fig = topic_model.visualize_term_rank()
        fig.write_html("path/to/file.html")
        ```

        Reference:

        This visualization was heavily inspired by the
        "Term Probability Decline" visualization found in an
        analysis by the amazing [tmtoolkit](https://tmtoolkit.readthedocs.io/).
        Reference to that specific analysis can be found
        [here](https://wzbsocialsciencecenter.github.io/tm_corona/tm_analysis.html).
        """
        check_is_fitted(self)
        return plotting.visualize_term_rank(self,
                                            topics=topics,
                                            log_scale=log_scale,
                                            custom_labels=custom_labels,
                                            width=width,
                                            height=height)

    def visualize_topics_over_time(self,
                                   topics_over_time: pd.DataFrame,
                                   top_n_topics: int = None,
                                   topics: List[int] = None,
                                   normalize_frequency: bool = False,
                                   custom_labels: bool = False,
                                   width: int = 1250,
                                   height: int = 450) -> go.Figure:
        """ Visualize topics over time

        Arguments:
            topics_over_time: The topics you would like to be visualized with the
                              corresponding topic representation
            top_n_topics: To visualize the most frequent topics instead of all
            topics: Select which topics you would like to be visualized
            normalize_frequency: Whether to normalize each topic's frequency individually
            custom_labels: Whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
            width: The width of the figure.
            height: The height of the figure.

        Returns:
            A plotly.graph_objects.Figure including all traces

        Usage:

        To visualize the topics over time, simply run:

        ```python
        topics_over_time = topic_model.topics_over_time(docs, topics, timestamps)
        topic_model.visualize_topics_over_time(topics_over_time)
        ```

        Or if you want to save the resulting figure:

        ```python
        fig = topic_model.visualize_topics_over_time(topics_over_time)
        fig.write_html("path/to/file.html")
        ```
        """
        check_is_fitted(self)
        return plotting.visualize_topics_over_time(self,
                                                   topics_over_time=topics_over_time,
                                                   top_n_topics=top_n_topics,
                                                   topics=topics,
                                                   normalize_frequency=normalize_frequency,
                                                   custom_labels=custom_labels,
                                                   width=width,
                                                   height=height)

    def visualize_topics_per_class(self,
                                   topics_per_class: pd.DataFrame,
                                   top_n_topics: int = 10,
                                   topics: List[int] = None,
                                   normalize_frequency: bool = False,
                                   custom_labels: bool = False,
                                   width: int = 1250,
                                   height: int = 900) -> go.Figure:
        """ Visualize topics per class

        Arguments:
            topics_per_class: The topics you would like to be visualized with the
                              corresponding topic representation
            top_n_topics: To visualize the most frequent topics instead of all
            topics: Select which topics you would like to be visualized
            normalize_frequency: Whether to normalize each topic's frequency individually
            custom_labels: Whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
            width: The width of the figure.
            height: The height of the figure.

        Returns:
            A plotly.graph_objects.Figure including all traces

        Usage:

        To visualize the topics per class, simply run:

        ```python
        topics_per_class = topic_model.topics_per_class(docs, topics, classes)
        topic_model.visualize_topics_per_class(topics_per_class)
        ```

        Or if you want to save the resulting figure:

        ```python
        fig = topic_model.visualize_topics_per_class(topics_per_class)
        fig.write_html("path/to/file.html")
        ```
        """
        check_is_fitted(self)
        return plotting.visualize_topics_per_class(self,
                                                   topics_per_class=topics_per_class,
                                                   top_n_topics=top_n_topics,
                                                   topics=topics,
                                                   normalize_frequency=normalize_frequency,
                                                   custom_labels=custom_labels,
                                                   width=width,
                                                   height=height)

    def visualize_distribution(self,
                               probabilities: np.ndarray,
                               min_probability: float = 0.015,
                               custom_labels: bool = False,
                               width: int = 800,
                               height: int = 600) -> go.Figure:
        """ Visualize the distribution of topic probabilities

        Arguments:
            probabilities: An array of probability scores
            min_probability: The minimum probability score to visualize.
                             All others are ignored.
            custom_labels: Whether to use custom topic labels that were defined using
                           `topic_model.set_topic_labels`.
            width: The width of the figure.
            height: The height of the figure.

        Usage:

        Make sure to fit the model before and only input the
        probabilities of a single document:

        ```python
        topic_model.visualize_distribution(probabilities[0])
        ```

        Or if you want to save the resulting figure:

        ```python
        fig = topic_model.visualize_distribution(probabilities[0])
        fig.write_html("path/to/file.html")
        ```
        """
        check_is_fitted(self)
        return plotting.visualize_distribution(self,
                                               probabilities=probabilities,
                                               min_probability=min_probability,
                                               custom_labels=custom_labels,
                                               width=width,
                                               height=height)

    def visualize_hierarchy(self,
                            orientation: str = "left",
                            topics: List[int] = None,
                            top_n_topics: int = None,
                            custom_labels: bool = False,
                            width: int = 1000,
                            height: int = 600,
                            hierarchical_topics: pd.DataFrame = None,
                            linkage_function: Callable[[csr_matrix], np.ndarray] = None,
                            distance_function: Callable[[csr_matrix], csr_matrix] = None,
                            color_threshold: int = 1) -> go.Figure:
        """ Visualize a hierarchical structure of the topics

        A ward linkage function is used to perform the
        hierarchical clustering based on the cosine distance
        matrix between topic embeddings.

        Arguments:
            topic_model: A fitted BERTopic instance.
            orientation: The orientation of the figure.
                        Either 'left' or 'bottom'
            topics: A selection of topics to visualize
            top_n_topics: Only select the top n most frequent topics
            custom_labels: Whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
                       NOTE: Custom labels are only generated for the original
                       un-merged topics.
            width: The width of the figure. Only works if orientation is set to 'left'
            height: The height of the figure. Only works if orientation is set to 'bottom'
            hierarchical_topics: A dataframe that contains a hierarchy of topics
                                represented by their parents and their children.
                                NOTE: The hierarchical topic names are only visualized
                                if both `topics` and `top_n_topics` are not set.
            linkage_function: The linkage function to use. Default is:
                            `lambda x: sch.linkage(x, 'ward', optimal_ordering=True)`
                            NOTE: Make sure to use the same `linkage_function` as used
                            in `topic_model.hierarchical_topics`.
            distance_function: The distance function to use on the c-TF-IDF matrix. Default is:
                            `lambda x: 1 - cosine_similarity(x)`
                            NOTE: Make sure to use the same `distance_function` as used
                            in `topic_model.hierarchical_topics`.
            color_threshold: Value at which the separation of clusters will be made which
                         will result in different colors for different clusters.
                         A higher value will typically lead in less colored clusters.

        Returns:
            fig: A plotly figure

        Usage:

        To visualize the hierarchical structure of
        topics simply run:

        ```python
        topic_model.visualize_hierarchy()
        ```

        If you also want the labels visualized of hierarchical topics,
        run the following:

        ```python
        # Extract hierarchical topics and their representations
        hierarchical_topics = topic_model.hierarchical_topics(docs, topics)

        # Visualize these representations
        topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
        ```

        If you want to save the resulting figure:

        ```python
        fig = topic_model.visualize_hierarchy()
        fig.write_html("path/to/file.html")
        ```
        <iframe src="../../getting_started/visualization/hierarchy.html"
        style="width:1000px; height: 680px; border: 0px;""></iframe>
        """
        check_is_fitted(self)
        return plotting.visualize_hierarchy(self,
                                            orientation=orientation,
                                            topics=topics,
                                            top_n_topics=top_n_topics,
                                            custom_labels=custom_labels,
                                            width=width,
                                            height=height,
                                            hierarchical_topics=hierarchical_topics,
                                            linkage_function=linkage_function,
                                            distance_function=distance_function,
                                            color_threshold=color_threshold
                                            )

    def visualize_heatmap(self,
                          topics: List[int] = None,
                          top_n_topics: int = None,
                          n_clusters: int = None,
                          custom_labels: bool = False,
                          width: int = 800,
                          height: int = 800) -> go.Figure:
        """ Visualize a heatmap of the topic's similarity matrix

        Based on the cosine similarity matrix between topic embeddings,
        a heatmap is created showing the similarity between topics.

        Arguments:
            topics: A selection of topics to visualize.
            top_n_topics: Only select the top n most frequent topics.
            n_clusters: Create n clusters and order the similarity
                        matrix by those clusters.
            custom_labels: Whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
            width: The width of the figure.
            height: The height of the figure.

        Returns:
            fig: A plotly figure

        Usage:

        To visualize the similarity matrix of
        topics simply run:

        ```python
        topic_model.visualize_heatmap()
        ```

        Or if you want to save the resulting figure:

        ```python
        fig = topic_model.visualize_heatmap()
        fig.write_html("path/to/file.html")
        ```
        """
        check_is_fitted(self)
        return plotting.visualize_heatmap(self,
                                          topics=topics,
                                          top_n_topics=top_n_topics,
                                          n_clusters=n_clusters,
                                          custom_labels=custom_labels,
                                          width=width,
                                          height=height)

    def visualize_barchart(self,
                           topics: List[int] = None,
                           top_n_topics: int = 8,
                           n_words: int = 5,
                           width: int = 250,
                           height: int = 250) -> go.Figure:
        """ Visualize a barchart of selected topics

        Arguments:
            topics: A selection of topics to visualize.
            top_n_topics: Only select the top n most frequent topics.
            n_words: Number of words to show in a topic
            width: The width of each figure.
            height: The height of each figure.

        Returns:
            fig: A plotly figure

        Usage:

        To visualize the barchart of selected topics
        simply run:

        ```python
        topic_model.visualize_barchart()
        ```

        Or if you want to save the resulting figure:

        ```python
        fig = topic_model.visualize_barchart()
        fig.write_html("path/to/file.html")
        ```
        """
        check_is_fitted(self)
        return plotting.visualize_barchart(self,
                                           topics=topics,
                                           top_n_topics=top_n_topics,
                                           n_words=n_words,
                                           width=width,
                                           height=height)

    def save(self,
             path: str,
             save_embedding_model: bool = True) -> None:
        """ Saves the model to the specified path

        Arguments:
            path: the location and name of the file you want to save
            save_embedding_model: Whether to save the embedding model in this class
                                  as you might have selected a local model or one that
                                  is downloaded automatically from the cloud.

        Usage:

        ```python
        topic_model.save("my_model")
        ```

        or if you do not want the embedding_model to be saved locally:

        ```python
        topic_model.save("my_model", save_embedding_model=False)
        ```
        """
        with open(path, 'wb') as file:

            # This prevents the vectorizer from being too large in size if `min_df` was
            # set to a value higher than 1
            self.vectorizer_model.stop_words_ = None

            if not save_embedding_model:
                embedding_model = self.embedding_model
                self.embedding_model = None
                joblib.dump(self, file)
                self.embedding_model = embedding_model
            else:
                joblib.dump(self, file)

    @classmethod
    def load(cls,
             path: str,
             embedding_model=None):
        """ Loads the model from the specified path

        Arguments:
            path: the location and name of the BERTopic file you want to load
            embedding_model: If the embedding_model was not saved to save space or to load
                             it in from the cloud, you can load it in by specifying it here.

        Usage:

        ```python
        BERTopic.load("my_model")
        ```

        or if you did not save the embedding model:

        ```python
        BERTopic.load("my_model", embedding_model="all-MiniLM-L6-v2")
        ```
        """
        with open(path, 'rb') as file:
            if embedding_model:
                topic_model = joblib.load(file)
                topic_model.embedding_model = select_backend(embedding_model)
            else:
                topic_model = joblib.load(file)
            return topic_model

    def get_params(self, deep: bool = False) -> Mapping[str, Any]:
        """ Get parameters for this estimator.

        Adapted from:
            https://github.com/scikit-learn/scikit-learn/blob/b3ea3ed6a/sklearn/base.py#L178

        Arguments:
            deep: bool, default=True
                  If True, will return the parameters for this estimator and
                  contained subobjects that are estimators.

        Returns:
            out: Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def _extract_embeddings(self,
                            documents: Union[List[str], str],
                            method: str = "document",
                            verbose: bool = None) -> np.ndarray:
        """ Extract sentence/document embeddings through pre-trained embeddings
        For an overview of pre-trained models: https://www.sbert.net/docs/pretrained_models.html

        Arguments:
            documents: Dataframe with documents and their corresponding IDs
            method: Whether to extract document or word-embeddings, options are "document" and "word"
            verbose: Whether to show a progressbar demonstrating the time to extract embeddings

        Returns:
            embeddings: The extracted embeddings.
        """
        if isinstance(documents, str):
            documents = [documents]

        if method == "word":
            embeddings = self.embedding_model.embed_words(documents, verbose)
        elif method == "document":
            embeddings = self.embedding_model.embed_documents(documents, verbose)
        else:
            raise ValueError("Wrong method for extracting document/word embeddings. "
                             "Either choose 'word' or 'document' as the method. ")

        return embeddings

    def _map_predictions(self, predictions: List[int]) -> List[int]:
        """ Map predictions to the correct topics if topics were reduced """
        mappings = self.topic_mapper.get_mappings(original_topics=True)
        mapped_predictions = [mappings[prediction]
                              if prediction in mappings
                              else -1
                              for prediction in predictions]
        return mapped_predictions

    def _reduce_dimensionality(self,
                               embeddings: Union[np.ndarray, csr_matrix],
                               y: Union[List[int], np.ndarray] = None) -> np.ndarray:
        """ Reduce dimensionality of embeddings using UMAP and train a UMAP model

        Arguments:
            embeddings: The extracted embeddings using the sentence transformer module.
            y: The target class for (semi)-supervised dimensionality reduction

        Returns:
            umap_embeddings: The reduced embeddings
        """
        try:
            self.umap_model.fit(embeddings, y=y)
        except TypeError:
            logger.info("The dimensionality reduction algorithm did not contain the `y` parameter and"
                        " therefore the `y` parameter was not used")
            self.umap_model.fit(embeddings)

        umap_embeddings = self.umap_model.transform(embeddings)
        logger.info("Reduced dimensionality")
        return np.nan_to_num(umap_embeddings)

    def _cluster_embeddings(self,
                            umap_embeddings: np.ndarray,
                            documents: pd.DataFrame) -> Tuple[pd.DataFrame,
                                                              np.ndarray]:
        """ Cluster UMAP embeddings with HDBSCAN

        Arguments:
            umap_embeddings: The reduced sentence embeddings with UMAP
            documents: Dataframe with documents and their corresponding IDs

        Returns:
            documents: Updated dataframe with documents and their corresponding IDs
                       and newly added Topics
            probabilities: The distribution of probabilities
        """
        # Fit and extract labels
        self.hdbscan_model.fit(umap_embeddings)
        documents['Topic'] = self.hdbscan_model.labels_

        # Some algorithms have outlier labels (-1) that can be tricky to work
        # with if you are slicing data based on that labels. Therefore, we
        # track if there are outlier labels and act accordingly when slicing.
        if -1 in set(self.hdbscan_model.labels_):
            self._outliers = 1
        else:
            self._outliers = 0

        self._update_topic_size(documents)

        # Save representative docs and calculate probabilities if it is a HDBSCAN model
        if isinstance(self.hdbscan_model, hdbscan.HDBSCAN):
            probabilities = self.hdbscan_model.probabilities_
            self._save_representative_docs(documents)
            if self.calculate_probabilities:
                probabilities = hdbscan.all_points_membership_vectors(self.hdbscan_model)
        else:
            probabilities = None

        self.topic_mapper = TopicMapper(self.hdbscan_model)
        logger.info("Clustered reduced embeddings")
        return documents, probabilities

    def _guided_topic_modeling(self, embeddings: np.ndarray) -> Tuple[List[int], np.array]:
        """ Apply Guided Topic Modeling

        We transform the seeded topics to embeddings using the
        same embedder as used for generating document embeddings.

        Then, we apply cosine similarity between the embeddings
        and set labels for documents that are more similar to
        one of the topics, then the average document.

        If a document is more similar to the average document
        than any of the topics, it gets the -1 label and is
        thereby not included in UMAP.

        Arguments:
            embeddings: The document embeddings

        Returns
            y: The labels for each seeded topic
            embeddings: Updated embeddings
        """
        # Create embeddings from the seeded topics
        seed_topic_list = [" ".join(seed_topic) for seed_topic in self.seed_topic_list]
        seed_topic_embeddings = self._extract_embeddings(seed_topic_list, verbose=self.verbose)
        seed_topic_embeddings = np.vstack([seed_topic_embeddings, embeddings.mean(axis=0)])

        # Label documents that are most similar to one of the seeded topics
        sim_matrix = cosine_similarity(embeddings, seed_topic_embeddings)
        y = [np.argmax(sim_matrix[index]) for index in range(sim_matrix.shape[0])]
        y = [val if val != len(seed_topic_list) else -1 for val in y]

        # Average the document embeddings related to the seeded topics with the
        # embedding of the seeded topic to force the documents in a cluster
        for seed_topic in range(len(seed_topic_list)):
            indices = [index for index, topic in enumerate(y) if topic == seed_topic]
            embeddings[indices] = np.average([embeddings[indices], seed_topic_embeddings[seed_topic]], weights=[3, 1])
        return y, embeddings

    def _extract_topics(self, documents: pd.DataFrame):
        """ Extract topics from the clusters using a class-based TF-IDF

        Arguments:
            documents: Dataframe with documents and their corresponding IDs

        Returns:
            c_tf_idf: The resulting matrix giving a value (importance score) for each word per topic
        """
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        self.c_tf_idf, words = self._c_tf_idf(documents_per_topic)
        self.topics = self._extract_words_per_topic(words)
        self._create_topic_vectors()
        self.topic_names = {key: f"{key}_" + "_".join([word[0] for word in values[:4]])
                            for key, values in
                            self.topics.items()}

    def _save_representative_docs(self, documents: pd.DataFrame):
        """ Save the most representative docs (3) per topic

        The most representative docs are extracted by taking
        the exemplars from the HDBSCAN-generated clusters.

        Full instructions can be found here:
            https://hdbscan.readthedocs.io/en/latest/soft_clustering_explanation.html

        Arguments:
            documents: Dataframe with documents and their corresponding IDs
        """
        # Prepare the condensed tree and luf clusters beneath a given cluster
        condensed_tree = self.hdbscan_model.condensed_tree_
        raw_tree = condensed_tree._raw_tree
        clusters = sorted(condensed_tree._select_clusters())
        cluster_tree = raw_tree[raw_tree['child_size'] > 1]

        #  Find the points with maximum lambda value in each leaf
        representative_docs = {}
        for topic in documents['Topic'].unique():
            if topic != -1:
                leaves = hdbscan.plots._recurse_leaf_dfs(cluster_tree, clusters[topic])

                result = np.array([])
                for leaf in leaves:
                    max_lambda = raw_tree['lambda_val'][raw_tree['parent'] == leaf].max()
                    points = raw_tree['child'][(raw_tree['parent'] == leaf) & (raw_tree['lambda_val'] == max_lambda)]
                    result = np.hstack((result, points))

                representative_docs[topic] = list(np.random.choice(result, 3, replace=False).astype(int))

        # Convert indices to documents
        self.representative_docs = {topic: [documents.iloc[doc_id].Document for doc_id in doc_ids]
                                    for topic, doc_ids in
                                    representative_docs.items()}

    def _map_representative_docs(self, original_topics: bool = False):
        """ Map the representative docs per topic to the correct topics

        If topics were reduced, remove documents from topics that were
        merged into larger topics as we assume that the documents from
        larger topics are better representative of the entire merged
        topic.

        Arguments:
            original_topics: Whether we want to map from the
                             original topics to the most recent topics
                             or from the second-most recent topics.
        """
        if isinstance(self.hdbscan_model, hdbscan.HDBSCAN):
            mappings = self.topic_mapper.get_mappings(original_topics)
            representative_docs = self.representative_docs.copy()

            # Update the representative documents
            updated_representative_docs = {mappings[old_topic]: []
                                           for old_topic, _ in representative_docs.items()}
            for old_topic, docs in representative_docs.items():
                new_topic = mappings[old_topic]
                updated_representative_docs[new_topic].extend(docs)

            self.representative_docs = updated_representative_docs

    def _create_topic_vectors(self):
        """ Creates embeddings per topics based on their topic representation

        We start by creating embeddings out of the topic representation. This
        results in a number of embeddings per topic. Then, we take the weighted
        average of embeddings in a topic by their c-TF-IDF score. This will put
        more emphasis to words that represent a topic best.

        Only allow topic vectors to be created if there are no custom embeddings and therefore
        a sentence-transformer model to be used or there are custom embeddings but it is allowed
        to use a different multi-lingual sentence-transformer model
        """
        if self.embedding_model is not None:
            topic_list = list(self.topics.keys())
            topic_list.sort()
            n = self.top_n_words

            # Extract embeddings for all words in all topics
            topic_words = [self.get_topic(topic) for topic in topic_list]
            topic_words = [word[0] for topic in topic_words for word in topic]
            embeddings = self._extract_embeddings(topic_words,
                                                  method="word",
                                                  verbose=False)

            # Take the weighted average of word embeddings in a topic based on their c-TF-IDF value
            # The embeddings var is a single numpy matrix and therefore slicing is necessary to
            # access the words per topic
            topic_embeddings = []
            for i, topic in enumerate(topic_list):
                word_importance = [val[1] for val in self.get_topic(topic)]
                if sum(word_importance) == 0:
                    word_importance = [1 for _ in range(len(self.get_topic(topic)))]
                topic_embedding = np.average(embeddings[i * n: n + (i * n)], weights=word_importance, axis=0)
                topic_embeddings.append(topic_embedding)

            self.topic_embeddings = topic_embeddings

    def _c_tf_idf(self, documents_per_topic: pd.DataFrame, fit: bool = True) -> Tuple[csr_matrix, List[str]]:
        """ Calculate a class-based TF-IDF where m is the number of total documents.

        Arguments:
            documents_per_topic: The joined documents per topic such that each topic has a single
                                 string made out of multiple documents
            m: The total number of documents (unjoined)
            fit: Whether to fit a new vectorizer or use the fitted self.vectorizer_model

        Returns:
            tf_idf: The resulting matrix giving a value (importance score) for each word per topic
            words: The names of the words to which values were given
        """
        documents = self._preprocess_text(documents_per_topic.Document.values)

        if fit:
            self.vectorizer_model.fit(documents)

        words = self.vectorizer_model.get_feature_names()
        X = self.vectorizer_model.transform(documents)

        if self.seed_topic_list:
            seed_topic_list = [seed for seeds in self.seed_topic_list for seed in seeds]
            multiplier = np.array([1.2 if word in seed_topic_list else 1 for word in words])
        else:
            multiplier = None

        if fit:
            self.transformer = ClassTFIDF().fit(X, multiplier=multiplier)

        c_tf_idf = self.transformer.transform(X)

        return c_tf_idf, words

    def _update_topic_size(self, documents: pd.DataFrame):
        """ Calculate the topic sizes

        Arguments:
            documents: Updated dataframe with documents and their corresponding IDs and newly added Topics
        """
        sizes = documents.groupby(['Topic']).count().sort_values("Document", ascending=False).reset_index()
        self.topic_sizes = dict(zip(sizes.Topic, sizes.Document))

    def _extract_words_per_topic(self,
                                 words: List[str],
                                 c_tf_idf: csr_matrix = None,
                                 labels: List[int] = None) -> Mapping[str,
                                                                      List[Tuple[str, float]]]:
        """ Based on tf_idf scores per topic, extract the top n words per topic

        If the top words per topic need to be extracted, then only the `words` parameter
        needs to be passed. If the top words per topic in a specific timestamp, then it
        is important to pass the timestamp-based c-TF-IDF matrix and its corresponding
        labels.

        Arguments:
            words: List of all words (sorted according to tf_idf matrix position)
            c_tf_idf: A c-TF-IDF matrix from which to calculate the top words
            labels: A list of topic labels

        Returns:
            topics: The top words per topic
        """
        if c_tf_idf is None:
            c_tf_idf = self.c_tf_idf

        if labels is None:
            labels = sorted(list(self.topic_sizes.keys()))

        # Get the top 30 indices and values per row in a sparse c-TF-IDF matrix
        indices = self._top_n_idx_sparse(c_tf_idf, 30)
        scores = self._top_n_values_sparse(c_tf_idf, indices)
        sorted_indices = np.argsort(scores, 1)
        indices = np.take_along_axis(indices, sorted_indices, axis=1)
        scores = np.take_along_axis(scores, sorted_indices, axis=1)

        # Get top 30 words per topic based on c-TF-IDF score
        topics = {label: [(words[word_index], score)
                          if word_index is not None and score > 0
                          else ("", 0.00001)
                          for word_index, score in zip(indices[index][::-1], scores[index][::-1])
                          ]
                  for index, label in enumerate(labels)}

        # Extract word embeddings for the top 30 words per topic and compare it
        # with the topic embedding to keep only the words most similar to the topic embedding
        if self.diversity is not None:
            if self.embedding_model is not None:

                for topic, topic_words in topics.items():
                    words = [word[0] for word in topic_words]
                    word_embeddings = self._extract_embeddings(words,
                                                               method="word",
                                                               verbose=False)
                    topic_embedding = self._extract_embeddings(" ".join(words),
                                                               method="word",
                                                               verbose=False).reshape(1, -1)
                    topic_words = mmr(topic_embedding, word_embeddings, words,
                                      top_n=self.top_n_words, diversity=self.diversity)
                    topics[topic] = [(word, value) for word, value in topics[topic] if word in topic_words]
        topics = {label: values[:self.top_n_words] for label, values in topics.items()}

        return topics

    def _reduce_topics(self, documents: pd.DataFrame) -> pd.DataFrame:
        """ Reduce topics to self.nr_topics

        Arguments:
            documents: Dataframe with documents and their corresponding IDs and Topics

        Returns:
            documents: Updated dataframe with documents and the reduced number of Topics
        """
        initial_nr_topics = len(self.get_topics())

        if isinstance(self.nr_topics, int):
            if self.nr_topics < initial_nr_topics:
                documents = self._reduce_to_n_topics(documents)
        elif isinstance(self.nr_topics, str):
            documents = self._auto_reduce_topics(documents)
        else:
            raise ValueError("nr_topics needs to be an int or 'auto'! ")

        logger.info(f"Reduced number of topics from {initial_nr_topics} to {len(self.get_topic_freq())}")
        return documents

    def _reduce_to_n_topics(self, documents: pd.DataFrame) -> pd.DataFrame:
        """ Reduce topics to self.nr_topics

        Arguments:
            documents: Dataframe with documents and their corresponding IDs and Topics

        Returns:
            documents: Updated dataframe with documents and the reduced number of Topics
        """
        # Track which topics where originally merged
        if not self.merged_topics:
            self.merged_topics = []

        # Create topic similarity matrix
        similarities = cosine_similarity(self.c_tf_idf)
        np.fill_diagonal(similarities, 0)

        # Find most similar topic to least common topic
        topics = documents.Topic.tolist().copy()
        mapped_topics = {}
        while len(self.get_topic_freq()) > self.nr_topics + self._outliers:
            topic_to_merge = self.get_topic_freq().iloc[-1].Topic
            topic_to_merge_into = np.argmax(similarities[topic_to_merge + self._outliers]) - self._outliers
            similarities[:, topic_to_merge + self._outliers] = -self._outliers
            self.merged_topics.append(topic_to_merge)

            # Update Topic labels
            documents.loc[documents.Topic == topic_to_merge, "Topic"] = topic_to_merge_into
            mapped_topics[topic_to_merge] = topic_to_merge_into
            self._update_topic_size(documents)

        # Map topics
        mapped_topics = {from_topic: to_topic for from_topic, to_topic in zip(topics, documents.Topic.tolist())}
        self.topic_mapper.add_mappings(mapped_topics)

        # Update representations
        documents = self._sort_mappings_by_frequency(documents)
        self._extract_topics(documents)
        self._update_topic_size(documents)
        return documents

    def _auto_reduce_topics(self, documents: pd.DataFrame) -> pd.DataFrame:
        """ Reduce the number of topics automatically using HDBSCAN

        Arguments:
            documents: Dataframe with documents and their corresponding IDs and Topics

        Returns:
            documents: Updated dataframe with documents and the reduced number of Topics
        """
        topics = documents.Topic.tolist().copy()
        unique_topics = sorted(list(documents.Topic.unique()))[self._outliers:]
        max_topic = unique_topics[-1]

        # Find similar topics
        if self.topic_embeddings is not None:
            embeddings = np.array(self.topic_embeddings)
        else:
            embeddings = self.c_tf_idf.toarray()
        norm_data = normalize(embeddings, norm='l2')
        predictions = hdbscan.HDBSCAN(min_cluster_size=2,
                                      metric='euclidean',
                                      cluster_selection_method='eom',
                                      prediction_data=True).fit_predict(norm_data[self._outliers:])

        # Map similar topics
        mapped_topics = {unique_topics[index]: prediction + max_topic
                         for index, prediction in enumerate(predictions)
                         if prediction != -1}
        documents.Topic = documents.Topic.map(mapped_topics).fillna(documents.Topic).astype(int)
        mapped_topics = {from_topic: to_topic for from_topic, to_topic in zip(topics, documents.Topic.tolist())}

        # Update documents and topics
        self.topic_mapper.add_mappings(mapped_topics)
        documents = self._sort_mappings_by_frequency(documents)
        self._extract_topics(documents)
        self._update_topic_size(documents)
        return documents

    def _sort_mappings_by_frequency(self, documents: pd.DataFrame) -> pd.DataFrame:
        """ Reorder mappings by their frequency.

        For example, if topic 88 was mapped to topic
        5 and topic 5 turns out to be the largest topic,
        then topic 5 will be topic 0. The second largest,
        will be topic 1, etc.

        If there are no mappings since no reduction of topics
        took place, then the topics will simply be ordered
        by their frequency and will get the topic ids based
        on that order.

        This means that -1 will remain the outlier class, and
        that the rest of the topics will be in descending order
        of ids and frequency.

        Arguments:
            documents: Dataframe with documents and their corresponding IDs and Topics

        Returns:
            documents: Updated dataframe with documents and the mapped
                       and re-ordered topic ids
        """
        self._update_topic_size(documents)

        # Map topics based on frequency
        df = pd.DataFrame(self.topic_sizes.items(), columns=["Old_Topic", "Size"]).sort_values("Size", ascending=False)
        df = df[df.Old_Topic != -1]
        sorted_topics = {**{-1: -1}, **dict(zip(df.Old_Topic, range(len(df))))}
        self.topic_mapper.add_mappings(sorted_topics)

        # Map documents
        documents.Topic = documents.Topic.map(sorted_topics).fillna(documents.Topic).astype(int)
        self._update_topic_size(documents)
        return documents

    def _map_probabilities(self,
                           probabilities: Union[np.ndarray, None],
                           original_topics: bool = False) -> Union[np.ndarray, None]:
        """ Map the probabilities to the reduced topics.
        This is achieved by adding the probabilities together
        of all topics that were mapped to the same topic. Then,
        the topics that were mapped from were set to 0 as they
        were reduced.

        Arguments:
            probabilities: An array containing probabilities
            original_topics: Whether we want to map from the
                             original topics to the most recent topics
                             or from the second-most recent topics.

        Returns:
            mapped_probabilities: Updated probabilities
        """
        mappings = self.topic_mapper.get_mappings(original_topics)

        # Map array of probabilities (probability for assigned topic per document)
        if probabilities is not None:
            if len(probabilities.shape) == 2 and self.get_topic(-1):
                mapped_probabilities = np.zeros((probabilities.shape[0],
                                                 len(set(mappings.values())) - 1))
                for from_topic, to_topic in mappings.items():
                    if to_topic != -1 and from_topic != -1:
                        mapped_probabilities[:, to_topic] += probabilities[:, from_topic]

                return mapped_probabilities

        return probabilities

    def _preprocess_text(self, documents: np.ndarray) -> List[str]:
        """ Basic preprocessing of text

        Steps:
            * Lower text
            * Replace \n and \t with whitespace
            * Only keep alpha-numerical characters
        """
        cleaned_documents = [doc.lower() for doc in documents]
        cleaned_documents = [doc.replace("\n", " ") for doc in cleaned_documents]
        cleaned_documents = [doc.replace("\t", " ") for doc in cleaned_documents]
        if self.language == "english":
            cleaned_documents = [re.sub(r'[^A-Za-z0-9 ]+', '', doc) for doc in cleaned_documents]
        cleaned_documents = [doc if doc != "" else "emptydoc" for doc in cleaned_documents]
        return cleaned_documents

    @staticmethod
    def _top_n_idx_sparse(matrix: csr_matrix, n: int) -> np.ndarray:
        """ Return indices of top n values in each row of a sparse matrix

        Retrieved from:
            https://stackoverflow.com/questions/49207275/finding-the-top-n-values-in-a-row-of-a-scipy-sparse-matrix

        Arguments:
            matrix: The sparse matrix from which to get the top n indices per row
            n: The number of highest values to extract from each row

        Returns:
            indices: The top n indices per row
        """
        indices = []
        for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
            n_row_pick = min(n, ri - le)
            values = matrix.indices[le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]]
            values = [values[index] if len(values) >= index + 1 else None for index in range(n)]
            indices.append(values)
        return np.array(indices)

    @staticmethod
    def _top_n_values_sparse(matrix: csr_matrix, indices: np.ndarray) -> np.ndarray:
        """ Return the top n values for each row in a sparse matrix

        Arguments:
            matrix: The sparse matrix from which to get the top n indices per row
            indices: The top n indices per row

        Returns:
            top_values: The top n scores per row
        """
        top_values = []
        for row, values in enumerate(indices):
            scores = np.array([matrix[row, value] if value is not None else 0 for value in values])
            top_values.append(scores)
        return np.array(top_values)

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator

        Adapted from:
            https://github.com/scikit-learn/scikit-learn/blob/b3ea3ed6a/sklearn/base.py#L178
        """
        init_signature = inspect.signature(cls.__init__)
        parameters = sorted([p.name for p in init_signature.parameters.values()
                             if p.name != 'self' and p.kind != p.VAR_KEYWORD])
        return parameters

    def __str__(self):
        """Get a string representation of the current object.

        Returns:
            str: Human readable representation of the most important model parameters.
                 The parameters that represent models are ignored due to their
        """
        parameters = ""
        for parameter, value in self.get_params().items():
            value = str(value)
            if "(" in value and value[0] != "(":
                value = value.split("(")[0] + "(...)"
            parameters += f"{parameter}={value}, "

        return f"BERTopic({parameters[:-2]})"


class TopicMapper:
    """ Keep track of Topic Mappings

    The number of topics can be reduced
    by merging them together. This mapping
    needs to be tracked in BERTopic as new
    predictions need to be mapped to the new
    topics.

    These mappings are tracked in the `self.mappings`
    variable where each set of topic are stacked horizontally.
    For example, the most recent topics can be found in the
    last column. To get a mapping, simply take the two columns
    of topics.

    In other words, it is represented as graph:
    Topic 1 --> Topic 11 --> Topic 4 --> etc.

    """
    def __init__(self, hdbscan_model: hdbscan.HDBSCAN):
        """ Initalization of Topic Mapper

        Arguments:
            hdbscan_model: The trained HDBSCAN-model which
                           is used to extract the topics from
        """
        self.base_topics = np.array(sorted(list(set(hdbscan_model.labels_))))
        topics = self.base_topics.copy().reshape(-1, 1)
        self.mappings = np.hstack([topics.copy(), topics.copy()]).tolist()

    def get_mappings(self, original_topics: bool = True) -> Mapping[int, int]:
        """ Get mappings from either the original topics or
        the second-most recent topics to the current topics

        Arguments:
            original_topics: Whether we want to map from the
                             original topics to the most recent topics
                             or from the second-most recent topics.

        Returns:
            mappings: The mappings from old topics to new topics

        Usage:

        To get mappings, simply call:
        ```python
        mapper = TopicMapper(hdbscan_model)
        mappings = mapper.get_mappings(original_topics=False)
        ```
        """
        if original_topics:
            mappings = np.array(self.mappings)[:, [0, -1]]
            mappings = dict(zip(mappings[:, 0], mappings[:, 1]))
        else:
            mappings = np.array(self.mappings)[:, [-3, -1]]
            mappings = dict(zip(mappings[:, 0], mappings[:, 1]))
        return mappings

    def add_mappings(self, mappings: Mapping[int, int]):
        """ Add new column(s) of topic mappings

        Arguments:
            mappings: The mappings to add
        """
        for topics in self.mappings:
            topic = topics[-1]
            if topic in mappings:
                topics.append(mappings[topic])
            else:
                topics.append(-1)
