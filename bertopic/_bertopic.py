import yaml
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    yaml._warnings_enabled["YAMLLoadWarning"] = False
except (KeyError, AttributeError, TypeError) as e:
    pass

import re
import math
import joblib
import inspect
import numpy as np
import pandas as pd
from tqdm import tqdm
from packaging import version
from scipy.sparse import csr_matrix
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import squareform
from typing import List, Tuple, Union, Mapping, Any, Callable, Iterable

# Models
import hdbscan
from umap import UMAP
from sklearn.preprocessing import normalize
from sklearn import __version__ as sklearn_version
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# BERTopic
from bertopic import plotting
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.backend import BaseEmbedder
from bertopic.backend._utils import select_backend
from bertopic.representation import BaseRepresentation
from bertopic.cluster._utils import hdbscan_delegator, is_supported_hdbscan
from bertopic._utils import MyLogger, check_documents_type, check_embeddings_shape, check_is_fitted

# Visualization
import plotly.graph_objects as go

logger = MyLogger("WARNING")


class BERTopic:
    """BERTopic is a topic modeling technique that leverages BERT embeddings and
    c-TF-IDF to create dense clusters allowing for easily interpretable topics
    whilst keeping important words in the topic descriptions.

    The default embedding model is `all-MiniLM-L6-v2` when selecting `language="english"`
    and `paraphrase-multilingual-MiniLM-L12-v2` when selecting `language="multilingual"`.

    Attributes:
        topics_ (List[int]) : The topics that are generated for each document after training or updating
                              the topic model. The most recent topics are tracked.
        probabilities_ (List[float]): The probability of the assigned topic per document. These are
                                      only calculated if a HDBSCAN model is used for the clustering step.
                                      When `calculate_probabilities=True`, then it is the probabilities
                                      of all topics per document.
        topic_sizes_ (Mapping[int, int]) : The size of each topic
        topic_mapper_ (TopicMapper) : A class for tracking topics and their mappings anytime they are
                                      merged, reduced, added, or removed.
        topic_representations_ (Mapping[int, Tuple[int, float]]) : The top n terms per topic and their respective
                                                                   c-TF-IDF values.
        c_tf_idf_ (csr_matrix) : The topic-term matrix as calculated through c-TF-IDF. To access its respective
                                 words, run `.vectorizer_model.get_feature_names()`  or
                                 `.vectorizer_model.get_feature_names_out()`
        topic_labels_ (Mapping[int, str]) : The default labels for each topic.
        custom_labels_ (List[str]) : Custom labels for each topic.
        topic_embeddings_ (np.ndarray) : The embeddings for each topic. It is calculated by taking the
                                         weighted average of word embeddings in a topic based on their c-TF-IDF values.
        representative_docs_ (Mapping[int, str]) : The representative documents for each topic.

    Examples:

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
                 seed_topic_list: List[List[str]] = None,
                 embedding_model=None,
                 umap_model: UMAP = None,
                 hdbscan_model: hdbscan.HDBSCAN = None,
                 vectorizer_model: CountVectorizer = None,
                 ctfidf_model: TfidfTransformer = None,
                 representation_model: BaseRepresentation = None,
                 verbose: bool = False,
                 ):
        """BERTopic initialization

        Arguments:
            language: The main language used in your documents. The default sentence-transformers
                      model for "english" is `all-MiniLM-L6-v2`. For a full overview of
                      supported languages see bertopic.backend.languages. Select
                      "multilingual" to load in the `paraphrase-multilingual-MiniLM-L12-v2`
                      sentence-tranformers model that supports 50+ languages.
                      NOTE: This is not used if `embedding_model` is used. 
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
                            NOTE: This param will not be used if you are not using HDBSCAN.
            nr_topics: Specifying the number of topics will reduce the initial
                       number of topics to the value specified. This reduction can take
                       a while as each reduction in topics (-1) activates a c-TF-IDF
                       calculation. If this is set to None, no reduction is applied. Use
                       "auto" to automatically reduce topics using HDBSCAN.
            low_memory: Sets UMAP low memory to True to make sure less memory is used.
                        NOTE: This is only used in UMAP. For example, if you use PCA instead of UMAP
                        this parameter will not be used.
            calculate_probabilities: Calculate the probabilities of all topics
                                     per document instead of the probability of the assigned
                                     topic per document. This could slow down the extraction
                                     of topics if you have many documents (> 100_000). 
                                     NOTE: If false you cannot use the corresponding
                                     visualization method `visualize_probabilities`.
                                     NOTE: This is an approximation of topic probabilities
                                     as used in HDBSCAN and not an exact representation.
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
            vectorizer_model: Pass in a custom `CountVectorizer` instead of the default model.
            ctfidf_model: Pass in a custom ClassTfidfTransformer instead of the default model.
            representation_model: Pass in a model that fine-tunes the topic representations 
                                  calculated through c-TF-IDF. Models from `bertopic.representation`
                                  are supported.
        """
        # Topic-based parameters
        if top_n_words > 100:
            warnings.warn("Note that extracting more than 100 words from a sparse "
                          "can slow down computation quite a bit.")

        self.top_n_words = top_n_words
        self.min_topic_size = min_topic_size
        self.nr_topics = nr_topics
        self.low_memory = low_memory
        self.calculate_probabilities = calculate_probabilities
        self.verbose = verbose
        self.seed_topic_list = seed_topic_list

        # Embedding model
        self.language = language if not embedding_model else None
        self.embedding_model = embedding_model

        # Vectorizer
        self.n_gram_range = n_gram_range
        self.vectorizer_model = vectorizer_model or CountVectorizer(ngram_range=self.n_gram_range)
        self.ctfidf_model = ctfidf_model or ClassTfidfTransformer()

        # Representation model
        self.representation_model = representation_model

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

        # Public attributes
        self.topics_ = None
        self.probabilities_ = None
        self.topic_sizes_ = None
        self.topic_mapper_ = None
        self.topic_representations_ = None
        self.topic_embeddings_ = None
        self.topic_labels_ = None
        self.custom_labels_ = None
        self.representative_docs_ = {}
        self.c_tf_idf_ = None

        # Private attributes for internal tracking purposes
        self._outliers = 1
        self._merged_topics = None

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

        Examples:

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

        Examples:

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
        documents, probabilities = self._cluster_embeddings(umap_embeddings, documents, y=y)

        # Sort and Map Topic IDs by their frequency
        if not self.nr_topics:
            documents = self._sort_mappings_by_frequency(documents)

        # Extract topics by calculating c-TF-IDF
        self._extract_topics(documents)

        # Reduce topics
        if self.nr_topics:
            documents = self._reduce_topics(documents)

        # Save the top 3 most representative documents per topic
        self._save_representative_docs(documents)

        # Resulting output
        self.probabilities_ = self._map_probabilities(probabilities, original_topics=True)
        predictions = documents.Topic.to_list()

        return predictions, self.probabilities_

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

        Examples:

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

        # Extract predictions and probabilities if it is a HDBSCAN-like model
        if is_supported_hdbscan(self.hdbscan_model):
            predictions, probabilities = hdbscan_delegator(self.hdbscan_model, "approximate_predict", umap_embeddings)

            # Calculate probabilities
            if self.calculate_probabilities and isinstance(self.hdbscan_model, hdbscan.HDBSCAN):
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

    def partial_fit(self,
                    documents: List[str],
                    embeddings: np.ndarray = None,
                    y: Union[List[int], np.ndarray] = None):
        """ Fit BERTopic on a subset of the data and perform online learning
        with batch-like data.

        Online topic modeling in BERTopic is performed by using dimensionality
        reduction and cluster algorithms that support a `partial_fit` method
        in order to incrementally train the topic model.

        Likewise, the `bertopic.vectorizers.OnlineCountVectorizer` is used
        to dynamically update its vocabulary when presented with new data.
        It has several parameters for modeling decay and updating the
        representations.

        In other words, although the main algorithm stays the same, the training
        procedure now works as follows:

        For each subset of the data:

        1. Generate embeddings with a pre-traing language model
        2. Incrementally update the dimensionality reduction algorithm with `partial_fit`
        3. Incrementally update the cluster algorithm with `partial_fit`
        4. Incrementally update the OnlineCountVectorizer and apply some form of decay

        Note that it is advised to use `partial_fit` with batches and
        not single documents for the best performance.

        Arguments:
            documents: A list of documents to fit on
            embeddings: Pre-trained document embeddings. These can be used
                        instead of the sentence-transformer model
            y: The target class for (semi)-supervised modeling. Use -1 if no class for a
               specific instance is specified.

        Examples:

        ```python
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.decomposition import IncrementalPCA
        from bertopic.vectorizers import OnlineCountVectorizer
        from bertopic import BERTopic

        # Prepare documents
        docs = fetch_20newsgroups(subset=subset,  remove=('headers', 'footers', 'quotes'))["data"]

        # Prepare sub-models that support online learning
        umap_model = IncrementalPCA(n_components=5)
        cluster_model = MiniBatchKMeans(n_clusters=50, random_state=0)
        vectorizer_model = OnlineCountVectorizer(stop_words="english", decay=.01)

        topic_model = BERTopic(umap_model=umap_model,
                               hdbscan_model=cluster_model,
                               vectorizer_model=vectorizer_model)

        # Incrementally fit the topic model by training on 1000 documents at a time
        for index in range(0, len(docs), 1000):
            topic_model.partial_fit(docs[index: index+1000])
        ```
        """
        # Checks
        check_embeddings_shape(embeddings, documents)
        if not hasattr(self.hdbscan_model, "partial_fit"):
            raise ValueError("In order to use `.partial_fit`, the cluster model should have "
                             "a `.partial_fit` function.")

        # Prepare documents
        if isinstance(documents, str):
            documents = [documents]
        documents = pd.DataFrame({"Document": documents,
                                  "ID": range(len(documents)),
                                  "Topic": None})

        # Extract embeddings
        if embeddings is None:
            if self.topic_representations_ is None:
                self.embedding_model = select_backend(self.embedding_model,
                                                      language=self.language)
            embeddings = self._extract_embeddings(documents.Document,
                                                  method="document",
                                                  verbose=self.verbose)
        else:
            if self.embedding_model is not None and self.topic_representations_ is None:
                self.embedding_model = select_backend(self.embedding_model,
                                                      language=self.language)

        # Reduce dimensionality
        if self.seed_topic_list is not None and self.embedding_model is not None:
            y, embeddings = self._guided_topic_modeling(embeddings)
        umap_embeddings = self._reduce_dimensionality(embeddings, y, partial_fit=True)

        # Cluster reduced embeddings
        documents, self.probabilities_ = self._cluster_embeddings(umap_embeddings, documents, partial_fit=True)
        topics = documents.Topic.to_list()

        # Map and find new topics
        if not self.topic_mapper_:
            self.topic_mapper_ = TopicMapper(topics)
        mappings = self.topic_mapper_.get_mappings()
        new_topics = set(topics).difference(set(mappings.keys()))
        new_topic_ids = {topic: max(mappings.values()) + index + 1 for index, topic in enumerate(new_topics)}
        self.topic_mapper_.add_new_topics(new_topic_ids)
        updated_mappings = self.topic_mapper_.get_mappings()
        updated_topics = [updated_mappings[topic] for topic in topics]
        documents["Topic"] = updated_topics

        # Add missing topics (topics that were originally created but are now missing)
        if self.topic_representations_:
            missing_topics = set(self.topic_representations_.keys()).difference(set(updated_topics))
            for missing_topic in missing_topics:
                documents.loc[len(documents), :] = [" ", len(documents), missing_topic]
        else:
            missing_topics = {}

        # Prepare documents
        documents_per_topic = documents.sort_values("Topic").groupby(['Topic'], as_index=False)
        updated_topics = documents_per_topic.first().Topic.astype(int)
        documents_per_topic = documents_per_topic.agg({'Document': ' '.join})

        # Update topic representations
        self.c_tf_idf_, updated_words = self._c_tf_idf(documents_per_topic, partial_fit=True)
        self.topic_representations_ = self._extract_words_per_topic(updated_words, documents, self.c_tf_idf_)
        self._create_topic_vectors()
        self.topic_labels_ = {key: f"{key}_" + "_".join([word[0] for word in values[:4]])
                              for key, values in self.topic_representations_.items()}

        # Update topic sizes
        if len(missing_topics) > 0:
            documents = documents.iloc[:-len(missing_topics)]

        if self.topic_sizes_ is None:
            self._update_topic_size(documents)
        else:
            sizes = documents.groupby(['Topic'], as_index=False).count()
            for _, row in sizes.iterrows():
                topic = int(row.Topic)
                if self.topic_sizes_.get(topic) is not None and topic not in missing_topics:
                    self.topic_sizes_[topic] += int(row.Document)
                elif self.topic_sizes_.get(topic) is None:
                    self.topic_sizes_[topic] = int(row.Document)
            self.topics_ = documents.Topic.astype(int).tolist()

        return self

    def topics_over_time(self,
                         docs: List[str],
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
            evolution_tuning: Fine-tune each topic representation at timestamp *t* by averaging its
                              c-TF-IDF matrix with the c-TF-IDF matrix at timestamp *t-1*. This creates
                              evolutionary topic representations.
            global_tuning: Fine-tune each topic representation at timestamp *t* by averaging its c-TF-IDF matrix
                       with the global c-TF-IDF matrix. Turn this off if you want to prevent words in
                       topic representations that could not be found in the documents at timestamp *t*.

        Returns:
            topics_over_time: A dataframe that contains the topic, words, and frequency of topic
                              at timestamp *t*.

        Examples:

        The timestamps variable represent the timestamp of each document. If you have over
        100 unique timestamps, it is advised to bin the timestamps as shown below:

        ```python
        from bertopic import BERTopic
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(docs)
        topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=20)
        ```
        """
        check_is_fitted(self)
        check_documents_type(docs)
        documents = pd.DataFrame({"Document": docs, "Topic": self.topics_, "Timestamps": timestamps})
        global_c_tf_idf = normalize(self.c_tf_idf_, axis=1, norm='l1', copy=False)

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
            words_per_topic = self._extract_words_per_topic(words, selection, c_tf_idf)
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
            classes: The class of each document. This can be either a list of strings or ints.
            global_tuning: Fine-tune each topic representation for class c t by averaging its c-TF-IDF matrix
                           with the global c-TF-IDF matrix. Turn this off if you want to prevent words in
                           topic representations that could not be found in the documents for class c.

        Returns:
            topics_per_class: A dataframe that contains the topic, words, and frequency of topics
                              for each class.

        Examples:

        ```python
        from bertopic import BERTopic
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(docs)
        topics_per_class = topic_model.topics_per_class(docs, classes)
        ```
        """
        documents = pd.DataFrame({"Document": docs, "Topic": self.topics_, "Class": classes})
        global_c_tf_idf = normalize(self.c_tf_idf_, axis=1, norm='l1', copy=False)

        # For each unique timestamp, create topic representations
        topics_per_class = []
        for _, class_ in tqdm(enumerate(set(classes)), disable=not self.verbose):

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
            words_per_topic = self._extract_words_per_topic(words, selection, c_tf_idf)
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
            linkage_function: The linkage function to use. Default is:
                            `lambda x: sch.linkage(x, 'ward', optimal_ordering=True)`
            distance_function: The distance function to use on the c-TF-IDF matrix. Default is:
                                `lambda x: 1 - cosine_similarity(x)`

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
        if distance_function is None:
            distance_function = lambda x: 1 - cosine_similarity(x)

        if linkage_function is None:
            linkage_function = lambda x: sch.linkage(x, 'ward', optimal_ordering=True)

        # Calculate distance
        embeddings = self.c_tf_idf_[self._outliers:]
        X = distance_function(embeddings)

        # Make sure it is the 1-D condensed distance matrix with zeros on the diagonal
        np.fill_diagonal(X, 0)
        X = squareform(X)

        # Use the 1-D condensed distance matrix as an input instead of the raw distance matrix
        Z = linkage_function(X)

        # Calculate basic bag-of-words to be iteratively merged later
        documents = pd.DataFrame({"Document": docs,
                                  "ID": range(len(docs)),
                                  "Topic": self.topics_})
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
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
            c_tf_idf = self.ctfidf_model.transform(grouped)
            selection = documents.loc[documents.Topic.isin(clustered_topics), :]
            selection.Topic = 0
            words_per_topic = self._extract_words_per_topic(words, selection, c_tf_idf)

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

    def approximate_distribution(self,
                                 documents: Union[str, List[str]],
                                 window: int = 4,
                                 stride: int = 1,
                                 min_similarity: float = 0.1,
                                 batch_size: int = 1000,
                                 padding: bool = False,
                                 use_embedding_model: bool = False,
                                 calculate_tokens: bool = False,
                                 separator: str = " ") -> Tuple[np.ndarray,
                                                                Union[List[np.ndarray], None]]:
        """ A post-hoc approximation of topic distributions across documents.

        In order to perform this approximation, each document is split into tokens
        according to the provided tokenizer in the `CountVectorizer`. Then, a
        sliding window is applied on each document creating subsets of the document.
        For example, with a window size of 3 and stride of 1, the sentence:

        `Solving the right problem is difficult.`

        can be split up into `solving the right`, `the right problem`, `right problem is`,
        and `problem is difficult`. These are called tokensets. For each of these
        tokensets, we calculate their c-TF-IDF representation and find out
        how similar they are to the previously generated topics. Then, the
        similarities to the topics for each tokenset are summed in order to
        create a topic distribution for the entire document.

        We can also dive into this a bit deeper by then splitting these tokensets
        up into individual tokens and calculate how much a word, in a specific sentence,
        contributes to the topics found in that document. This can be enabled by
        setting `calculate_tokens=True` which can be used for visualization purposes
        in `topic_model.visualize_approximate_distribution`.

        The main output, `topic_distributions`, can also be used directly in
        `.visualize_distribution(topic_distributions[index])` by simply selecting
        a single distribution.

        Arguments:
            documents: A single document or a list of documents for which we
                    approximate their topic distributions
            window: Size of the moving window which indicates the number of
                    tokens being considered.
            stride: How far the window should move at each step.
            min_similarity: The minimum similarity of a document's tokenset
                            with respect to the topics.
            batch_size: The number of documents to process at a time. If None,
                        then all documents are processed at once.
                        NOTE: With a large number of documents, it is not
                        advised to process all documents at once.
            padding: Whether to pad the beginning and ending of a document with
                     empty tokens.
            use_embedding_model: Whether to use the topic model's embedding
                                model to calculate the similarity between
                                tokensets and topics instead of using c-TF-IDF.
            calculate_tokens: Calculate the similarity of tokens with all topics.
                            NOTE: This is computation-wise more expensive and
                            can require more memory. Using this over batches of
                            documents might be preferred.
            separator: The separator used to merge tokens into tokensets.

        Returns:
            topic_distributions: A `n` x `m` matrix containing the topic distributions
                                for all input documents with `n` being the documents
                                and `m` the topics.
            topic_token_distributions: A list of `t` x `m` arrays with `t` being the
                                    number of tokens for the respective document
                                    and `m` the topics.

        Examples:

        After fitting the model, the topic distributions can be calculated regardless
        of the clustering model and regardless of whether the documents were previously
        seen or not:

        ```python
        topic_distr, _ = topic_model.approximate_distribution(docs)
        ```

        As a result, the topic distributions are calculated in `topic_distr` for the
        entire document based on token set with a specific window size and stride.

        If you want to calculate the topic distributions on a token-level:

        ```python
        topic_distr, topic_token_distr = topic_model.approximate_distribution(docs, calculate_tokens=True)
        ```

        The `topic_token_distr` then contains, for each token, the best fitting topics.
        As with `topic_distr`, it can contain multiple topics for a single token.
        """
        if isinstance(documents, str):
            documents = [documents]

        if batch_size is None:
            batch_size = len(documents)
            batches = 1
        else:
            batches = math.ceil(len(documents)/batch_size)

        topic_distributions = []
        topic_token_distributions = []

        for i in tqdm(range(batches), disable=not self.verbose):
            doc_set = documents[i*batch_size: (i+1) * batch_size]

            # Extract tokens
            analyzer = self.vectorizer_model.build_tokenizer()
            tokens = [analyzer(document) for document in doc_set]

            # Extract token sets
            all_sentences = []
            all_indices = [0]
            all_token_sets_ids = []

            for tokenset in tokens:
                if len(tokenset) < window:
                    token_sets = [tokenset]
                    token_sets_ids = [list(range(len(tokenset)))]
                else:

                    # Extract tokensets using window and stride parameters
                    stride_indices = list(range(len(tokenset)))[::stride]
                    token_sets = []
                    token_sets_ids = []
                    for stride_index in stride_indices:
                        selected_tokens = tokenset[stride_index: stride_index+window]

                        if padding or len(selected_tokens) == window:
                            token_sets.append(selected_tokens)
                            token_sets_ids.append(list(range(stride_index, stride_index+len(selected_tokens))))

                    # Add empty tokens at the beginning and end of a document
                    if padding:
                        padded = []
                        padded_ids = []
                        t = math.ceil(window / stride) - 1
                        for i in range(math.ceil(window / stride) - 1):
                            padded.append(tokenset[:window - ((t-i) * stride)])
                            padded_ids.append(list(range(0, window - ((t-i) * stride))))

                        token_sets = padded + token_sets
                        token_sets_ids = padded_ids + token_sets_ids

                # Join the tokens
                sentences = [separator.join(token) for token in token_sets]
                all_sentences.extend(sentences)
                all_token_sets_ids.extend(token_sets_ids)
                all_indices.append(all_indices[-1] + len(sentences))

            # Calculate similarity between embeddings of token sets and the topics
            if use_embedding_model:
                embeddings = self._extract_embeddings(all_sentences, method="document", verbose=True)
                similarity = cosine_similarity(embeddings, self.topic_embeddings_[self._outliers:])

            # Calculate similarity between c-TF-IDF of token sets and the topics
            else:
                bow_doc = self.vectorizer_model.transform(all_sentences)
                c_tf_idf_doc = self.ctfidf_model.transform(bow_doc)
                similarity = cosine_similarity(c_tf_idf_doc, self.c_tf_idf_[self._outliers:])

            # Only keep similarities that exceed the minimum
            similarity[similarity < min_similarity] = 0

            # Aggregate results on an individual token level
            if calculate_tokens:
                topic_distribution = []
                topic_token_distribution = []
                for index, token in enumerate(tokens):
                    start = all_indices[index]
                    end = all_indices[index+1]

                    if start == end:
                        end = end + 1

                    # Assign topics to individual tokens
                    token_id = [i for i in range(len(token))]
                    token_val = {index: [] for index in token_id}
                    for sim, token_set in zip(similarity[start:end], all_token_sets_ids[start:end]):
                        for token in token_set:
                            if token in token_val:
                                token_val[token].append(sim)

                    matrix = []
                    for _, value in token_val.items():
                        matrix.append(np.add.reduce(value))

                    # Take empty documents into account
                    matrix = np.array(matrix)
                    if len(matrix.shape) == 1:
                        matrix = np.zeros((1, len(self.topic_labels_) - self._outliers))

                    topic_token_distribution.append(np.array(matrix))
                    topic_distribution.append(np.add.reduce(matrix))

                topic_distribution = normalize(topic_distribution, norm='l1', axis=1)

            # Aggregate on a tokenset level indicated by the window and stride
            else:
                topic_distribution = []
                for index in range(len(all_indices)-1):
                    start = all_indices[index]
                    end = all_indices[index+1]

                    if start == end:
                        end = end + 1
                    group = similarity[start:end].sum(axis=0)
                    topic_distribution.append(group)
                topic_distribution = normalize(np.array(topic_distribution), norm='l1', axis=1)
                topic_token_distribution = None

            # Combine results
            topic_distributions.append(topic_distribution)
            if topic_token_distribution is None:
                topic_token_distributions = None
            else:
                topic_token_distributions.extend(topic_token_distribution)

        topic_distributions = np.vstack(topic_distributions)

        return topic_distributions, topic_token_distributions

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

        Examples:

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

        topic_list = list(self.topic_representations_.keys())
        topic_list.sort()

        # Extract search_term embeddings and compare with topic embeddings
        search_embedding = self._extract_embeddings([search_term],
                                                    method="word",
                                                    verbose=False).flatten()
        sims = cosine_similarity(search_embedding.reshape(1, -1), self.topic_embeddings_).flatten()

        # Extract topics most similar to search_term
        ids = np.argsort(sims)[-top_n:]
        similarity = [sims[i] for i in ids][::-1]
        similar_topics = [topic_list[index] for index in ids][::-1]

        return similar_topics, similarity

    def update_topics(self,
                      docs: List[str],
                      topics: List[int] = None,
                      top_n_words: int = 10,
                      n_gram_range: Tuple[int, int] = None,
                      vectorizer_model: CountVectorizer = None,
                      ctfidf_model: ClassTfidfTransformer = None,
                      representation_model: BaseRepresentation = None):
        """ Updates the topic representation by recalculating c-TF-IDF with the new
        parameters as defined in this function.

        When you have trained a model and viewed the topics and the words that represent them,
        you might not be satisfied with the representation. Perhaps you forgot to remove
        stop_words or you want to try out a different n_gram_range. This function allows you
        to update the topic representation after they have been formed.

        Arguments:
            docs: The documents you used when calling either `fit` or `fit_transform`
            topics: A list of topics where each topic is related to a document in `docs`.
                    Use this variable to change or map the topics.
                    NOTE: Using a custom list of topic assignments may lead to errors if
                          topic reduction techniques are used afterwards. Make sure that
                          manually assigning topics is the last step in the pipeline
            top_n_words: The number of words per topic to extract. Setting this
                         too high can negatively impact topic embeddings as topics
                         are typically best represented by at most 10 words.
            n_gram_range: The n-gram range for the CountVectorizer.
            vectorizer_model: Pass in your own CountVectorizer from scikit-learn
            ctfidf_model: Pass in your own c-TF-IDF model to update the representations
            representation_model: Pass in a model that fine-tunes the topic representations 
                        calculated through c-TF-IDF. Models from `bertopic.representation`
                        are supported.

        Examples:

        In order to update the topic representation, you will need to first fit the topic
        model and extract topics from them. Based on these, you can update the representation:

        ```python
        topic_model.update_topics(docs, n_gram_range=(2, 3))
        ```

        You can also use a custom vectorizer to update the representation:

        ```python
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
        topic_model.update_topics(docs, vectorizer_model=vectorizer_model)
        ```

        You can also use this function to change or map the topics to something else.
        You can update them as follows:

        ```python
        topic_model.update_topics(docs, my_updated_topics)
        ```
        """
        check_is_fitted(self)
        if not n_gram_range:
            n_gram_range = self.n_gram_range

        if top_n_words > 100:
            warnings.warn("Note that extracting more than 100 words from a sparse "
                          "can slow down computation quite a bit.")
        self.top_n_words = top_n_words
        self.vectorizer_model = vectorizer_model or CountVectorizer(ngram_range=n_gram_range)
        self.ctfidf_model = ctfidf_model or ClassTfidfTransformer()
        self.representation_model = representation_model

        if topics is None:
            topics = self.topics_
        else:
            warnings.warn("Using a custom list of topic assignments may lead to errors if "
                          "topic reduction techniques are used afterwards. Make sure that "
                          "manually assigning topics is the last step in the pipeline.")
        # Extract words
        documents = pd.DataFrame({"Document": docs, "Topic": topics})
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        self.c_tf_idf_, words = self._c_tf_idf(documents_per_topic)
        self.topic_representations_ = self._extract_words_per_topic(words, documents)
        self._create_topic_vectors()
        self.topic_labels_ = {key: f"{key}_" + "_".join([word[0] for word in values[:4]])
                              for key, values in
                              self.topic_representations_.items()}
        self._update_topic_size(documents)

    def get_topics(self) -> Mapping[str, Tuple[str, float]]:
        """ Return topics with top n words and their c-TF-IDF score

        Returns:
            self.topic_representations_: The top n words per topic and the corresponding c-TF-IDF score

        Examples:

        ```python
        all_topics = topic_model.get_topics()
        ```
        """
        check_is_fitted(self)
        return self.topic_representations_

    def get_topic(self, topic: int) -> Union[Mapping[str, Tuple[str, float]], bool]:
        """ Return top n words for a specific topic and their c-TF-IDF scores

        Arguments:
            topic: A specific topic for which you want its representation

        Returns:
            The top n words for a specific word and its respective c-TF-IDF scores

        Examples:

        ```python
        topic = topic_model.get_topic(12)
        ```
        """
        check_is_fitted(self)
        if topic in self.topic_representations_:
            return self.topic_representations_[topic]
        else:
            return False

    def get_topic_info(self, topic: int = None) -> pd.DataFrame:
        """ Get information about each topic including its ID, frequency, and name.

        Arguments:
            topic: A specific topic for which you want the frequency

        Returns:
            info: The information relating to either a single topic or all topics

        Examples:

        ```python
        info_df = topic_model.get_topic_info()
        ```
        """
        check_is_fitted(self)

        info = pd.DataFrame(self.topic_sizes_.items(), columns=["Topic", "Count"]).sort_values("Topic")
        info["Name"] = info.Topic.map(self.topic_labels_)

        if self.custom_labels_ is not None:
            if len(self.custom_labels_) == len(info):
                labels = {topic - self._outliers: label for topic, label in enumerate(self.custom_labels_)}
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

        Examples:

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
            return self.topic_sizes_[topic]
        else:
            return pd.DataFrame(self.topic_sizes_.items(), columns=['Topic', 'Count']).sort_values("Count",
                                                                                                   ascending=False)

    def get_document_info(self,
                          docs: List[str],
                          df: pd.DataFrame = None,
                          metadata: Mapping[str, Any] = None) -> pd.DataFrame:
        """ Get information about the documents on which the topic was trained
        including the documents themselves, their respective topics, the name
        of each topic, the top n words of each topic, whether it is a
        representative document, and probability of the clustering if the cluster
        model supports it.

        There are also options to include other meta data, such as the topic
        distributions or the x and y coordinates of the reduced embeddings.

        Arguments:
            docs: The documents on which the topic model was trained.
            df: A dataframe containing the metadata and the documents on which
                the topic model was originally trained on.
            metadata: A dictionary with meta data for each document in the form
                    of column name (key) and the respective values (value).

        Returns:
            document_info: A dataframe with several statistics regarding
                        the documents on which the topic model was trained.

        Usage:

        To get the document info, you will only need to pass the documents on which
        the topic model was trained:

        ```python
        document_info = topic_model.get_document_info(docs)
        ```

        There are additionally options to include meta data, such as the topic
        distributions. Moreover, we can pass the original dataframe that contains
        the documents and extend it with the information retrieved from BERTopic:

        ```python
        from sklearn.datasets import fetch_20newsgroups

        # The original data in a dataframe format to include the target variable
        data= fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))
        df = pd.DataFrame({"Document": data['data'], "Class": data['target']})

        # Add information about the percentage of the document that relates to the topic
        topic_distr, _ = topic_model.approximate_distribution(docs, batch_size=1000)
        distributions = [distr[topic] if topic != -1 else 0 for topic, distr in zip(topics, topic_distr)]

        # Create our documents dataframe using the original dataframe and meta data about
        # the topic distributions
        document_info = topic_model.get_document_info(docs, df=df,
                                                      metadata={"Topic_distribution": distributions})
        ```
        """
        if df is not None:
            document_info = df.copy()
            document_info["Document"] = docs
            document_info["Topic"] = self.topics_
        else:
            document_info = pd.DataFrame({"Document": docs, "Topic": self.topics_})

        # Add topic info through `.get_topic_info()`
        topic_info = self.get_topic_info().drop("Count", axis=1)
        document_info = pd.merge(document_info, topic_info, on="Topic", how="left")

        # Add top n words
        top_n_words = {topic: " - ".join(list(zip(*self.get_topic(topic)))[0]) for topic in set(self.topics_)}
        document_info["Top_n_words"] = document_info.Topic.map(top_n_words)

        # Add flat probabilities
        if self.probabilities_ is not None:
            if len(self.probabilities_.shape) == 1:
                document_info["Probability"] = self.probabilities_
            else:
                document_info["Probability"] = [max(probs) if topic != -1 else 1-sum(probs)
                                                for topic, probs in zip(self.topics_, self.probabilities_)]

        # Add representative document labels
        repr_docs = [repr_doc for repr_docs in self.representative_docs_.values() for repr_doc in repr_docs]
        document_info["Representative_document"] = False
        document_info.loc[document_info.Document.isin(repr_docs), "Representative_document"] = True

        # Add custom meta data provided by the user
        if metadata is not None:
            for column, values in metadata.items():
                document_info[column] = values
        return document_info

    def get_representative_docs(self, topic: int = None) -> List[str]:
        """ Extract the best representing documents per topic.

        NOTE:
            This does not extract all documents per topic as all documents
            are not saved within BERTopic. To get all documents, please
            run the following:

            ```python
            # When you used `.fit_transform`:
            df = pd.DataFrame({"Document": docs, "Topic": topic})

            # When you used `.fit`:
            df = pd.DataFrame({"Document": docs, "Topic": topic_model.topics_})
            ```

        Arguments:
            topic: A specific topic for which you want
                   the representative documents

        Returns:
            Representative documents of the chosen topic

        Examples:

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
            if self.representative_docs_.get(topic):
                return self.representative_docs_[topic]
            else:
                return None
        else:
            return self.representative_docs_

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

        Examples:

        ```python
        # Train model
        from bertopic import BERTopic
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(docs)
        hierarchical_topics = topic_model.hierarchical_topics(docs)

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

        Examples:

        First, we define our topic labels with `.get_topic_labels` in which
        we can customize our topic labels:

        ```python
        topic_labels = topic_model.get_topic_labels(nr_words=2,
                                                    topic_prefix=True,
                                                    word_length=10,
                                                    separator=", ")
        ```

        Then, we pass these `topic_labels` to our topic model which
        can be accessed at any time with `.custom_labels_`:

        ```python
        topic_model.set_topic_labels(topic_labels)
        topic_model.custom_labels_
        ```

        You might want to change only a few topic labels instead of all of them.
        To do so, you can pass a dictionary where the keys are the topic IDs and
        its keys the topic labels:

        ```python
        topic_model.set_topic_labels({0: "Space", 1: "Sports", 2: "Medicine"})
        topic_model.custom_labels_
        ```
        """
        unique_topics = sorted(set(self.topics_))

        if isinstance(topic_labels, dict):
            if self.custom_labels_ is not None:
                original_labels = {topic: label for topic, label in zip(unique_topics, self.custom_labels_)}
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

        self.custom_labels_ = custom_labels

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

        Examples:

        To create our custom topic labels, usage is rather straightforward:

        ```python
        topic_labels = topic_model.get_topic_labels(nr_words=2, separator=", ")
        ```
        """
        unique_topics = sorted(set(self.topics_))

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
                     topics_to_merge: List[Union[Iterable[int], int]]) -> None:
        """
        Arguments:
            docs: The documents you used when calling either `fit` or `fit_transform`
            topics_to_merge: Either a list of topics or a list of list of topics
                            to merge. For example:
                                [1, 2, 3] will merge topics 1, 2 and 3
                                [[1, 2], [3, 4]] will merge topics 1 and 2, and
                                separately merge topics 3 and 4.

        Examples:

        If you want to merge topics 1, 2, and 3:

        ```python
        topics_to_merge = [1, 2, 3]
        topic_model.merge_topics(docs, topics_to_merge)
        ```

        or if you want to merge topics 1 and 2, and separately
        merge topics 3 and 4:

        ```python
        topics_to_merge = [[1, 2]
                            [3, 4]]
        topic_model.merge_topics(docs, topics_to_merge)
        ```
        """
        check_is_fitted(self)
        documents = pd.DataFrame({"Document": docs, "Topic": self.topics_})

        mapping = {topic: topic for topic in set(self.topics_)}
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
        self.topic_mapper_.add_mappings(mapping)
        documents = self._sort_mappings_by_frequency(documents)
        self._extract_topics(documents)
        self._update_topic_size(documents)
        self._save_representative_docs(documents)
        self.probabilities_ = self._map_probabilities(self.probabilities_)

    def reduce_topics(self,
                      docs: List[str],
                      nr_topics: Union[int, str] = 20) -> None:
        """ Reduce the number of topics to a fixed number of topics
        or automatically.

        If nr_topics is a integer, then the number of topics is reduced
        to nr_topics using `AgglomerativeClustering` on the cosine distance matrix
        of the topic embeddings.

        If nr_topics is `"auto"`, then HDBSCAN is used to automatically
        reduce the number of topics by running it on the topic embeddings.

        The topics, their sizes, and representations are updated.

        Arguments:
            docs: The docs you used when calling either `fit` or `fit_transform`
            nr_topics: The number of topics you want reduced to

        Updates:
            topics_ : Assigns topics to their merged representations.
            probabilities_ : Assigns probabilities to their merged representations.

        Examples:

        You can further reduce the topics by passing the documents with its
        topics and probabilities (if they were calculated):

        ```python
        topic_model.reduce_topics(docs, nr_topics=30)
        ```

        You can then access the updated topics and probabilities with:

        ```python
        topics = topic_model.topics_
        probabilities = topic_model.probabilities_
        ```
        """
        check_is_fitted(self)

        self.nr_topics = nr_topics
        documents = pd.DataFrame({"Document": docs, "Topic": self.topics_})

        # Reduce number of topics
        documents = self._reduce_topics(documents)
        self._merged_topics = None
        self._save_representative_docs(documents)
        self.probabilities_ = self._map_probabilities(self.probabilities_)

        return self

    def reduce_outliers(self,
                        documents: List[str],
                        topics: List[int],
                        strategy: str = "distributions",
                        probabilities: np.ndarray = None,
                        threshold: int = 0,
                        embeddings: np.ndarray = None,
                        distributions_params: Mapping[str, Any] = {}) -> List[int]:
        """ Reduce outliers by merging them with their nearest topic according
        to one of several strategies.

        When using HDBSCAN, DBSCAN, or OPTICS, a number of outlier documents might be created
        that do not fall within any of the created topics. These are labeled as -1.
        This function allows the user to match outlier documents with their nearest topic
        using one of the following strategies using the `strategy` parameter:
            * "probabilities"
                This uses the soft-clustering as performed by HDBSCAN to find the
                best matching topic for each outlier document. To use this, make
                sure to calculate the `probabilities` beforehand by instantiating
                BERTopic with `calculate_probabilities=True`.
            * "distributions"
                Use the topic distributions, as calculated with `.approximate_distribution`
                to find the most frequent topic in each outlier document. You can use the
                `distributions_params` variable to tweak the parameters of
                `.approximate_distribution`.
            * "c-tf-idf"
                Calculate the c-TF-IDF representation for each outlier document and
                find the best matching c-TF-IDF topic representation using
                cosine similarity.
            * "embeddings"
                Using the embeddings of each outlier documents, find the best
                matching topic embedding using cosine similarity.

        Arguments:
            documents: A list of documents for which we reduce or remove the outliers.
            topics: The topics that correspond to the documents
            strategy: The strategy used for reducing outliers.
                    Options:
                        * "probabilities"
                            This uses the soft-clustering as performed by HDBSCAN
                            to find the best matching topic for each outlier document.

                        * "distributions"
                            Use the topic distributions, as calculated with `.approximate_distribution`
                            to find the most frequent topic in each outlier document.

                        * "c-tf-idf"
                            Calculate the c-TF-IDF representation for outlier documents and
                            find the best matching c-TF-IDF topic representation.

                        * "embeddings"
                            Calculate the embeddings for outlier documents and
                            find the best matching topic embedding.
            threshold: The threshold for assigning topics to outlier documents. This value
                    represents the minimum probability when `strategy="probabilities"`.
                    For all other strategies, it represents the minimum similarity.
            embeddings: The pre-computed embeddings to be used when `strategy="embeddings"`.
                        If this is None, then it will compute the embeddings for the outlier documents.
            distributions_params: The parameters used in `.approximate_distribution` when using
                                  the strategy `"distributions"`.

        Returns:
            new_topics: The updated topics

        Usage:

        The default settings uses the `"distributions"` strategy:

        ```python
        new_topics = topic_model.reduce_outliers(docs, topics)
        ```

        When you use the `"probabilities"` strategy, make sure to also pass the probabilities
        as generated through HDBSCAN:

        ```python
        from bertopic import BERTopic
        topic_model = BERTopic(calculate_probabilities=True)
        topics, probs = topic_model.fit_transform(docs)

        new_topics = topic_model.reduce_outliers(docs, topics, probabilities=probs, strategy="probabilities")
        ```
        """

        # Check correct use of parameters
        if strategy.lower() == "probabilities" and probabilities is None:
            raise ValueError("Make sure to pass in `probabilities` in order to use the probabilities strategy")

        # Reduce outliers by extracting most likely topics through the topic-term probability matrix
        if strategy.lower() == "probabilities":
            new_topics = [np.argmax(prob) if max(prob) >= threshold and topic == -1 else topic
                          for topic, prob in zip(topics, probabilities)]

        # Reduce outliers by extracting most frequent topics through calculating of Topic Distributions
        elif strategy.lower() == "distributions":
            outlier_ids = [index for index, topic in enumerate(topics) if topic == -1]
            outlier_docs = [documents[index] for index in outlier_ids]
            topic_distr, _ = self.approximate_distribution(outlier_docs, min_similarity=threshold, **distributions_params)
            outlier_topics = iter([np.argmax(prob) if sum(prob) > 0 else -1 for prob in topic_distr])
            new_topics = [topic if topic != -1 else next(outlier_topics) for topic in topics]

        # Reduce outliers by finding the most similar c-TF-IDF representations
        elif strategy.lower() == "c-tf-idf":
            outlier_ids = [index for index, topic in enumerate(topics) if topic == -1]
            outlier_docs = [documents[index] for index in outlier_ids]

            # Calculate c-TF-IDF of outlier documents with all topics
            bow_doc = self.vectorizer_model.transform(outlier_docs)
            c_tf_idf_doc = self.ctfidf_model.transform(bow_doc)
            similarity = cosine_similarity(c_tf_idf_doc, self.c_tf_idf_[self._outliers:])

            # Update topics
            similarity[similarity < threshold] = 0
            outlier_topics = iter([np.argmax(sim) if sum(sim) > 0 else -1 for sim in similarity])
            new_topics = [topic if topic != -1 else next(outlier_topics) for topic in topics]

        # Reduce outliers by finding the most similar topic embeddings
        elif strategy.lower() == "embeddings":
            if self.embedding_model is None:
                raise ValueError("To use this strategy, you will need to pass a model to `embedding_model`"
                                  "when instantiating BERTopic.")
            outlier_ids = [index for index, topic in enumerate(topics) if topic == -1]
            outlier_docs = [documents[index] for index in outlier_ids]

            # Extract or calculate embeddings for outlier documents
            if embeddings is not None:
                outlier_embeddings = np.array([embeddings[index] for index in outlier_ids])
            else:
                outlier_embeddings = self.embedding_model.embed_documents(outlier_docs)
            similarity = cosine_similarity(outlier_embeddings, self.topic_embeddings_[self._outliers:])

            # Update topics
            similarity[similarity < threshold] = 0
            outlier_topics = iter([np.argmax(sim) if sum(sim) > 0 else -1 for sim in similarity])
            new_topics = [topic if topic != -1 else next(outlier_topics) for topic in topics]

        return new_topics

    def visualize_topics(self,
                         topics: List[int] = None,
                         top_n_topics: int = None,
                         custom_labels: bool = False,
                         title: str = "<b>Intertopic Distance Map</b>",
                         width: int = 650,
                         height: int = 650) -> go.Figure:
        """ Visualize topics, their sizes, and their corresponding words

        This visualization is highly inspired by LDAvis, a great visualization
        technique typically reserved for LDA.

        Arguments:
            topics: A selection of topics to visualize
            top_n_topics: Only select the top n most frequent topics
            custom_labels: Whether to use custom topic labels that were defined using 
                       `topic_model.set_topic_labels`.
            title: Title of the plot.
            width: The width of the figure.
            height: The height of the figure.

        Examples:

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
                                         custom_labels=custom_labels,
                                         title=title,
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
                            title: str = "<b>Documents and Topics</b>",
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
            title: Title of the plot.
            width: The width of the figure.
            height: The height of the figure.

        Examples:

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

        <iframe src="../getting_started/visualization/documents.html"
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
                                            title=title,
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
                                         title: str = "<b>Hierarchical Documents and Topics</b>",
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
            title: Title of the plot.
            width: The width of the figure.
            height: The height of the figure.

        Examples:

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
        hierarchical_topics = topic_model.hierarchical_topics(docs)

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

        <iframe src="../getting_started/visualization/hierarchical_documents.html"
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
                                                         title=title,
                                                         width=width,
                                                         height=height)

    def visualize_term_rank(self,
                            topics: List[int] = None,
                            log_scale: bool = False,
                            custom_labels: bool = False,
                            title: str = "<b>Term score decline per Topic</b>",
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
            title: Title of the plot.
            width: The width of the figure.
            height: The height of the figure.

        Returns:
            fig: A plotly figure

        Examples:

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
                                            title=title,
                                            width=width,
                                            height=height)

    def visualize_topics_over_time(self,
                                   topics_over_time: pd.DataFrame,
                                   top_n_topics: int = None,
                                   topics: List[int] = None,
                                   normalize_frequency: bool = False,
                                   custom_labels: bool = False,
                                   title: str = "<b>Topics over Time</b>",
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
            title: Title of the plot.
            width: The width of the figure.
            height: The height of the figure.

        Returns:
            A plotly.graph_objects.Figure including all traces

        Examples:

        To visualize the topics over time, simply run:

        ```python
        topics_over_time = topic_model.topics_over_time(docs, timestamps)
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
                                                   title=title,
                                                   width=width,
                                                   height=height)

    def visualize_topics_per_class(self,
                                   topics_per_class: pd.DataFrame,
                                   top_n_topics: int = 10,
                                   topics: List[int] = None,
                                   normalize_frequency: bool = False,
                                   custom_labels: bool = False,
                                   title: str = "<b>Topics per Class</b>",
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
            title: Title of the plot.
            width: The width of the figure.
            height: The height of the figure.

        Returns:
            A plotly.graph_objects.Figure including all traces

        Examples:

        To visualize the topics per class, simply run:

        ```python
        topics_per_class = topic_model.topics_per_class(docs, classes)
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
                                                   title=title,
                                                   width=width,
                                                   height=height)

    def visualize_distribution(self,
                               probabilities: np.ndarray,
                               min_probability: float = 0.015,
                               custom_labels: bool = False,
                               title: str = "<b>Topic Probability Distribution</b>",
                               width: int = 800,
                               height: int = 600) -> go.Figure:
        """ Visualize the distribution of topic probabilities

        Arguments:
            probabilities: An array of probability scores
            min_probability: The minimum probability score to visualize.
                             All others are ignored.
            custom_labels: Whether to use custom topic labels that were defined using
                           `topic_model.set_topic_labels`.
            title: Title of the plot.
            width: The width of the figure.
            height: The height of the figure.

        Examples:

        Make sure to fit the model before and only input the
        probabilities of a single document:

        ```python
        topic_model.visualize_distribution(topic_model.probabilities_[0])
        ```

        Or if you want to save the resulting figure:

        ```python
        fig = topic_model.visualize_distribution(topic_model.probabilities_[0])
        fig.write_html("path/to/file.html")
        ```
        """
        check_is_fitted(self)
        return plotting.visualize_distribution(self,
                                               probabilities=probabilities,
                                               min_probability=min_probability,
                                               custom_labels=custom_labels,
                                               title=title,
                                               width=width,
                                               height=height)

    def visualize_approximate_distribution(self,
                                           document: str,
                                           topic_token_distribution: np.ndarray,
                                           normalize: bool = False):
        """ Visualize the topic distribution calculated by `.approximate_topic_distribution`
        on a token level. Thereby indicating the extend to which a certain word or phrases belong
        to a specific topic. The assumption here is that a single word can belong to multiple
        similar topics and as such give information about the broader set of topics within
        a single document.

        Arguments:
            topic_model: A fitted BERTopic instance.
            document: The document for which you want to visualize
                    the approximated topic distribution.
            topic_token_distribution: The topic-token distribution of the document as
                                    extracted by `.approximate_topic_distribution`
            normalize: Whether to normalize, between 0 and 1 (summing to 1), the
                    topic distribution values.

        Returns:
            df: A stylized dataframe indicating the best fitting topics
                for each token.

        Examples:

        ```python
        # Calculate the topic distributions on a token level
        # Note that we need to have `calculate_token_level=True`
        topic_distr, topic_token_distr = topic_model.approximate_distribution(
                docs, calculate_token_level=True
        )

        # Visualize the approximated topic distributions
        df = topic_model.visualize_approximate_distribution(docs[0], topic_token_distr[0])
        df
        ```

        To revert this stylized dataframe back to a regular dataframe,
        you can run the following:

        ```python
        df.data.columns = [column.strip() for column in df.data.columns]
        df = df.data
        ```
        """
        check_is_fitted(self)
        return plotting.visualize_approximate_distribution(self,
                                                           document=document,
                                                           topic_token_distribution=topic_token_distribution,
                                                           normalize=normalize)

    def visualize_hierarchy(self,
                            orientation: str = "left",
                            topics: List[int] = None,
                            top_n_topics: int = None,
                            custom_labels: bool = False,
                            title: str = "<b>Hierarchical Clustering</b>",
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
            title: Title of the plot.
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

        Examples:

        To visualize the hierarchical structure of
        topics simply run:

        ```python
        topic_model.visualize_hierarchy()
        ```

        If you also want the labels visualized of hierarchical topics,
        run the following:

        ```python
        # Extract hierarchical topics and their representations
        hierarchical_topics = topic_model.hierarchical_topics(docs)

        # Visualize these representations
        topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
        ```

        If you want to save the resulting figure:

        ```python
        fig = topic_model.visualize_hierarchy()
        fig.write_html("path/to/file.html")
        ```
        <iframe src="../getting_started/visualization/hierarchy.html"
        style="width:1000px; height: 680px; border: 0px;""></iframe>
        """
        check_is_fitted(self)
        return plotting.visualize_hierarchy(self,
                                            orientation=orientation,
                                            topics=topics,
                                            top_n_topics=top_n_topics,
                                            custom_labels=custom_labels,
                                            title=title,
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
                          title: str = "<b>Similarity Matrix</b>",
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
            title: Title of the plot.
            width: The width of the figure.
            height: The height of the figure.

        Returns:
            fig: A plotly figure

        Examples:

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
                                          title=title,
                                          width=width,
                                          height=height)

    def visualize_barchart(self,
                           topics: List[int] = None,
                           top_n_topics: int = 8,
                           n_words: int = 5,
                           custom_labels: bool = False,
                           title: str = "Topic Word Scores",
                           width: int = 250,
                           height: int = 250) -> go.Figure:
        """ Visualize a barchart of selected topics

        Arguments:
            topics: A selection of topics to visualize.
            top_n_topics: Only select the top n most frequent topics.
            n_words: Number of words to show in a topic
            custom_labels: Whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
            title: Title of the plot.
            width: The width of each figure.
            height: The height of each figure.

        Returns:
            fig: A plotly figure

        Examples:

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
                                           custom_labels=custom_labels,
                                           title=title,
                                           width=width,
                                           height=height)

    def save(self,
             path: str,
             save_embedding_model: bool = True) -> None:
        """ Saves the model to the specified path

        When saving the model, make sure to also keep track of the versions
        of dependencies and Python used. Loading and saving the model should
        be done using the same dependencies and Python. Moreover, models
        saved in one version of BERTopic should not be loaded in other versions.

        Arguments:
            path: the location and name of the file you want to save
            save_embedding_model: Whether to save the embedding model in this class
                                  as you might have selected a local model or one that
                                  is downloaded automatically from the cloud.

        Examples:

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

        Examples:

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
        mappings = self.topic_mapper_.get_mappings(original_topics=True)
        mapped_predictions = [mappings[prediction]
                              if prediction in mappings
                              else -1
                              for prediction in predictions]
        return mapped_predictions

    def _reduce_dimensionality(self,
                               embeddings: Union[np.ndarray, csr_matrix],
                               y: Union[List[int], np.ndarray] = None,
                               partial_fit: bool = False) -> np.ndarray:
        """ Reduce dimensionality of embeddings using UMAP and train a UMAP model

        Arguments:
            embeddings: The extracted embeddings using the sentence transformer module.
            y: The target class for (semi)-supervised dimensionality reduction
            partial_fit: Whether to run `partial_fit` for online learning

        Returns:
            umap_embeddings: The reduced embeddings
        """
        # Partial fit
        if partial_fit:
            if hasattr(self.umap_model, "partial_fit"):
                self.umap_model = self.umap_model.partial_fit(embeddings)
            elif self.topic_representations_ is None:
                self.umap_model.fit(embeddings)

        # Regular fit
        else:
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
                            documents: pd.DataFrame,
                            partial_fit: bool = False,
                            y: np.ndarray = None) -> Tuple[pd.DataFrame,
                                                           np.ndarray]:
        """ Cluster UMAP embeddings with HDBSCAN

        Arguments:
            umap_embeddings: The reduced sentence embeddings with UMAP
            documents: Dataframe with documents and their corresponding IDs
            partial_fit: Whether to run `partial_fit` for online learning

        Returns:
            documents: Updated dataframe with documents and their corresponding IDs
                       and newly added Topics
            probabilities: The distribution of probabilities
        """
        if partial_fit:
            self.hdbscan_model = self.hdbscan_model.partial_fit(umap_embeddings)
            labels = self.hdbscan_model.labels_
            documents['Topic'] = labels
            self.topics_ = labels
        else:
            try:
                self.hdbscan_model.fit(umap_embeddings, y=y)
            except TypeError:
                self.hdbscan_model.fit(umap_embeddings)

            try:
                labels = self.hdbscan_model.labels_
            except AttributeError:
                labels = y
            documents['Topic'] = labels
            self._update_topic_size(documents)

        # Some algorithms have outlier labels (-1) that can be tricky to work
        # with if you are slicing data based on that labels. Therefore, we
        # track if there are outlier labels and act accordingly when slicing.
        self._outliers = 1 if -1 in set(labels) else 0

        # Extract probabilities
        probabilities = None
        if hasattr(self.hdbscan_model, "probabilities_"):
            probabilities = self.hdbscan_model.probabilities_

            if self.calculate_probabilities and is_supported_hdbscan(self.hdbscan_model):
                probabilities = hdbscan_delegator(self.hdbscan_model, "all_points_membership_vectors")

        if not partial_fit:
            self.topic_mapper_ = TopicMapper(self.topics_)
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
        self.c_tf_idf_, words = self._c_tf_idf(documents_per_topic)
        self.topic_representations_ = self._extract_words_per_topic(words, documents)
        self._create_topic_vectors()
        self.topic_labels_ = {key: f"{key}_" + "_".join([word[0] for word in values[:4]])
                              for key, values in
                              self.topic_representations_.items()}

    def _save_representative_docs(self, documents: pd.DataFrame):
        """ Save the 3 most representative docs per topic

        Arguments:
            documents: Dataframe with documents and their corresponding IDs

        Updates:
            self.representative_docs_: Populate each topic with 3 representative docs
        """
        repr_docs, _, _= self._extract_representative_docs(self.c_tf_idf_, 
                                                           documents, 
                                                           self.topic_representations_, 
                                                           nr_samples=500, 
                                                           nr_repr_docs=3)
        self.representative_docs_ = repr_docs

    def _extract_representative_docs(self,
                                     c_tf_idf: csr_matrix,
                                     documents: pd.DataFrame,
                                     topics: Mapping[str, List[Tuple[str, float]]],
                                     nr_samples: int = 500,
                                     nr_repr_docs: int = 5,
                                     ) -> Union[List[str], List[List[int]]]:
        """ Approximate most representative documents per topic by sampling
        a subset of the documents in each topic and calculating which are
        most represenative to their topic based on the cosine similarity between
        c-TF-IDF representations.

        Arguments:
            c_tf_idf: The topic c-TF-IDF representation
            documents: All input documents
            topics: The candidate topics as calculated with c-TF-IDF
            nr_samples: The number of candidate documents to extract per topic
            nr_repr_docs: The number of representative documents to extract per topic

        Returns:
            repr_docs_mappings: A dictionary from topic to representative documents
            representative_docs: A flat list of representative documents
            repr_doc_indices: The indices of representative documents
                              that belong to each topic
        """
        # Sample documents per topic
        documents_per_topic = (
            documents.groupby('Topic')
                     .sample(n=nr_samples, replace=True, random_state=42)
                     .drop_duplicates()
        )

        # Find and extract documents that are most similar to the topic
        repr_docs = []
        repr_docs_indices = []
        repr_docs_mappings = {}
        labels = sorted(list(topics.keys()))
        for index, topic in enumerate(labels):

            # Calculate similarity
            selected_docs = documents_per_topic.loc[documents_per_topic.Topic == topic, "Document"].values
            bow = self.vectorizer_model.transform(selected_docs)
            ctfidf = self.ctfidf_model.transform(bow)
            sim_matrix = cosine_similarity(ctfidf, c_tf_idf[index])

            # Extract top n most representative documents
            nr_docs = nr_repr_docs if len(selected_docs) > nr_repr_docs else len(selected_docs)
            indices = np.argpartition(sim_matrix.reshape(1, -1)[0],
                                      -nr_docs)[-nr_docs:]
            repr_docs.extend([selected_docs[index] for index in indices])
            repr_docs_indices.append([repr_docs_indices[-1][-1] + i + 1 if index != 0 else i for i in range(nr_docs)])
        repr_docs_mappings = {topic: repr_docs[i[0]:i[-1]+1] for topic, i in zip(topics.keys(), repr_docs_indices)}

        return repr_docs_mappings, repr_docs, repr_docs_indices

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
        if self.embedding_model is not None and type(self.embedding_model) is not BaseEmbedder:
            topic_list = list(self.topic_representations_.keys())
            topic_list.sort()

            # Only extract top n words
            n = len(self.topic_representations_[topic_list[0]])
            if self.top_n_words < n:
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

            self.topic_embeddings_ = topic_embeddings

    def _c_tf_idf(self,
                  documents_per_topic: pd.DataFrame,
                  fit: bool = True,
                  partial_fit: bool = False) -> Tuple[csr_matrix, List[str]]:
        """ Calculate a class-based TF-IDF where m is the number of total documents.

        Arguments:
            documents_per_topic: The joined documents per topic such that each topic has a single
                                 string made out of multiple documents
            m: The total number of documents (unjoined)
            fit: Whether to fit a new vectorizer or use the fitted self.vectorizer_model
            partial_fit: Whether to run `partial_fit` for online learning

        Returns:
            tf_idf: The resulting matrix giving a value (importance score) for each word per topic
            words: The names of the words to which values were given
        """
        documents = self._preprocess_text(documents_per_topic.Document.values)

        if partial_fit:
            X = self.vectorizer_model.partial_fit(documents).update_bow(documents)
        elif fit:
            self.vectorizer_model.fit(documents)
            X = self.vectorizer_model.transform(documents)
        else:
            X = self.vectorizer_model.transform(documents)

        # Scikit-Learn Deprecation: get_feature_names is deprecated in 1.0
        # and will be removed in 1.2. Please use get_feature_names_out instead.
        if version.parse(sklearn_version) >= version.parse("1.0.0"):
            words = self.vectorizer_model.get_feature_names_out()
        else:
            words = self.vectorizer_model.get_feature_names()

        if self.seed_topic_list:
            seed_topic_list = [seed for seeds in self.seed_topic_list for seed in seeds]
            multiplier = np.array([1.2 if word in seed_topic_list else 1 for word in words])
        else:
            multiplier = None

        if fit:
            self.ctfidf_model = self.ctfidf_model.fit(X, multiplier=multiplier)

        c_tf_idf = self.ctfidf_model.transform(X)

        return c_tf_idf, words

    def _update_topic_size(self, documents: pd.DataFrame):
        """ Calculate the topic sizes

        Arguments:
            documents: Updated dataframe with documents and their corresponding IDs and newly added Topics
        """
        sizes = documents.groupby(['Topic']).count().sort_values("Document", ascending=False).reset_index()
        self.topic_sizes_ = dict(zip(sizes.Topic, sizes.Document))
        self.topics_ = documents.Topic.astype(int).tolist()

    def _extract_words_per_topic(self,
                                 words: List[str],
                                 documents: pd.DataFrame,
                                 c_tf_idf: csr_matrix = None) -> Mapping[str,
                                                                         List[Tuple[str, float]]]:
        """ Based on tf_idf scores per topic, extract the top n words per topic

        If the top words per topic need to be extracted, then only the `words` parameter
        needs to be passed. If the top words per topic in a specific timestamp, then it
        is important to pass the timestamp-based c-TF-IDF matrix and its corresponding
        labels.

        Arguments:
            words: List of all words (sorted according to tf_idf matrix position)
            documents: DataFrame with documents and their topic IDs
            c_tf_idf: A c-TF-IDF matrix from which to calculate the top words

        Returns:
            topics: The top words per topic
        """
        if c_tf_idf is None:
            c_tf_idf = self.c_tf_idf_

        labels = sorted(list(documents.Topic.unique()))
        labels = [int(label) for label in labels]

        # Get at least the top 30 indices and values per row in a sparse c-TF-IDF matrix
        top_n_words = max(self.top_n_words, 30)
        indices = self._top_n_idx_sparse(c_tf_idf, top_n_words)
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

        # Fine-tune the topic representations
        if isinstance(self.representation_model, list):
            for tuner in self.representation_model:
                topics = tuner.extract_topics(self, documents, c_tf_idf, topics)
        elif isinstance(self.representation_model, BaseRepresentation):
            topics = self.representation_model.extract_topics(self, documents, c_tf_idf, topics)

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
        topics = documents.Topic.tolist().copy()

        # Create topic distance matrix
        if self.topic_embeddings_ is not None:
            topic_embeddings = np.array(self.topic_embeddings_)[self._outliers:, ]
        else:
            topic_embeddings = self.c_tf_idf_[self._outliers:, ].toarray()
        distance_matrix = 1-cosine_similarity(topic_embeddings)
        np.fill_diagonal(distance_matrix, 0)

        # Cluster the topic embeddings using AgglomerativeClustering
        if version.parse(sklearn_version) >= version.parse("1.4.0"):
            cluster = AgglomerativeClustering(self.nr_topics - self._outliers, metric="precomputed", linkage="average")
        else:
            cluster = AgglomerativeClustering(self.nr_topics - self._outliers, affinity="precomputed", linkage="average")
        cluster.fit(distance_matrix)
        new_topics = [cluster.labels_[topic] if topic != -1 else -1 for topic in topics]

        # Map topics
        documents.Topic = new_topics
        self._update_topic_size(documents)
        mapped_topics = {from_topic: to_topic for from_topic, to_topic in zip(topics, new_topics)}
        self.topic_mapper_.add_mappings(mapped_topics)

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
        if self.topic_embeddings_ is not None:
            embeddings = np.array(self.topic_embeddings_)
        else:
            embeddings = self.c_tf_idf_.toarray()
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
        self.topic_mapper_.add_mappings(mapped_topics)
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
        df = pd.DataFrame(self.topic_sizes_.items(), columns=["Old_Topic", "Size"]).sort_values("Size", ascending=False)
        df = df[df.Old_Topic != -1]
        sorted_topics = {**{-1: -1}, **dict(zip(df.Old_Topic, range(len(df))))}
        self.topic_mapper_.add_mappings(sorted_topics)

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
        mappings = self.topic_mapper_.get_mappings(original_topics)

        # Map array of probabilities (probability for assigned topic per document)
        if probabilities is not None:
            if len(probabilities.shape) == 2:
                mapped_probabilities = np.zeros((probabilities.shape[0],
                                                 len(set(mappings.values())) - self._outliers))
                for from_topic, to_topic in mappings.items():
                    if to_topic != -1 and from_topic != -1:
                        mapped_probabilities[:, to_topic] += probabilities[:, from_topic]

                return mapped_probabilities

        return probabilities

    def _preprocess_text(self, documents: np.ndarray) -> List[str]:
        """ Basic preprocessing of text

        Steps:
            * Replace \n and \t with whitespace
            * Only keep alpha-numerical characters
        """
        cleaned_documents = [doc.replace("\n", " ") for doc in documents]
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
                 The parameters that represent models are ignored due to their length.
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

    These mappings are tracked in the `self.mappings_`
    attribute where each set of topic are stacked horizontally.
    For example, the most recent topics can be found in the
    last column. To get a mapping, simply take the two columns
    of topics.

    In other words, it is represented as graph:
    Topic 1 --> Topic 11 --> Topic 4 --> etc.

    Attributes:
        self.mappings_ (np.ndarray) : A  matrix indicating the mappings from one topic
                                      to another. The columns represent a collection of topics
                                      at any time. The last column represents the current state
                                      of topics and the first column represents the initial state
                                      of topics.
    """
    def __init__(self, topics: List[int]):
        """ Initalization of Topic Mapper

        Arguments:
            topics: A list of topics per document
        """
        base_topics = np.array(sorted(set(topics)))
        topics = base_topics.copy().reshape(-1, 1)
        self.mappings_ = np.hstack([topics.copy(), topics.copy()]).tolist()

    def get_mappings(self, original_topics: bool = True) -> Mapping[int, int]:
        """ Get mappings from either the original topics or
        the second-most recent topics to the current topics

        Arguments:
            original_topics: Whether we want to map from the
                             original topics to the most recent topics
                             or from the second-most recent topics.

        Returns:
            mappings: The mappings from old topics to new topics

        Examples:

        To get mappings, simply call:
        ```python
        mapper = TopicMapper(hdbscan_model)
        mappings = mapper.get_mappings(original_topics=False)
        ```
        """
        if original_topics:
            mappings = np.array(self.mappings_)[:, [0, -1]]
            mappings = dict(zip(mappings[:, 0], mappings[:, 1]))
        else:
            mappings = np.array(self.mappings_)[:, [-3, -1]]
            mappings = dict(zip(mappings[:, 0], mappings[:, 1]))
        return mappings

    def add_mappings(self, mappings: Mapping[int, int]):
        """ Add new column(s) of topic mappings

        Arguments:
            mappings: The mappings to add
        """
        for topics in self.mappings_:
            topic = topics[-1]
            if topic in mappings:
                topics.append(mappings[topic])
            else:
                topics.append(-1)

    def add_new_topics(self, mappings: Mapping[int, int]):
        """ Add new row(s) of topic mappings

        Arguments:
            mappings: The mappings to add
        """
        length = len(self.mappings_[0])
        for key, value in mappings.items():
            to_append = [key] + ([None] * (length-2)) + [value]
            self.mappings_.append(to_append)
