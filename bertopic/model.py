import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Union

# Models
import umap
import hdbscan
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# BERTopic
from .ctfidf import ClassTFIDF
from .utils import create_logger, check_documents_type
logger = create_logger()


class BERTopic:
    """Transformer-based model for Topic Modeling

    Arguments:
        bert_model: Model to use. Overview of options can be found here
                    https://www.sbert.net/docs/pretrained_models.html
        top_n_words: The number of words per topic to extract
        nr_topics: Specifying the number of topics will reduce the initial
                   number of topics to the value specified. This reduction can take
                   a while as each reduction in topics (-1) activates a c-TF-IDF calculation.
                   IF this is set to None, no reduction is applied.
        n_gram_range: The n-gram range for the CountVectorizer.
                      Advised to keep high values between 1 and 3.
                      More would likely lead to memory issues.
        min_topic_size: The minimum size of the topic.
        n_neighbors: The size of local neighborhood (in terms of number of neighboring sample points) used
                     for manifold approximation (UMAP).
        n_components: The dimension of the space to embed into when reducing dimensionality with UMAP.
        stop_words: Stopwords that can be used as either a list of strings, or the name of the
                    language as a string. For example: 'english' or ['the', 'and', 'I'].
        verbose: Changes the verbosity of the model, Set to True if you want
                 to track the stages of the model.

    Usage:
    ```python
    from bertopic import BERTopic
    from sklearn.datasets import fetch_20newsgroups

    docs = fetch_20newsgroups(subset='all')['data']

    model = BERTopic("distilbert-base-nli-mean-tokens", verbose=True)
    topics = model.fit_transform(docs)
    ```
    """
    def __init__(self,
                 bert_model: str = 'distilbert-base-nli-mean-tokens',
                 top_n_words: int = 20,
                 nr_topics: int = None,
                 n_gram_range: Tuple[int, int] = (1, 1),
                 min_topic_size: int = 30,
                 n_neighbors: int = 15,
                 n_components: int = 5,
                 stop_words: Union[str, List[str]] = None,
                 verbose: bool = False):
        self.bert_model = bert_model
        self.top_n_words = top_n_words
        self.nr_topics = nr_topics
        self.n_gram_range = n_gram_range
        self.min_topic_size = min_topic_size
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.stop_words = stop_words

        self.umap_model = None
        self.cluster_model = None
        self.topics = None
        self.topic_sizes = None
        self.reduced_topics_mapped = None
        self.mapped_topics = None

        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)

    def fit(self, documents: List[str]):
        """ Fit the models (Bert, UMAP, and, HDBSCAN) on a collection of documents and generate topics

        Arguments:
            documents: A list of documents to fit on
        """
        check_documents_type(documents)
        self.fit_transform(documents)
        return self

    def fit_transform(self,
                      documents: List[str]) -> Tuple[List[int],
                                                     np.ndarray]:
        """ Fit the models on a collection of documents, generate topics, and return the docs with topics

        Arguments:
            documents: A list of documents to fit on

        Returns:
            predictions: Topic predictions for each documents
            probabilities: The topic probability distribution
        """
        check_documents_type(documents)
        documents = pd.DataFrame({"Document": documents,
                                  "ID": range(len(documents)),
                                  "Topic": None})

        # Extract BERT sentence embeddings
        embeddings = self._extract_embeddings(documents.Document)

        # Reduce dimensionality with UMAP
        umap_embeddings = self._reduce_dimensionality(embeddings)

        # Cluster UMAP embeddings with HDBSCAN
        documents, probabilities = self._cluster_embeddings(umap_embeddings, documents)

        # Extract topics by calculating c-TF-IDF
        c_tf_idf = self._extract_topics(documents)

        if self.nr_topics:
            documents = self._reduce_topics(documents, c_tf_idf)
            probabilities = self._map_probabilities(probabilities)

        predictions = documents.Topic.to_list()

        return predictions, probabilities

    def transform(self, documents: Union[str, List[str]]) -> Tuple[List[int], np.ndarray]:
        """ After having fit a model, use transform to predict new instances

        Arguments:
            documents: A single document or a list of documents to fit on

        Returns:
            predictions: Topic predictions for each documents
            probabilities: The topic probability distribution
        """
        if isinstance(documents, str):
            documents = [documents]

        embeddings = self._extract_embeddings(documents)
        umap_embeddings = self.umap_model.transform(embeddings)
        probabilities = hdbscan.membership_vector(self.cluster_model, umap_embeddings)
        predictions, _ = hdbscan.approximate_predict(self.cluster_model, umap_embeddings)

        if self.mapped_topics:
            predictions = self._map_predictions(predictions)
            probabilities = self._map_probabilities(probabilities)

        if len(documents) == 1:
            probabilities = probabilities.flatten()

        return predictions, probabilities

    def _extract_embeddings(self, documents: List[str]) -> np.ndarray:
        """ Extract sentence/document embeddings through pre-trained embeddings
        For an overview of pre-trained models: https://www.sbert.net/docs/pretrained_models.html

        Arguments:
            documents: Dataframe with documents and their corresponding IDs

        Returns:
            embeddings: The extracted embeddings using the sentence transformer
                        module. Typically uses pre-trained huggingface models.
        """
        model = SentenceTransformer(self.bert_model)
        logger.info("Loaded BERT model")
        embeddings = model.encode(documents, show_progress_bar=False)
        logger.info("Transformed documents to Embeddings")
        return embeddings

    def _map_predictions(self, predictions):
        """ Map predictions to the correct topics if topics were reduced """
        mapped_predictions = []
        for prediction in predictions:
            while self.mapped_topics.get(prediction):
                prediction = self.mapped_topics[prediction]
            mapped_predictions.append(prediction)
        return mapped_predictions

    def _reduce_dimensionality(self, embeddings: np.ndarray) -> np.ndarray:
        """ Reduce dimensionality of embeddings using UMAP and train a UMAP model

        Arguments:
            embeddings: The extracted embeddings using the sentence transformer module.

        Returns:
            umap_embeddings: The reduced embeddings
        """
        self.umap_model = umap.UMAP(n_neighbors=self.n_neighbors,
                                    n_components=self.n_components,
                                    min_dist=0.0,
                                    metric='cosine').fit(embeddings)
        umap_embeddings = self.umap_model.transform(embeddings)
        logger.info("Reduced dimensionality with UMAP")
        return umap_embeddings

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
        self.cluster_model = hdbscan.HDBSCAN(min_cluster_size=self.min_topic_size,
                                             metric='euclidean',
                                             cluster_selection_method='eom',
                                             prediction_data=True).fit(umap_embeddings)
        documents['Topic'] = self.cluster_model.labels_
        probabilities = hdbscan.all_points_membership_vectors(self.cluster_model)
        self._update_topic_size(documents)
        logger.info("Clustered UMAP embeddings with HDBSCAN")
        return documents, probabilities

    def _extract_topics(self,
                        documents: pd.DataFrame,
                        topic_reduction: bool = False) -> np.ndarray:
        """ Extract topics from the clusters using a class-based TF-IDF

        Arguments:
            documents: Dataframe with documents and their corresponding IDs

            topic_reduction: Controls verbosity if topic reduction is applied since
                             it should not show reduction over and over again.

        Returns:
            c_tf_idf: The resulting matrix giving a value (importance score) for each word per topic
        """
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        c_tf_idf, words = self._c_tf_idf(documents_per_topic, m=len(documents))
        self._extract_words_per_topic(c_tf_idf, words)

        if topic_reduction:
            logger.info("Constructed topics with c-TF-IDF")

        return c_tf_idf

    def _c_tf_idf(self, documents_per_topic: pd.DataFrame, m: int) -> Tuple[np.ndarray, List[str]]:
        """ Calculate a class-based TF-IDF where m is the number of total documents.

        Arguments:
            documents_per_topic: The joined documents per topic such that each topic has a single
                                 string made out of multiple documents
            m: The total number of documents (unjoined)

        Returns:
            tf_idf: The resulting matrix giving a value (importance score) for each word per topic
            words: The names of the words to which values were given
        """
        documents = documents_per_topic.Document.values
        count = CountVectorizer(ngram_range=self.n_gram_range, stop_words=self.stop_words).fit(documents)
        words = count.get_feature_names()
        X = count.transform(documents)
        transformer = ClassTFIDF().fit(X, n_samples=m)
        c_tf_idf = transformer.transform(X).toarray()

        return c_tf_idf, words

    def _update_topic_size(self, documents: pd.DataFrame) -> None:
        """ Calculate the topic sizes

        Arguments:
            documents: Updated dataframe with documents and their corresponding IDs and newly added Topics
        """
        sizes = documents.groupby(['Topic']).count().sort_values("Document", ascending=False).reset_index()
        self.topic_sizes = dict(zip(sizes.Topic, sizes.Document))

    def _extract_words_per_topic(self, c_tf_idf: np.ndarray, words: List[str]):
        """ Based on tf_idf scores per topic, extract the top n words per topic

        Arguments:
        tf_idf: tf_idf matrix
        words: List of all words (sorted according to tf_idf matrix position)
        """
        labels = sorted(list(self.topic_sizes.keys()))
        indices = c_tf_idf.argsort()[:, -self.top_n_words:]
        self.topics = {label: [(words[j], c_tf_idf[i][j])
                               for j in indices[i]][::-1]
                       for i, label in enumerate(labels)}

    def get_topics(self) -> Dict[str, Tuple[str, float]]:
        """ Return topics with top n words and their c-TF-IDF score """
        return self.topics

    def get_topic(self, topic: int) -> Union[Dict[str, Tuple[str, float]], bool]:
        """ Return top n words for a specific topic and their c-TF-IDF scores """
        if self.topics.get(topic):
            return self.topics[topic]
        else:
            return False

    def get_topics_freq(self) -> pd.DataFrame:
        """ Return the the size of topics (descending order) """
        return pd.DataFrame(self.topic_sizes.items(), columns=['Topic', 'Count']).sort_values("Count", ascending=False)

    def get_topic_freq(self, topic: int) -> int:
        """ Return the the size of a topic """
        return self.topic_sizes.items()[topic]

    def _reduce_topics(self, documents: pd.DataFrame, c_tf_idf: np.ndarray) -> pd.DataFrame:
        """ Reduce topics to self.nr_topics

        Arguments:
            documents: Dataframe with documents and their corresponding IDs and Topics
            c_tf_idf: c-TF-IDF matrix

        Returns:
            documents: Updated dataframe with documents and the reduced number of Topics
        """
        self.mapped_topics = {}
        initial_nr_topics = len(self.get_topics())
        nr_to_reduce = initial_nr_topics - self.nr_topics

        for _ in range(nr_to_reduce):
            # Create topic similarity matrix
            similarities = cosine_similarity(c_tf_idf)
            np.fill_diagonal(similarities, 0)

            # Find most similar topic to least common topic
            topic_to_merge = self.get_topics_freq().iloc[-1].Topic
            topic_to_merge_into = np.argmax(similarities[topic_to_merge + 1]) - 1

            # Update Topic labels
            documents.loc[documents.Topic == topic_to_merge, "Topic"] = topic_to_merge_into
            self.mapped_topics[topic_to_merge] = topic_to_merge_into

            # Update new topic content
            self._update_topic_size(documents)
            self._extract_topics(documents, topic_reduction=True)

        if initial_nr_topics <= self.nr_topics:
            logger.info(f"Since {initial_nr_topics} were found, they could not be reduced to {self.nr_topics}")
        else:
            logger.info(f"Reduced number of topics from {initial_nr_topics} to {self.nr_topics}")

        return documents

    def _map_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        """ Map the probabilities to the reduced topics.
        This is achieved by adding the probabilities together
        of all topics that were mapped to the same topic. Then,
        the topics that were mapped from were set to 0 as they
        were reduced.
        """
        for from_topic, to_topic in self.mapped_topics.items():
            probabilities[:, to_topic] += probabilities[:, from_topic]
            probabilities[:, from_topic] = 0

        return probabilities.round(3)

    def save(self, path: str) -> None:
        """ Saves the model to the specified path """
        with open(path, 'wb') as file:
            joblib.dump(self, file)

    @classmethod
    def load(cls, path: str):
        """ Loads the model from the specified path """
        with open(path, 'rb') as file:
            return joblib.load(file)



