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

# utils
from bertopic.utils import create_logger
logger = create_logger()


class BERTopic:
    """ Transformer-based model for Topic Modeling

    Parameters
    ----------
    bert_model, str, default 'distilbert-base-nli-mean-tokens'
        Model to use. Overview of options:
            https://www.sbert.net/docs/pretrained_models.html

    top_n_words : int, default 20
        The number of words per topic to extract

    nr_topics : int, default None
        Specifying the number of topics will reduce the initial
        number of topics to the value specified. This reduction can take
        a while as each reduction in topics (-1) activates a c-TF-IDF calculation.
        IF this is set to None, no reduction is applied.

    n_gram_range : Tuple[int (low), int (high)], default (1, 1)
        The n-gram range for the CountVectorizer. Advised to keep high values
        between 1 and 3. More would likely lead to memory issues.

    min_topic_size : int, optional (default=30)
        The minimum size of the topic.

    n_neighbors: int, default 15
        The size of local neighborhood (in terms of number of neighboring sample points) used
        for manifold approximation (UMAP).

    n_components: int, default 5
        The dimension of the space to embed into when reducing dimensionality with UMAP.

    verbose, bool, optional (default=False)
        Changes the verbosity of the model, Set to True if you want
        to track the stages of the model.
    """
    def __init__(self,
                 bert_model: str = 'distilbert-base-nli-mean-tokens',
                 top_n_words: int = 20,
                 nr_topics: int = None,
                 n_gram_range: Tuple[int, int] = (1, 1),
                 min_topic_size: int = 30,
                 n_neighbors: int = 15,
                 n_components: int = 5,
                 verbose: bool = False):
        self.bert_model = bert_model
        self.top_n_words = top_n_words
        self.nr_topics = nr_topics
        self.n_gram_range = n_gram_range
        self.min_topic_size = min_topic_size
        self.n_neighbors = n_neighbors
        self.n_components = n_components

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

        Parameters
        ----------
        documents : List[str]
            A list of documents to fit on
        """
        self.fit_transform(documents)
        return self

    def fit_transform(self,
                      documents: List[str],
                      debug: bool = False) -> Union[List[int],
                                                    Tuple[pd.DataFrame,
                                                          np.ndarray,
                                                          np.ndarray,
                                                          np.ndarray]]:
        """ Fit the models on a collection of documents, generate topics, and return the docs with topics

        Parameters
        ----------
        documents : List[str]
            A list of documents to fit on

        debug : bool, default False
            Whether to return all intermediate results

        Returns
        -------
        predictions : List[int]
            Topic predictions for each documents
        """
        documents = pd.DataFrame({"Document": documents,
                                  "ID": range(len(documents)),
                                  "Topic": None})

        # Extract BERT sentence embeddings
        embeddings = self._extract_embeddings(documents.Document)

        # Reduce dimensionality with UMAP
        umap_embeddings = self._reduce_dimensionality(embeddings)

        # Cluster UMAP embeddings with HDBSCAN
        documents = self._cluster_embeddings(umap_embeddings, documents)

        # Extract topics by calculating c-TF-IDF
        tf_idf = self._extract_topics(documents)

        if self.nr_topics:
            documents = self._reduce_topics(documents, tf_idf)

        predictions = documents.Topic.to_list()

        if debug:
            return documents, embeddings, umap_embeddings, tf_idf

        return predictions

    def transform(self, documents: Union[str, List[str]]) -> List[int]:
        """ After having fit a model, use transform to predict new instances """
        if isinstance(documents, str):
            documents = [documents]

        embeddings = self._extract_embeddings(documents)
        umap_embeddings = self.umap_model.transform(embeddings)
        predictions, strengths = hdbscan.approximate_predict(self.cluster_model, umap_embeddings)

        if self.mapped_topics:
            predictions = self._map_predictions(predictions)

        return predictions

    def _extract_embeddings(self, documents: List[str]) -> np.ndarray:
        """ Extract sentence/document embeddings through pre-trained embeddings
        For an overview of pre-trained models: https://www.sbert.net/docs/pretrained_models.html

        Parameters
        ----------
        documents : pd.DataFrame
            Dataframe with documents and their corresponding IDs

        Returns
        -------
        embeddings : np.ndarray
            The extracted embeddings using the sentence transformer
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

        Parameters
        ----------
        embeddings : np.ndarray
            The extracted embeddings using the sentence transformer module.

        Returns
        -------
        umap_embeddings : np.ndarray
            The reduced embeddings
        """
        self.umap_model = umap.UMAP(n_neighbors=self.n_neighbors,
                                    n_components=self.n_components,
                                    min_dist=0.0,
                                    metric='cosine').fit(embeddings)
        umap_embeddings = self.umap_model.transform(embeddings)
        logger.info("Reduced dimensionality with UMAP")
        return umap_embeddings

    def _cluster_embeddings(self, umap_embeddings: np.ndarray, documents: pd.DataFrame) -> pd.DataFrame:
        """ Cluster UMAP embeddings with HDBSCAN

        Parameters
        ----------
        umap_embeddings : np.ndarray
            The reduced sentence embeddings with UMAP

        documents : pd.DataFrame
            Dataframe with documents and their corresponding IDs

        Returns
        -------
        documents : pd.DataFrame
            Updated dataframe with documents and their corresponding IDs and newly added Topics
        """
        self.cluster_model = hdbscan.HDBSCAN(min_cluster_size=self.min_topic_size,
                                             metric='euclidean',
                                             cluster_selection_method='eom',
                                             prediction_data=True).fit(umap_embeddings)
        documents['Topic'] = self.cluster_model.labels_
        self._update_topic_size(documents)
        logger.info("Clustered UMAP embeddings with HDBSCAN")
        return documents

    def _extract_topics(self,
                        documents: pd.DataFrame,
                        topic_reduction: bool = False) -> np.ndarray:
        """ Extract topics from the clusters using a class-based TF-IDF

        Parameters
        ----------
        documents : pd.DataFrame
            Dataframe with documents and their corresponding IDs

        topic_reduction : bool, default = False
            Controls verbosity if topic reduction is applied since
            it should not show reduction over and over again.

        Returns
        -------
        tf_idf : np.ndarray
            The resulting matrix giving a value (importance score) for each word per topic
        """
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        tf_idf, words = self._c_tf_idf(documents_per_topic, m=len(documents))
        self._extract_words_per_topic(tf_idf, words)

        if topic_reduction:
            logger.info("Constructed topics with c-TF-IDF")

        return tf_idf

    def _c_tf_idf(self, documents_per_topic: pd.DataFrame, m: int) -> Tuple[np.ndarray, List[str]]:
        """ Calculate a class-based TF-IDF where m is the number of total documents.

        C-TF-IDF can best be explained as a TF-IDF formula adopted for multiple classes
        by joining all documents per class. Thus, each class is converted to a single document
        instead of set of documents. Then, the frequency of words **t** are extracted for
        each class **i** and divided by the total number of words **w**.
        Next, the total, unjoined, number of documents across all classes **m** is divided by the total
        sum of word **i** across all classes.

        Parameters
        ----------
        documents_per_topic : pd.DataFrame
            The joined documents per topic such that each topic has a single
            string made out of multiple documents

        m : int
            The total number of documents (unjoined)


        Returns
        -------
        tf_idf : np.ndarray
            The resulting matrix giving a value (importance score) for each word per topic

        count : CountVectorizer
            The vectorized used
        """
        documents = documents_per_topic.Document.values
        count = CountVectorizer(ngram_range=self.n_gram_range, stop_words="english").fit(documents)
        t = count.transform(documents).toarray()
        w = t.sum(axis=1)
        tf = np.divide(t.T, w)
        sum_t = t.sum(axis=0)
        idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
        tf_idf = np.multiply(tf, idf)
        words = count.get_feature_names()
        return tf_idf, words

    def _update_topic_size(self, documents: pd.DataFrame) -> None:
        """ Calculate the topic sizes

        Parameters
        ----------
        documents : pd.DataFrame
            Updated dataframe with documents and their corresponding IDs and newly added Topics
        """
        sizes = documents.groupby(['Topic']).count().sort_values("Document", ascending=False).reset_index()
        self.topic_sizes = dict(zip(sizes.Topic, sizes.Document))

    def _extract_words_per_topic(self, tf_idf: np.ndarray, words: List[str]):
        """ Based on tf_idf scores per topic, extract the top n words per topic

        Parameters
        ----------
        tf_idf : np.ndarray
            tf_idf matrix

        words : List[str]
            List of all words (sorted according to tf_idf matrix position)
        """
        labels = sorted(list(self.topic_sizes.keys()))
        tf_idf_transposed = tf_idf.T
        indices = tf_idf_transposed.argsort()[:, -self.top_n_words:]
        self.topics = {label: [(words[j], tf_idf_transposed[i][j])
                               for j in indices[i]][::-1]
                       for i, label in enumerate(labels)}

    def get_topics(self) -> Dict[str, Tuple[str, float]]:
        """ Return topics with top n words and their c-TF-IDF score """
        return self.topics

    def get_topic(self, topic: int) -> Dict[str, Tuple[str, float]]:
        """ Return top n words for a specific topic and their c-TF-IDF scores """
        return self.topics[topic]

    def get_topics_freq(self) -> pd.DataFrame:
        """ Return the the size of topics (descending order) """
        return pd.DataFrame(self.topic_sizes.items(), columns=['Topic', 'Count']).sort_values("Count", ascending=False)

    def get_topic_freq(self, topic: int) -> int:
        """ Return the the size of a topic """
        return self.topic_sizes.items()[topic]

    def _reduce_topics(self, documents: pd.DataFrame, tf_idf: np.ndarray) -> pd.DataFrame:
        """ Reduce topics to self.nr_topics

        Parameters
        ----------
        documents : pd.DataFrame
            Dataframe with documents and their corresponding IDs and Topics

        tf_idf : np.ndarray
            c-TF-IDF matrix

        Returns
        -------
        documents : pd.DataFrame
            Updated dataframe with documents and the reduced number of Topics
        """
        self.mapped_topics = {}
        initial_nr_topics = len(self.get_topics())
        nr_to_reduce = initial_nr_topics - self.nr_topics

        for _ in range(nr_to_reduce):
            # Create topic similarity matrix
            similarities = cosine_similarity(tf_idf.T)
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

    def save(self, path: str) -> None:
        """ Saves the model to the specified path """
        with open(path, 'wb') as file:
            joblib.dump(self, file)

    @classmethod
    def load(cls, path: str):
        """ Loads the model from the specified path """
        with open(path, 'rb') as file:
            return joblib.load(file)



