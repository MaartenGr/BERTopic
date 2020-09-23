import numpy as np
import pandas as pd

import umap
import hdbscan
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from typing import List, Tuple, Dict
import time


def ts():
    """ Generates a timestamp for use in logging messages when verbose=True """
    return time.ctime(time.time())


class Topic:
    def __init__(self,
                 bert_model: str = 'distilbert-base-nli-mean-tokens',
                 verbose=False,
                 nr_topics: int = None):
        self.bert_model = bert_model
        self.umap_model = None
        self.cluster_model = None
        self.topics = None
        self.topic_sizes = None
        self.verbose = verbose
        self.nr_topics = nr_topics
        self.reduced_topics_mapped = None

    def fit(self, documents: List[str]):
        """ Fit the models (Bert, UMAP, and, HDBSCAN) on a collection of documents and generate topics

        Parameters
        ----------
        documents : List[str]
            A list of documents to fit on
        """
        self.fit_transform(documents)
        return self

    def fit_transform(self, documents: List[str]) -> pd.DataFrame:
        """ Fit the models on a collection of documents, generate topics, and return the docs with topics

        Parameters
        ----------
        documents : List[str]
            A list of documents to fit on

        Returns
        -------
        documents : pd.DataFrame
            Contains documents and their respective topics
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
        tf_idf = self._extract_topics(documents, ngram_range=(1, 1))

        if self.nr_topics:
            documents = self._reduce_topics(documents, tf_idf)

        return documents

    def transform(self, documents: List[str]) -> List[int]:
        """ After having fit a model, use transform to predict new instances """
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

        if self.verbose:
            print(ts(), "Loaded BERT model")

        embeddings = model.encode(documents, show_progress_bar=False)

        if self.verbose:
            print(ts(), "Transformed documents to Embeddings")
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
        self.umap_model = umap.UMAP(n_neighbors=15,
                                    n_components=5,
                                    min_dist=0.0,
                                    metric='cosine').fit(embeddings)
        umap_embeddings = self.umap_model.transform(embeddings)

        if self.verbose:
            print(ts(), "Reduced dimensionality with UMAP")
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
        self.cluster_model = hdbscan.HDBSCAN(min_cluster_size=30,
                                             metric='euclidean',
                                             cluster_selection_method='eom',
                                             prediction_data=True).fit(umap_embeddings)
        documents['Topic'] = self.cluster_model.labels_
        self._update_topic_size(documents)

        if self.verbose:
            print(ts(), "Clustered UMAP embeddings with HDBSCAN")
        return documents

    def _extract_topics(self,
                        documents: pd.DataFrame,
                        ngram_range: tuple = (1, 1),
                        topic_reduction: bool = False) -> np.ndarray:
        """ Extract topics from the clusters using a class-based TF-IDF

        Parameters
        ----------
        documents : pd.DataFrame
            Dataframe with documents and their corresponding IDs

        ngram_range : tuple, default (1, 1)
            The n-gram range, a higher range can significantly slow
            down the calculations.

        topic_reduction : bool, default = False
            Controls verbosity if topic reduction is applied since
            it should not show reduction over and over again.

        Returns
        -------
        tf_idf : np.ndarray
            The resulting matrix giving a value (importance score) for each word per topic
        """
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        tf_idf, words = self._c_tf_idf(documents_per_topic, m=len(documents), ngram_range=ngram_range)
        self._extract_words_per_topic(tf_idf, words, n=20)

        if self.verbose and not topic_reduction:
            print(ts(), "Constructed topics with c-TF-IDF")

        return tf_idf

    @staticmethod
    def _c_tf_idf(documents_per_topic: pd.DataFrame,
                  m: int,
                  ngram_range: tuple = (1, 1)) -> Tuple[np.ndarray, List[str]]:
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

        ngram_range : tuple, default (1, 1)
            The n-gram range, a higher range can significantly slow
            down the calculations.

        Returns
        -------
        tf_idf : np.ndarray
            The resulting matrix giving a value (importance score) for each word per topic

        count : CountVectorizer
            The vectorized used
        """
        documents = documents_per_topic.Document.values
        count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
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

    def _extract_words_per_topic(self, tf_idf: np.ndarray, words: List[str], n: int = 20):
        """ Based on tf_idf scores per topic, extract the top n words per topic

        Parameters
        ----------
        tf_idf : np.ndarray
            tf_idf matrix

        words : List[str]
            List of all words (sorted according to tf_idf matrix position)

        n : int, default 20
            The number of words per topic to extract
        """
        labels = sorted(list(self.topic_sizes.keys()))
        tf_idf_transposed = tf_idf.T
        indices = tf_idf_transposed.argsort()[:, -n:]
        self.topics = {label: [(words[j], tf_idf_transposed[i][j])
                               for j in indices[i]][::-1]
                       for i, label in enumerate(labels)}

    def get_topics(self) -> Dict[str, Tuple[str, float]]:
        """ Return topics with top n words and their c-TF-IDF score """
        return self.topics

    def get_topic_sizes(self) -> pd.DataFrame:
        """ Return the the size of topics (descending order) """
        return pd.DataFrame(self.topic_sizes.items(), columns=['Topic', 'Count']).sort_values("Count", ascending=False)

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
            topic_to_merge = self.get_topic_sizes().iloc[-1].Topic
            topic_to_merge_into = np.argmax(similarities[topic_to_merge + 1]) - 1

            # Update Topic labels
            documents.loc[documents.Topic == topic_to_merge, "Topic"] = topic_to_merge_into
            self.mapped_topics[topic_to_merge] = topic_to_merge_into

            # Update new topic content
            self._update_topic_size(documents)
            self._extract_topics(documents, ngram_range=(1, 1), topic_reduction=True)

        if self.verbose:
            print(ts(), f"Reduced number of topics from {initial_nr_topics} to {self.nr_topics}")

        return documents



