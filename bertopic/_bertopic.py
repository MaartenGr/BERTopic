import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import gc
import re
import joblib
import inspect
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse.csr import csr_matrix
from typing import List, Tuple, Union, Mapping, Any

# Models
import torch
import umap
import hdbscan
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# BERTopic
from ._ctfidf import ClassTFIDF
from ._utils import MyLogger, check_documents_type, check_embeddings_shape, check_is_fitted
from ._embeddings import languages
from ._mmr import mmr

# Visualization
try:
    import matplotlib.pyplot as plt
    import plotly.express as px
    _HAS_VIZ = True
except ModuleNotFoundError as e:
    _HAS_VIZ = False

# Flair
try:
    from flair.embeddings import DocumentEmbeddings, TokenEmbeddings, DocumentPoolEmbeddings
    from flair.data import Sentence
    _HAS_FLAIR = True
except ModuleNotFoundError as e:
    DocumentEmbeddings, TokenEmbeddings, DocumentPoolEmbeddings = None, None, None
    _HAS_FLAIR = False

logger = MyLogger("WARNING")


class BERTopic:
    """BERTopic is a topic modeling technique that leverages BERT embeddings and
    c-TF-IDF to create dense clusters allowing for easily interpretable topics
    whilst keeping important words in the topic descriptions.

    Usage:

    ```python
    from bertopic import BERTopic
    from sklearn.datasets import fetch_20newsgroups

    docs = fetch_20newsgroups(subset='all')['data']

    model = BERTopic("distilbert-base-nli-mean-tokens", verbose=True)
    topics = model.fit_transform(docs)
    ```

    If you want to use your own embeddings, use it as follows:

    ```python
    from bertopic import BERTopic
    from sklearn.datasets import fetch_20newsgroups
    from sentence_transformers import SentenceTransformer

    # Create embeddings
    docs = fetch_20newsgroups(subset='all')['data']
    sentence_model = SentenceTransformer("distilbert-base-nli-mean-tokens")
    embeddings = sentence_model.encode(docs, show_progress_bar=True)

    # Create topic model
    model = BERTopic(verbose=True)
    topics = model.fit_transform(docs, embeddings)
    ```

    Due to the stochastisch nature of UMAP, the results from BERTopic might differ
    and the quality can degrade. Using your own embeddings allows you to
    try out BERTopic several times until you find the topics that suit
    you best.
    """
    def __init__(self,
                 language: str = "english",
                 top_n_words: int = 10,
                 nr_topics: Union[int, str] = None,
                 n_gram_range: Tuple[int, int] = (1, 1),
                 min_topic_size: int = 10,
                 n_neighbors: int = 15,
                 stop_words: Union[str, List[str]] = None,
                 verbose: bool = False,
                 low_memory: bool = False,
                 embedding_model: Union[str, SentenceTransformer, DocumentEmbeddings, TokenEmbeddings] = None,
                 umap_model: umap.UMAP = None,
                 hdbscan_model: hdbscan.HDBSCAN = None,
                 vectorizer_model: CountVectorizer = None
                 ):
        """BERTopic initialization

        Arguments:
            language: The main language used in your documents. For a full overview of supported languages
                      see bertopic.embeddings.languages. Select "multilingual" to load in a model that
                      support 50+ languages.
            top_n_words: The number of words per topic to extract
            nr_topics: Specifying the number of topics will reduce the initial
                       number of topics to the value specified. This reduction can take
                       a while as each reduction in topics (-1) activates a c-TF-IDF calculation.
                       IF this is set to None, no reduction is applied. Use "auto" to automatically
                       reduce topics that have a similarity of at least 0.9, do not maps all others.
            n_gram_range: The n-gram range for the CountVectorizer.
                          Advised to keep high values between 1 and 3.
                          More would likely lead to memory issues.
                          Note that this will not be used if you pass in your own CountVectorizer.
            min_topic_size: The minimum size of the topic.
            n_neighbors: The size of local neighborhood (in terms of number of neighboring sample points) used
                         for manifold approximation (UMAP).
            stop_words: Stopwords that can be used as either a list of strings, or the name of the
                        language as a string. For example: 'english' or ['the', 'and', 'I'].
                        Note that this will not be used if you pass in your own CountVectorizer.
            verbose: Changes the verbosity of the model, Set to True if you want
                     to track the stages of the model.
            low_memory: Removes the calculation of probabilities and sets UMAP low memory to True to
                        make sure less memory is used. This also speeds up computation.
                        NOTE: since probabilities are not calculated, you cannot use the corresponding
                        visualization `visualize_probabilities`.
            embedding_model: You can pass in either a string relating to one of the following models:
                                - https://www.sbert.net/docs/pretrained_models.html
                             You can use your own SentenceTransformer() model to be used instead
                             with your own custom parameters. Moreover, it can also take in
                             any Flair DocumentEmbedding model.
            umap_model: You can pass in a umap.UMAP model to be used instead of the default
            hdbscan_model: You can pass in a hdbscan.HDBSCAN model to be used instead of the default
            vectorizer_model: Pass in your own CountVectorizer from scikit-learn


        Usage:

        ```python
        from bertopic import BERTopic
        model = BERTopic(language = "english",
                         embedding_model = None,
                         top_n_words = 10,
                         nr_topics = 30,
                         n_gram_range = (1, 1),
                         min_topic_size = 10,
                         n_neighbors = 15,
                         n_components = 5,
                         stop_words = None,
                         verbose = True,
                         vectorizer_model = None)
        ```
        """
        # Topic-based parameters
        if top_n_words > 30:
            raise ValueError("top_n_words should be lower or equal to 30. The preferred value is 10.")
        self.top_n_words = top_n_words
        self.nr_topics = nr_topics
        self.low_memory = low_memory
        self.verbose = verbose
        self.min_topic_size = min_topic_size

        # Embedding model
        self.language = language if not embedding_model else None
        self.embedding_model = embedding_model

        # Vectorizer
        self.stop_words = stop_words
        self.n_gram_range = n_gram_range
        self.vectorizer_model = vectorizer_model or CountVectorizer(ngram_range=self.n_gram_range,
                                                                    stop_words=self.stop_words)

        # UMAP
        self.n_neighbors = n_neighbors
        self.umap_model = umap_model or umap.UMAP(n_neighbors=self.n_neighbors,
                                                  n_components=5,
                                                  min_dist=0.0,
                                                  metric='cosine',
                                                  low_memory=self.low_memory)

        # HDBSCAN
        self.hdbscan_model = hdbscan_model or hdbscan.HDBSCAN(min_cluster_size=self.min_topic_size,
                                                              metric='euclidean',
                                                              cluster_selection_method='eom',
                                                              prediction_data=True)

        self.topics = None
        self.topic_sizes = None
        self.reduced_topics_mapped = None
        self.mapped_topics = None
        self.topic_embeddings = None
        self.topic_sim_matrix = None
        self.custom_embeddings = False

        if verbose:
            logger.set_level("DEBUG")

    def fit(self,
            documents: List[str],
            embeddings: np.ndarray = None):
        """ Fit the models (Bert, UMAP, and, HDBSCAN) on a collection of documents and generate topics

        Arguments:
            documents: A list of documents to fit on
            embeddings: Pre-trained document embeddings. These can be used
                        instead of the sentence-transformer model

        Usage:

        ```python
        from bertopic import BERTopic
        from sklearn.datasets import fetch_20newsgroups

        docs = fetch_20newsgroups(subset='all')['data']
        model = BERTopic("distilbert-base-nli-mean-tokens", verbose=True).fit(docs)
        ```

        If you want to use your own embeddings, use it as follows:

        ```python
        from bertopic import BERTopic
        from sklearn.datasets import fetch_20newsgroups
        from sentence_transformers import SentenceTransformer

        # Create embeddings
        docs = fetch_20newsgroups(subset='all')['data']
        sentence_model = SentenceTransformer("distilbert-base-nli-mean-tokens")
        embeddings = sentence_model.encode(docs, show_progress_bar=True)

        # Create topic model
        model = BERTopic(None, verbose=True).fit(docs, embeddings)
        ```
        """
        self.fit_transform(documents, embeddings)
        return self

    def fit_transform(self,
                      documents: List[str],
                      embeddings: np.ndarray = None) -> Tuple[List[int],
                                                              Union[np.ndarray, None]]:
        """ Fit the models on a collection of documents, generate topics, and return the docs with topics

        Arguments:
            documents: A list of documents to fit on
            embeddings: Pre-trained document embeddings. These can be used
                        instead of the sentence-transformer model

        Returns:
            predictions: Topic predictions for each documents
            probabilities: The topic probability distribution which is returned by default.
                           If `low_memory` in BERTopic is set to False, then the
                           probabilities are not calculated to speed up computation and
                           decrease memory usage.

        Usage:

        ```python
        from bertopic import BERTopic
        from sklearn.datasets import fetch_20newsgroups

        docs = fetch_20newsgroups(subset='all')['data']

        model = BERTopic("distilbert-base-nli-mean-tokens", verbose=True)
        topics = model.fit_transform(docs)
        ```

        If you want to use your own embeddings, use it as follows:

        ```python
        from bertopic import BERTopic
        from sklearn.datasets import fetch_20newsgroups
        from sentence_transformers import SentenceTransformer

        # Create embeddings
        docs = fetch_20newsgroups(subset='all')['data']
        sentence_model = SentenceTransformer("distilbert-base-nli-mean-tokens")
        embeddings = sentence_model.encode(docs, show_progress_bar=True)

        # Create topic model
        model = BERTopic(None, verbose=True)
        topics = model.fit_transform(docs, embeddings)
        ```
        """
        check_documents_type(documents)
        check_embeddings_shape(embeddings, documents)

        documents = pd.DataFrame({"Document": documents,
                                  "ID": range(len(documents)),
                                  "Topic": None})

        # Extract embeddings
        if not any([isinstance(embeddings, np.ndarray), isinstance(embeddings, csr_matrix)]):
            self.embedding_model = self._select_embedding_model()
            embeddings = self._extract_embeddings(documents.Document, verbose=self.verbose)
            logger.info("Transformed documents to Embeddings")
        else:
            self.custom_embeddings = True

        # Reduce dimensionality with UMAP
        umap_embeddings = self._reduce_dimensionality(embeddings)

        # Cluster UMAP embeddings with HDBSCAN
        documents, probabilities = self._cluster_embeddings(umap_embeddings, documents)

        # Extract topics by calculating c-TF-IDF
        self._extract_topics(documents)

        if self.nr_topics:
            documents = self._reduce_topics(documents)
            probabilities = self._map_probabilities(probabilities)

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
                           If `low_memory` in BERTopic is set to False, then the
                           probabilities are not calculated to speed up computation and
                           decrease memory usage.

        Usage:

        ```python
        from bertopic import BERTopic
        from sklearn.datasets import fetch_20newsgroups

        docs = fetch_20newsgroups(subset='all')['data']
        model = BERTopic("distilbert-base-nli-mean-tokens", verbose=True).fit(docs)
        topics = model.transform(docs)
        ```

        If you want to use your own embeddings:

        ```python
        from bertopic import BERTopic
        from sklearn.datasets import fetch_20newsgroups
        from sentence_transformers import SentenceTransformer

        # Create embeddings
        docs = fetch_20newsgroups(subset='all')['data']
        sentence_model = SentenceTransformer("distilbert-base-nli-mean-tokens")
        embeddings = sentence_model.encode(docs, show_progress_bar=True)

        # Create topic model
        model = BERTopic(None, verbose=True).fit(docs, embeddings)
        topics = model.transform(docs, embeddings)
        ```
        """
        check_is_fitted(self)
        check_embeddings_shape(embeddings, documents)

        if isinstance(documents, str):
            documents = [documents]

        if not isinstance(embeddings, np.ndarray):
            self.embedding_model = self._select_embedding_model()
            embeddings = self._extract_embeddings(documents, verbose=self.verbose)

        umap_embeddings = self.umap_model.transform(embeddings)
        predictions, _ = hdbscan.approximate_predict(self.hdbscan_model, umap_embeddings)

        if not self.low_memory:
            probabilities = hdbscan.membership_vector(self.hdbscan_model, umap_embeddings)
            if len(documents) == 1:
                probabilities = probabilities.flatten()
        else:
            probabilities = None

        if self.mapped_topics:
            predictions = self._map_predictions(predictions)
            probabilities = self._map_probabilities(probabilities)

        return predictions, probabilities

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

        """
        if self.custom_embeddings:
            raise Exception("This method can only be used if you did not use custom embeddings.")

        topic_list = list(self.topics.keys())
        topic_list.sort()

        # Extract search_term embeddings and compare with topic embeddings
        search_embedding = self._extract_embeddings([search_term], verbose=False).flatten()
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
                      stop_words: str = None,
                      vectorizer_model: CountVectorizer = None):
        """ Updates the topic representation by recalculating c-TF-IDF with the new
        parameters as defined in this function.

        When you have trained a model and viewed the topics and the words that represent them,
        you might not be satisfied with the representation. Perhaps you forgot to remove
        stop_words or you want to try out a different n_gram_range. This function allows you
        to update the topic representation after they have been formed.

        Arguments:
            docs: The docs you used when calling either `fit` or `fit_transform`
            topics: The topics that were returned when calling either `fit` or `fit_transform`
            n_gram_range: The n-gram range for the CountVectorizer.
            stop_words: Stopwords that can be used as either a list of strings, or the name of the
                        language as a string. For example: 'english' or ['the', 'and', 'I'].
                        Note that this will not be used if you pass in your own CountVectorizer.
            vectorizer_model: Pass in your own CountVectorizer from scikit-learn

        Usage:
        ```python
        from bertopic import BERTopic
        from sklearn.datasets import fetch_20newsgroups

        # Create topics
        docs = fetch_20newsgroups(subset='train')['data']
        model = BERTopic(n_gram_range=(1, 1), stop_words=None)
        topics, probs = model.fit_transform(docs)

        # Update topic representation
        model.update_topics(docs, topics, n_gram_range=(2, 3), stop_words="english")
        ```
        """
        check_is_fitted(self)
        if not n_gram_range:
            n_gram_range = self.n_gram_range

        if not stop_words:
            stop_words = self.stop_words

        self.vectorizer_model = vectorizer_model or CountVectorizer(ngram_range=n_gram_range,
                                                                    stop_words=stop_words)

        documents = pd.DataFrame({"Document": docs, "Topic": topics})
        self._extract_topics(documents)

    def get_topics(self) -> Mapping[str, Tuple[str, float]]:
        """ Return topics with top n words and their c-TF-IDF score

        Usage:

        ```python
        all_topics = model.get_topics()
        ```
        """
        check_is_fitted(self)
        return self.topics

    def get_topic(self, topic: int) -> Union[Mapping[str, Tuple[str, float]], bool]:
        """ Return top n words for a specific topic and their c-TF-IDF scores

        Usage:

        ```python
        topic = model.get_topic(12)
        ```
        """
        check_is_fitted(self)
        if self.topics.get(topic):
            return self.topics[topic]
        else:
            return False

    def get_topic_freq(self, topic: int = None) -> Union[pd.DataFrame, int]:
        """ Return the the size of topics (descending order)

        Usage:

        To extract the frequency of all topics:

        ```python
        frequency = model.get_topic_freq()
        ```

        To get the frequency of a single topic:

        ```python
        frequency = model.get_topic_freq(12)
        ```
        """
        check_is_fitted(self)
        if isinstance(topic, int):
            return self.topic_sizes[topic]
        else:
            return pd.DataFrame(self.topic_sizes.items(), columns=['Topic', 'Count']).sort_values("Count",
                                                                                                  ascending=False)

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

        ```python
        from bertopic import BERTopic
        from sklearn.datasets import fetch_20newsgroups

        # Create topics -> Typically over 50 topics
        docs = fetch_20newsgroups(subset='train')['data']
        model = BERTopic()
        topics, probs = model.fit_transform(docs)

        # Further reduce topics
        new_topics, new_probs = model.reduce_topics(docs, topics, probs, nr_topics=30)
        ```
        """
        check_is_fitted(self)
        self.nr_topics = nr_topics
        documents = pd.DataFrame({"Document": docs, "Topic": topics})

        # Reduce number of topics
        self._extract_topics(documents)
        documents = self._reduce_topics(documents)
        new_topics = documents.Topic.to_list()
        new_probabilities = self._map_probabilities(probabilities)

        return new_topics, new_probabilities

    def visualize_topics(self):
        """ Visualize topics, their sizes, and their corresponding words

        This visualization is highly inspired by LDAvis, a great visualization
        technique typically reserved for LDA.
        """
        check_is_fitted(self)
        if not _HAS_VIZ:
            raise ModuleNotFoundError(f"In order to use this function you'll need to install "
                                      f"additional dependencies;\npip install bertopic[visualization]")

        # Extract topic words and their frequencies
        topic_list = sorted(list(self.topics.keys()))
        frequencies = [self.topic_sizes[topic] for topic in topic_list]
        words = [" | ".join([word[0] for word in self.get_topic(topic)[:5]]) for topic in topic_list]

        # Embed c-TF-IDF into 2D
        embeddings = MinMaxScaler().fit_transform(self.c_tf_idf.toarray())
        embeddings = umap.UMAP(n_neighbors=2, n_components=2, metric='hellinger').fit_transform(embeddings)

        # Visualize with plotly
        df = pd.DataFrame({"x": embeddings[1:, 0], "y": embeddings[1:, 1],
                           "Topic": topic_list[1:], "Words": words[1:], "Size": frequencies[1:]})
        return self._plotly_topic_visualization(df, topic_list)

    def visualize_distribution(self,
                               probabilities: np.ndarray,
                               min_probability: float = 0.015,
                               figsize: tuple = (10, 5),
                               save: bool = False):
        """ Visualize the distribution of topic probabilities

        Arguments:
            probabilities: An array of probability scores
            min_probability: The minimum probability score to visualize.
                             All others are ignored.
            figsize: The size of the figure
            save: Whether to save the resulting graph to probility.png

        Usage:

        Make sure to fit the model before and only input the
        probabilities of a single document:

        ```python
        model.visualize_distribution(probabilities[0])
        ```

        ![](../img/probabilities.png)
        """
        check_is_fitted(self)
        if not _HAS_VIZ:
            raise ModuleNotFoundError(f"In order to use this function you'll need to install "
                                      f"additional dependencies;\npip install bertopic[visualization]")
        if len(probabilities[probabilities > min_probability]) == 0:
            raise ValueError("There are no values where `min_probability` is higher than the "
                             "probabilities that were supplied. Lower `min_probability` to prevent this error.")
        if self.low_memory:
            raise ValueError("This visualization cannot be used if you have set `low_memory` to True "
                             "as it uses the topic probabilities. ")

        # Get values and indices equal or exceed the minimum probability
        labels_idx = np.argwhere(probabilities >= min_probability).flatten()
        vals = probabilities[labels_idx].tolist()

        # Create labels
        labels = []
        for idx in labels_idx:
            label = []
            words = self.get_topic(idx)
            if words:
                for word in words[:5]:
                    label.append(word[0])
                label = str(r"$\bf{Topic }$ " +
                            r"$\bf{" + str(idx) + ":}$ " +
                            " ".join(label))
                labels.append(label)
            else:
                vals.remove(probabilities[idx])
        pos = range(len(vals))

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        plt.hlines(y=pos, xmin=0, xmax=vals, color='#333F4B', alpha=0.2, linewidth=15)
        plt.hlines(y=np.argmax(vals), xmin=0, xmax=max(vals), color='#333F4B', alpha=1, linewidth=15)

        # Set ticks and labels
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlabel('Probability', fontsize=15, fontweight='black', color='#333F4B')
        ax.set_ylabel('')
        plt.yticks(pos, labels)
        fig.text(0, 1, 'Topic Probability Distribution', fontsize=15, fontweight='black', color='#333F4B')

        # Update spine style
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_bounds(pos[0], pos[-1])
        ax.spines['bottom'].set_bounds(0, max(vals))
        ax.spines['bottom'].set_position(('axes', -0.02))
        ax.spines['left'].set_position(('axes', 0.02))

        fig.tight_layout()

        if save:
            fig.savefig("probability.png", dpi=300, bbox_inches='tight')

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
        model.save("my_model")
        ```

        or if you do not want the embedding_model to be saved locally:

        ```python
        model.save("my_model", save_embedding_model=False)
        ```
        """
        with open(path, 'wb') as file:
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
             embedding_model: Union[str, SentenceTransformer, DocumentEmbeddings, TokenEmbeddings] = None):
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
        BERTopic.load("my_model", embedding_model="xlm-r-bert-base-nli-stsb-mean-tokens")
        ```
        """
        with open(path, 'rb') as file:
            if embedding_model:
                topic_model = joblib.load(file)
                topic_model.embedding_model = embedding_model
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

    def _extract_embeddings(self, documents: Union[List[str], str], verbose: bool = None) -> np.ndarray:
        """ Extract sentence/document embeddings through pre-trained embeddings
        For an overview of pre-trained models: https://www.sbert.net/docs/pretrained_models.html

        Arguments:
            documents: Dataframe with documents and their corresponding IDs
            verbose: Whether to show a progressbar demonstrating the time to extract embeddings

        Returns:
            embeddings: The extracted embeddings using the sentence transformer
                        module. Typically uses pre-trained huggingface models.
        """
        if isinstance(documents, str):
            documents = [documents]

        # Infer embeddings with SentenceTransformer
        if isinstance(self.embedding_model, SentenceTransformer):
            embeddings = self.embedding_model.encode(documents, show_progress_bar=verbose)

        # Infer embeddings with Flair
        elif isinstance(self.embedding_model, DocumentEmbeddings):
            embeddings = []
            for index, document in tqdm(enumerate(documents), disable=not verbose):
                try:
                    sentence = Sentence(document) if document else Sentence("an empty document")
                    self.embedding_model.embed(sentence)
                except RuntimeError:
                    sentence = Sentence("an empty document")
                    self.embedding_model.embed(sentence)
                embedding = sentence.embedding.detach().cpu().numpy()
                embeddings.append(embedding)
            embeddings = np.asarray(embeddings)

        else:
            raise ValueError("An incorrect embedding model type was selected.")

        return embeddings

    def _map_predictions(self, predictions: List[int]) -> List[int]:
        """ Map predictions to the correct topics if topics were reduced """
        mapped_predictions = []
        for prediction in predictions:
            while self.mapped_topics.get(prediction):
                prediction = self.mapped_topics[prediction]
            mapped_predictions.append(prediction)
        return mapped_predictions

    def _reduce_dimensionality(self, embeddings: Union[np.ndarray, csr_matrix]) -> np.ndarray:
        """ Reduce dimensionality of embeddings using UMAP and train a UMAP model

        Arguments:
            embeddings: The extracted embeddings using the sentence transformer module.

        Returns:
            umap_embeddings: The reduced embeddings
        """
        if isinstance(embeddings, csr_matrix):
            self.umap_model = umap.UMAP(n_neighbors=self.n_neighbors,
                                        n_components=5,
                                        metric='hellinger',
                                        low_memory=self.low_memory).fit(embeddings)
        else:
            self.umap_model.fit(embeddings)
        umap_embeddings = self.umap_model.transform(embeddings)
        logger.info("Reduced dimensionality with UMAP")
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
        self.hdbscan_model.fit(umap_embeddings)
        documents['Topic'] = self.hdbscan_model.labels_

        if not self.low_memory:
            probabilities = hdbscan.all_points_membership_vectors(self.hdbscan_model)
        else:
            probabilities = None

        self._update_topic_size(documents)
        logger.info("Clustered UMAP embeddings with HDBSCAN")
        return documents, probabilities

    def _extract_topics(self, documents: pd.DataFrame):
        """ Extract topics from the clusters using a class-based TF-IDF

        Arguments:
            documents: Dataframe with documents and their corresponding IDs

        Returns:
            c_tf_idf: The resulting matrix giving a value (importance score) for each word per topic
        """
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        self.c_tf_idf, words = self._c_tf_idf(documents_per_topic, m=len(documents))
        self._extract_words_per_topic(words)
        self._create_topic_vectors()

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
        if not self.custom_embeddings:
            topic_list = list(self.topics.keys())
            topic_list.sort()
            n = self.top_n_words

            # Extract embeddings for all words in all topics
            topic_words = [self.get_topic(topic) for topic in topic_list]
            topic_words = [word[0] for topic in topic_words for word in topic]
            embeddings = self._extract_embeddings(topic_words, verbose=False)

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

    def _c_tf_idf(self, documents_per_topic: pd.DataFrame, m: int) -> Tuple[csr_matrix, List[str]]:
        """ Calculate a class-based TF-IDF where m is the number of total documents.

        Arguments:
            documents_per_topic: The joined documents per topic such that each topic has a single
                                 string made out of multiple documents
            m: The total number of documents (unjoined)

        Returns:
            tf_idf: The resulting matrix giving a value (importance score) for each word per topic
            words: The names of the words to which values were given
        """
        documents = self._preprocess_text(documents_per_topic.Document.values)
        count = self.vectorizer_model.fit(documents)
        words = count.get_feature_names()
        X = count.transform(documents)
        transformer = ClassTFIDF().fit(X, n_samples=m)
        c_tf_idf = transformer.transform(X)
        self.topic_sim_matrix = cosine_similarity(c_tf_idf)

        return c_tf_idf, words

    def _update_topic_size(self, documents: pd.DataFrame):
        """ Calculate the topic sizes

        Arguments:
            documents: Updated dataframe with documents and their corresponding IDs and newly added Topics
        """
        sizes = documents.groupby(['Topic']).count().sort_values("Document", ascending=False).reset_index()
        self.topic_sizes = dict(zip(sizes.Topic, sizes.Document))

    def _extract_words_per_topic(self, words: List[str]):
        """ Based on tf_idf scores per topic, extract the top n words per topic

        Arguments:
            words: List of all words (sorted according to tf_idf matrix position)
        """

        # Get top 30 words per topic based on c-TF-IDF score
        c_tf_idf = self.c_tf_idf.toarray()
        labels = sorted(list(self.topic_sizes.keys()))
        indices = c_tf_idf.argsort()[:, -30:]
        self.topics = {label: [(words[j], c_tf_idf[i][j])
                               for j in indices[i]][::-1]
                       for i, label in enumerate(labels)}

        # Extract word embeddings for the top 30 words per topic and compare it
        # with the topic embedding to keep only the words most similar to the topic embedding
        if not self.custom_embeddings:

            for topic, topic_words in self.topics.items():
                words = [word[0] for word in topic_words]
                word_embeddings = self._extract_embeddings(words, verbose=False)
                topic_embedding = self._extract_embeddings(" ".join(words), verbose=False).reshape(1, -1)

                topic_words = mmr(topic_embedding, word_embeddings, words, top_n=self.top_n_words, diversity=0)
                self.topics[topic] = [(word, value) for word, value in self.topics[topic] if word in topic_words]

    def _select_embedding_model(self) -> Union[SentenceTransformer, DocumentEmbeddings]:
        """ Select an embedding model based on language or a specific sentence transformer models.
        When selecting a language, we choose distilbert-base-nli-stsb-mean-tokens for English and
        xlm-r-bert-base-nli-stsb-mean-tokens for all other languages as it support 100+ languages.

        Returns:
            model: Either a Sentence-Transformer or Flair model
        """

        # Sentence Transformer embeddings
        if isinstance(self.embedding_model, SentenceTransformer):
            return self.embedding_model

        # Flair word embeddings
        elif _HAS_FLAIR and isinstance(self.embedding_model, TokenEmbeddings):
            return DocumentPoolEmbeddings([self.embedding_model])

        # Flair document embeddings + disable fine tune to prevent CUDA OOM
        # https://github.com/flairNLP/flair/issues/1719
        elif _HAS_FLAIR and isinstance(self.embedding_model, DocumentEmbeddings):
            if "fine_tune" in self.embedding_model.__dict__:
                self.embedding_model.fine_tune = False
            return self.embedding_model

        # Select embedding model based on specific sentence transformer model
        elif isinstance(self.embedding_model, str):
            self.sentence_pointer = self.embedding_model
            return SentenceTransformer(self.embedding_model)

        # Select embedding model based on language
        elif self.language:
            if self.language.lower() in ["English", "english", "en"]:
                return SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")

            elif self.language.lower() in languages:
                return SentenceTransformer("xlm-r-bert-base-nli-stsb-mean-tokens")

            elif self.language == "multilingual":
                return SentenceTransformer("xlm-r-bert-base-nli-stsb-mean-tokens")

            else:
                raise ValueError(f"{self.language} is currently not supported. However, you can "
                                 f"create any embeddings yourself and pass it through fit_transform(docs, embeddings)\n"
                                 "Else, please select a language from the following list:\n"
                                 f"{languages}")

        elif self.custom_embeddings:
            return None

        return SentenceTransformer("xlm-r-bert-base-nli-stsb-mean-tokens")

    def _reduce_topics(self, documents: pd.DataFrame) -> pd.DataFrame:
        """ Reduce topics to self.nr_topics

        Arguments:
            documents: Dataframe with documents and their corresponding IDs and Topics

        Returns:
            documents: Updated dataframe with documents and the reduced number of Topics
        """
        if isinstance(self.nr_topics, int):
            documents = self._reduce_to_n_topics(documents)
        elif isinstance(self.nr_topics, str):
            documents = self._auto_reduce_topics(documents)
        else:
            raise ValueError("nr_topics needs to be an int or 'auto'! ")

        return documents

    def _reduce_to_n_topics(self, documents: pd.DataFrame) -> pd.DataFrame:
        """ Reduce topics to self.nr_topics

        Arguments:
            documents: Dataframe with documents and their corresponding IDs and Topics

        Returns:
            documents: Updated dataframe with documents and the reduced number of Topics
        """
        if not self.mapped_topics:
            self.mapped_topics = {}
        initial_nr_topics = len(self.get_topics())

        # Create topic similarity matrix
        similarities = cosine_similarity(self.c_tf_idf)
        np.fill_diagonal(similarities, 0)

        while len(self.get_topic_freq()) > self.nr_topics + 1:
            # Find most similar topic to least common topic
            topic_to_merge = self.get_topic_freq().iloc[-1].Topic
            topic_to_merge_into = np.argmax(similarities[topic_to_merge + 1]) - 1
            similarities[:, topic_to_merge + 1] = -1

            # Update Topic labels
            documents.loc[documents.Topic == topic_to_merge, "Topic"] = topic_to_merge_into
            self.mapped_topics[topic_to_merge] = topic_to_merge_into

            # Update new topic content
            self._update_topic_size(documents)

        self._extract_topics(documents)

        if initial_nr_topics <= self.nr_topics:
            logger.info(f"Since {initial_nr_topics} were found, they could not be reduced to {self.nr_topics}")
        else:
            logger.info(f"Reduced number of topics from {initial_nr_topics} to {len(self.get_topic_freq())}")

        return documents

    def _auto_reduce_topics(self, documents: pd.DataFrame) -> pd.DataFrame:
        """ Reduce the number of topics as long as it exceeds a minimum similarity of 0.915

        Arguments:
            documents: Dataframe with documents and their corresponding IDs and Topics

        Returns:
            documents: Updated dataframe with documents and the reduced number of Topics
        """
        initial_nr_topics = len(self.get_topics())
        has_mapped = []
        if not self.mapped_topics:
            self.mapped_topics = {}

        # Create topic similarity matrix
        similarities = cosine_similarity(self.c_tf_idf)
        np.fill_diagonal(similarities, 0)

        # Do not map the top 10% most frequent topics
        not_mapped = int(np.ceil(len(self.get_topic_freq()) * 0.1))
        to_map = self.get_topic_freq().Topic.values[not_mapped:][::-1]

        for topic_to_merge in to_map:
            # Find most similar topic to least common topic
            similarity = np.max(similarities[topic_to_merge + 1])
            topic_to_merge_into = np.argmax(similarities[topic_to_merge + 1]) - 1

            # Only map topics if they have a high similarity
            if (similarity > 0.915) & (topic_to_merge_into not in has_mapped):
                # Update Topic labels
                documents.loc[documents.Topic == topic_to_merge, "Topic"] = topic_to_merge_into
                self.mapped_topics[topic_to_merge] = topic_to_merge_into
                similarities[:, topic_to_merge + 1] = -1

                # Update new topic content
                self._update_topic_size(documents)
                has_mapped.append(topic_to_merge)

        self._extract_topics(documents)

        logger.info(f"Reduced number of topics from {initial_nr_topics} to {len(self.get_topic_freq())}")

        return documents

    def _map_probabilities(self, probabilities: Union[np.ndarray, None]) -> Union[np.ndarray, None]:
        """ Map the probabilities to the reduced topics.
        This is achieved by adding the probabilities together
        of all topics that were mapped to the same topic. Then,
        the topics that were mapped from were set to 0 as they
        were reduced.

        Arguments:
            probabilities: An array containing probabilities

        Returns:
            probabilities: Updated probabilities

        """
        if isinstance(probabilities, np.ndarray):
            for from_topic, to_topic in self.mapped_topics.items():
                probabilities[:, to_topic] += probabilities[:, from_topic]
                probabilities[:, from_topic] = 0

            return probabilities.round(3)
        else:
            return None

    @staticmethod
    def _plotly_topic_visualization(df: pd.DataFrame,
                                    topic_list: List[str]):
        """ Create plotly-based visualization of topics with a slider for topic selection """

        def get_color(topic_selected):
            if topic_selected == -1:
                marker_color = ["#B0BEC5" for _ in topic_list[1:]]
            else:
                marker_color = ["red" if topic == topic_selected else "#B0BEC5" for topic in topic_list[1:]]
            return [{'marker.color': [marker_color]}]

        # Prepare figure range
        x_range = (df.x.min() - abs((df.x.min()) * .15), df.x.max() + abs((df.x.max()) * .15))
        y_range = (df.y.min() - abs((df.y.min()) * .15), df.y.max() + abs((df.y.max()) * .15))

        # Plot topics
        fig = px.scatter(df, x="x", y="y", size="Size", size_max=40, template="simple_white", labels={"x": "", "y": ""},
                         hover_data={"x": False, "y": False, "Topic": True, "Words": True, "Size": True})
        fig.update_traces(marker=dict(color="#B0BEC5", line=dict(width=2, color='DarkSlateGrey')))

        # Update hover order
        fig.update_traces(hovertemplate="<br>".join(["<b>Topic %{customdata[2]}</b>",
                                                     "Words: %{customdata[3]}",
                                                     "Size: %{customdata[4]}"]))

        # Create a slider for topic selection
        steps = [dict(label=f"Topic {topic}", method="update", args=get_color(topic)) for topic in topic_list[1:]]
        sliders = [dict(active=0, pad={"t": 50}, steps=steps)]

        # Stylize layout
        fig.update_layout(
            title={
                'text': "<b>Intertopic Distance Map",
                'y': .95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(
                    size=22,
                    color="Black")
            },
            width=650,
            height=650,
            hoverlabel=dict(
                bgcolor="white",
                font_size=16,
                font_family="Rockwell"
            ),
            xaxis={"visible": False},
            yaxis={"visible": False},
            sliders=sliders
        )

        # Update axes ranges
        fig.update_xaxes(range=x_range)
        fig.update_yaxes(range=y_range)

        # Add grid in a 'plus' shape
        fig.add_shape(type="line",
                      x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1],
                      line=dict(color="#CFD8DC", width=2))
        fig.add_shape(type="line",
                      x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2,
                      line=dict(color="#9E9E9E", width=2))
        fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
        fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)
        fig.data = fig.data[::-1]

        return fig

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

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     # Don't pickle embedding model
    #     if isinstance(self.embedding_model, SentenceTransformer):
    #         state["embedding_model"] = None
    #     return state


