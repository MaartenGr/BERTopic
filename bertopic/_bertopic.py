import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import numpy as np
import pandas as pd
from scipy.sparse.csr import csr_matrix
from typing import List, Tuple, Dict, Union
import matplotlib.pyplot as plt

# Models
import umap
import hdbscan
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# BERTopic
from ._ctfidf import ClassTFIDF
from ._utils import MyLogger, check_documents_type, check_embeddings_shape, check_is_fitted
from ._embeddings import languages, embedding_models


logger = MyLogger("WARNING")


class BERTopic:
    """
    BERTopic is a topic modeling technique that leverages BERT embeddings and
    c-TF-IDF to create dense clusters allowing for easily interpretable topics
    whilst keeping important words in the topic descriptions.

    Arguments:
        language: The main language used in your documents. For a full overview of supported languages
                  see bertopic.embeddings.languages
        embedding_model: Model to use. Overview of options can be found here
                        https://www.sbert.net/docs/pretrained_models.html
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
        n_components: The dimension of the space to embed into when reducing dimensionality with UMAP.
        stop_words: Stopwords that can be used as either a list of strings, or the name of the
                    language as a string. For example: 'english' or ['the', 'and', 'I'].
                    Note that this will not be used if you pass in your own CountVectorizer.
        verbose: Changes the verbosity of the model, Set to True if you want
                 to track the stages of the model.
        vectorizer: Pass in your own CountVectorizer from scikit-learn

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

    Due to the stochastisch nature of UMAP, the results from BERTopic might differ
    and the quality can degrade. Using your own embeddings allows you to
    try out BERTopic several times until you find the topics that suit
    you best.
    """
    def __init__(self,
                 language: str = "english",
                 embedding_model: str = None,
                 top_n_words: int = 20,
                 nr_topics: Union[int, str] = None,
                 n_gram_range: Tuple[int, int] = (1, 1),
                 min_topic_size: int = 15,
                 n_neighbors: int = 15,
                 n_components: int = 5,
                 stop_words: Union[str, List[str]] = None,
                 verbose: bool = False,
                 vectorizer: CountVectorizer = None):

        # Embedding model
        self.language = language
        self.embedding_model = embedding_model

        # Topic-based parameters
        self.top_n_words = top_n_words
        self.nr_topics = nr_topics
        self.min_topic_size = min_topic_size

        # Umap parameters
        self.n_neighbors = n_neighbors
        self.n_components = n_components

        # Vectorizer parameters
        self.stop_words = stop_words
        self.n_gram_range = n_gram_range
        self.vectorizer = vectorizer or CountVectorizer(ngram_range=self.n_gram_range, stop_words=self.stop_words)

        self.umap_model = None
        self.cluster_model = None
        self.topics = None
        self.topic_sizes = None
        self.reduced_topics_mapped = None
        self.mapped_topics = None
        self.topic_embeddings = None

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
                                                              np.ndarray]:
        """ Fit the models on a collection of documents, generate topics, and return the docs with topics

        Arguments:
            documents: A list of documents to fit on
            embeddings: Pre-trained document embeddings. These can be used
                        instead of the sentence-transformer model

        Returns:
            predictions: Topic predictions for each documents
            probabilities: The topic probability distribution

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
            probabilities: The topic probability distribution

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
        model = self._select_embedding_model()
        logger.info("Loaded embedding model")
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

    def _reduce_dimensionality(self, embeddings: Union[np.ndarray, csr_matrix]) -> np.ndarray:
        """ Reduce dimensionality of embeddings using UMAP and train a UMAP model

        Arguments:
            embeddings: The extracted embeddings using the sentence transformer module.

        Returns:
            umap_embeddings: The reduced embeddings
        """
        if isinstance(embeddings, csr_matrix):
            self.umap_model = umap.UMAP(n_neighbors=self.n_neighbors,
                                        n_components=self.n_components,
                                        metric='hellinger').fit(embeddings)
        else:
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
                        documents: pd.DataFrame) -> np.ndarray:
        """ Extract topics from the clusters using a class-based TF-IDF

        Arguments:
            documents: Dataframe with documents and their corresponding IDs

        Returns:
            c_tf_idf: The resulting matrix giving a value (importance score) for each word per topic
        """
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        c_tf_idf, words = self._c_tf_idf(documents_per_topic, m=len(documents))
        self._extract_words_per_topic(c_tf_idf, words)
        self._create_topic_vectors()
        return c_tf_idf

    def _create_topic_vectors(self):
        """ Creates embeddings per topics based on their topic representation

        We start by creating embeddings out of the topic representation. This
        results in a number of embeddings per topic. Then, we take the weighted
        average of embeddings in a topic by their c-TF-IDF score. This will put
        more emphasis to words that represent a topic best.
        """
        topic_list = list(self.topics.keys())
        topic_list.sort()
        n = self.top_n_words

        # Extract embeddings for all words in all topics
        topic_words = [self.get_topic(topic) for topic in topic_list]
        topic_words = [word[0] for topic in topic_words for word in topic]
        embeddings = self._extract_embeddings(topic_words)

        # Take the weighted average of word embeddings in a topic based on their c-TF-IDF value
        # The embeddings var is a single numpy matrix and therefore slicing is necessary to
        # access the words per topic
        topic_embeddings = []
        for i, topic in enumerate(topic_list):
            word_importance = [val[1] for val in self.get_topic(topic)]
            topic_embedding = np.average(embeddings[i * n: n + (i * n)], weights=word_importance, axis=0)
            topic_embeddings.append(topic_embedding)

        self.topic_embeddings = topic_embeddings

    def find_topics(self, search_term: str, top_n: int = 5) -> Tuple[List[int], List[float]]:
        """ Find topics most similar to a search_term

        Creates an embedding for search_term and compares that with
        the topic embeddings. The most similar topics are returned
        along with their similarity values.

        The search_term can be of any size but since it compares
        with the topic representation it is advised to keep it
        below 5 words.

        Args:
            search_term: the term you want to use to search for topics
            top_n: the number of topics to return

        Returns:
            similar_topics: the most similar topics from high to low
            similarity: the similarity scores from high to low

        """
        topic_list = list(self.topics.keys())
        topic_list.sort()

        # Extract search_term embeddings and compare with topic embeddings
        search_embedding = self._extract_embeddings([search_term]).flatten()
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
                      vectorizer: CountVectorizer = None):
        """ Updates the topic representation by recalculating c-TF-IDF with the new
        parameters as defined in this function.

        When you have trained a model and viewed the topics and the words that represent them,
        you might not be satisfied with the representation. Perhaps you forgot to remove
        stop_words or you want to try out a different n_gram_range. This function allows you
        to update the topic representation after they have been formed.

        Args:
            docs: The docs you used when calling either `fit` or `fit_transform`
            topics: The topics that were returned when calling either `fit` or `fit_transform`
            n_gram_range: The n-gram range for the CountVectorizer.
            stop_words: Stopwords that can be used as either a list of strings, or the name of the
                        language as a string. For example: 'english' or ['the', 'and', 'I'].
                        Note that this will not be used if you pass in your own CountVectorizer.
            vectorizer: Pass in your own CountVectorizer from scikit-learn

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
        if not n_gram_range:
            n_gram_range = self.n_gram_range

        if not stop_words:
            stop_words = self.stop_words

        self.vectorizer = vectorizer or CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words)
        documents = pd.DataFrame({"Document": docs, "Topic": topics})
        self._extract_topics(documents)

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
        count = self.vectorizer.fit(documents)
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

    def _select_embedding_model(self) -> SentenceTransformer:
        """ Select an embedding model based on language or a specific sentence transformer models.
        When selecting a language, we choose distilbert-base-nli-stsb-mean-tokens for English and
        xlm-r-bert-base-nli-stsb-mean-tokens for all other languages as it support 100+ languages.
        """

        # Select embedding model based on language
        if self.language:
            if self.language.lower() in ["English", "english", "en"]:
                return SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")

            elif self.language.lower() in languages:
                return SentenceTransformer("xlm-r-bert-base-nli-stsb-mean-tokens")

            else:
                raise ValueError(f"{self.language} is currently not supported. However, you can "
                                 f"create any embeddings yourself and pass it through fit_transform(docs, embeddings)\n"
                                 "Else, please select a language from the following list:\n"
                                 f"{languages}")

        # Select embedding model based on specific sentence transformer model
        elif self.embedding_model:
            if self.embedding_model in embedding_models:
                return SentenceTransformer(self.embedding_model)
            else:
                raise ValueError("Please select an embedding model from the following list:\n"
                                 f"{embedding_models}\n\n"
                                 f"For more information about the models, see:\n"
                                 f"https://www.sbert.net/docs/pretrained_models.html")

        return SentenceTransformer("xlm-r-bert-base-nli-stsb-mean-tokens")

    def _reduce_topics(self, documents: pd.DataFrame, c_tf_idf: np.ndarray) -> pd.DataFrame:
        """ Reduce topics to self.nr_topics

        Arguments:
            documents: Dataframe with documents and their corresponding IDs and Topics
            c_tf_idf: c-TF-IDF matrix

        Returns:
            documents: Updated dataframe with documents and the reduced number of Topics
        """
        if isinstance(self.nr_topics, int):
            documents = self._reduce_to_n_topics(c_tf_idf, documents)
        elif isinstance(self.nr_topics, str):
            documents = self._auto_reduce_topics(c_tf_idf, documents)
        else:
            raise ValueError("nr_topics needs to be an int or 'auto'! ")

        return documents

    def _reduce_to_n_topics(self, c_tf_idf, documents):
        """ Reduce topics to self.nr_topics

        Arguments:
            documents: Dataframe with documents and their corresponding IDs and Topics
            c_tf_idf: c-TF-IDF matrix

        Returns:
            documents: Updated dataframe with documents and the reduced number of Topics
        """
        if not self.mapped_topics:
            self.mapped_topics = {}
        initial_nr_topics = len(self.get_topics())

        # Create topic similarity matrix
        similarities = cosine_similarity(c_tf_idf)
        np.fill_diagonal(similarities, 0)

        while len(self.get_topics_freq()) > self.nr_topics + 1:
            # Find most similar topic to least common topic
            topic_to_merge = self.get_topics_freq().iloc[-1].Topic
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
            logger.info(f"Reduced number of topics from {initial_nr_topics} to {len(self.get_topics_freq())}")

        return documents

    def _auto_reduce_topics(self, c_tf_idf, documents):
        """ Reduce the number of topics as long as it exceeds a minimum similarity of 0.9

        Arguments:
            documents: Dataframe with documents and their corresponding IDs and Topics
            c_tf_idf: c-TF-IDF matrix

        Returns:
            documents: Updated dataframe with documents and the reduced number of Topics
        """
        initial_nr_topics = len(self.get_topics())
        has_mapped = []
        if not self.mapped_topics:
            self.mapped_topics = {}

        # Create topic similarity matrix
        similarities = cosine_similarity(c_tf_idf)
        np.fill_diagonal(similarities, 0)

        # Do not map the top 10% most frequent topics
        not_mapped = int(np.ceil(len(self.get_topics_freq()) * 0.1))
        to_map = self.get_topics_freq().Topic.values[not_mapped:][::-1]

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

        _ = self._extract_topics(documents)

        logger.info(f"Reduced number of topics from {initial_nr_topics} to {len(self.get_topics_freq())}")

        return documents

    def _map_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
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
        for from_topic, to_topic in self.mapped_topics.items():
            probabilities[:, to_topic] += probabilities[:, from_topic]
            probabilities[:, from_topic] = 0

        return probabilities.round(3)

    def get_topics(self) -> Dict[str, Tuple[str, float]]:
        """ Return topics with top n words and their c-TF-IDF score

        Usage:

        ```python
        all_topics = model.get_topics()
        ```
        """
        check_is_fitted(self)
        return self.topics

    def get_topic(self, topic: int) -> Union[Dict[str, Tuple[str, float]], bool]:
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

    def get_topics_freq(self) -> pd.DataFrame:
        """ Return the the size of topics (descending order)

        Usage:

        ```python
        frequency = model.get_topics_freq()
        ```
        """
        check_is_fitted(self)
        return pd.DataFrame(self.topic_sizes.items(), columns=['Topic', 'Count']).sort_values("Count", ascending=False)

    def get_topic_freq(self, topic: int) -> int:
        """ Return the the size of a topic

        Arguments:
             topic: the name of the topic as retrieved by get_topics

        Usage:

        ```python
        frequency = model.get_topic_freq(12)
        ```
        """
        check_is_fitted(self)
        return self.topic_sizes.items()[topic]

    def reduce_topics(self,
                      docs: List[str],
                      topics: List[int],
                      probabilities: np.ndarray,
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
            nr_topics: The number of topics you want reduced to
            probabilities: The probabilities that were returned when calling either `fit` or `fit_transform`

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
        self.nr_topics = nr_topics
        documents = pd.DataFrame({"Document": docs, "Topic": topics})

        # Reduce number of topics
        c_tf_idf = self._extract_topics(documents)
        documents = self._reduce_topics(documents, c_tf_idf)
        new_topics = documents.Topic.to_list()
        new_probabilities = self._map_probabilities(probabilities)

        return new_topics, new_probabilities

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
                print(idx, probabilities[idx])
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

    def save(self, path: str) -> None:
        """ Saves the model to the specified path

        Arguments:
            path: the location and name of the file you want to save

        Usage:

        ```python
        model.save("my_model")
        ```
        """
        with open(path, 'wb') as file:
            joblib.dump(self, file)

    @classmethod
    def load(cls, path: str):
        """ Loads the model from the specified path

        Arguments:
            path: the location and name of the BERTopic file you want to load

        Usage:

        ```python
        BERTopic.load("my_model")
        ```
        """
        with open(path, 'rb') as file:
            return joblib.load(file)
