import numpy as np

import spacy
from spacy.matcher import Matcher
from spacy.language import Language

from packaging import version
from scipy.sparse import csr_matrix
from sklearn import __version__ as sklearn_version
from bertopic.representation._base import BaseRepresentation
from bertopic._topics import Keywords
from bertopic._corpus import Corpus

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bertopic import BERTopic


class PartOfSpeech(BaseRepresentation):
    """Extract Topic Keywords based on their Part-of-Speech.

    DEFAULT_PATTERNS = [
                [{'POS': 'ADJ'}, {'POS': 'NOUN'}],
                [{'POS': 'NOUN'}],
                [{'POS': 'ADJ'}]
    ]

    From candidate topics, as extracted with c-TF-IDF,
    find documents that contain keywords found in the
    candidate topics. These candidate documents then
    serve as the representative set of documents from
    which the Spacy model can extract a set of candidate
    keywords for each topic.

    These candidate keywords are first judged by whether
    they fall within the DEFAULT_PATTERNS or the user-defined
    pattern. Then, the resulting keywords are sorted by
    their respective c-TF-IDF values.

    Arguments:
        model: The Spacy model to use
        top_n_words: The top n words to extract
        pos_patterns: Patterns for Spacy to use.
                      See https://spacy.io/usage/rule-based-matching

    Usage:

    ```python
    from bertopic.representation import PartOfSpeech
    from bertopic import BERTopic

    # Create your representation model
    representation_model = PartOfSpeech("en_core_web_sm")

    # Use the representation model in BERTopic on top of the default pipeline
    topic_model = BERTopic(representation_model=representation_model)
    ```

    You can define custom POS patterns to be extracted:

    ```python
    pos_patterns = [
                [{'POS': 'ADJ'}, {'POS': 'NOUN'}],
                [{'POS': 'NOUN'}], [{'POS': 'ADJ'}]
    ]
    representation_model = PartOfSpeech("en_core_web_sm", pos_patterns=pos_patterns)
    ```
    """

    def __init__(
        self,
        model: str | Language = "en_core_web_sm",
        top_n_words: int = 10,
        pos_patterns: list[str] | None = None,
    ):
        if isinstance(model, str):
            self.model = spacy.load(model)
        elif isinstance(model, Language):
            self.model = model
        else:
            raise ValueError(
                "Make sure that the Spacy model that you"
                "pass is either a string referring to a"
                "Spacy model or a Spacy nlp object."
            )

        self.top_n_words = top_n_words

        if pos_patterns is None:
            self.pos_patterns = [
                [{"POS": "ADJ"}, {"POS": "NOUN"}],
                [{"POS": "NOUN"}],
                [{"POS": "ADJ"}],
            ]
        else:
            self.pos_patterns = pos_patterns

    def extract_topics(
        self,
        topic_model: "BERTopic",
        corpus: Corpus,
        topic_representations: dict[int, Keywords],
        c_tf_idf: csr_matrix,
        embeddings: np.ndarray = None,
    ) -> dict[int, Keywords]:
        """Extract topics.

        Arguments:
            topic_model: A BERTopic model
            corpus: The input documents including (calculated) embeddings
            topic_representations: The candidate topic representations
            c_tf_idf: Not used
            embeddings: Not used

        Returns:
            updated_topics: Updated topic representations
        """
        matcher = Matcher(self.model.vocab)
        matcher.add("Pattern", self.pos_patterns)

        # Build a lookup of documents per topic for efficient filtering
        docs_by_topic = {}
        for doc, topic_id in zip(corpus.documents, corpus.topics):
            docs_by_topic.setdefault(topic_id, []).append(doc)

        candidate_topics = {}
        for topic, keywords_repr in topic_representations.items():
            keywords = keywords_repr.words

            # Extract candidate documents
            candidate_documents = []
            topic_docs = docs_by_topic.get(topic, [])
            for keyword in keywords:
                selection = [doc for doc in topic_docs if keyword in doc]
                candidate_documents.extend(selection[:2])
            candidate_documents = list(set(candidate_documents))

            # Extract keywords
            docs_pipeline = self.model.pipe(candidate_documents)
            updated_keywords = []
            for doc in docs_pipeline:
                matches = matcher(doc)
                for _, start, end in matches:
                    updated_keywords.append(doc[start:end].text)
            candidate_topics[topic] = list(set(updated_keywords))

        # Scikit-Learn Deprecation: get_feature_names is deprecated in 1.0
        # and will be removed in 1.2. Please use get_feature_names_out instead.
        if version.parse(sklearn_version) >= version.parse("1.0.0"):
            words = list(topic_model.vectorizer_model.get_feature_names_out())
        else:
            words = list(topic_model.vectorizer_model.get_feature_names())

        # Match updated keywords with c-TF-IDF values
        words_lookup = dict(zip(words, range(len(words))))
        updated_topics = {}

        for topic, candidate_keywords in candidate_topics.items():
            word_indices = np.sort(
                [words_lookup.get(keyword) for keyword in candidate_keywords if keyword in words_lookup]
            )
            vals = topic_model.c_tf_idf_[:, word_indices][topic + corpus._outliers]
            indices = np.argsort(np.array(vals.todense().reshape(1, -1))[0])[-self.top_n_words :][::-1]
            vals = np.sort(np.array(vals.todense().reshape(1, -1))[0])[-self.top_n_words :][::-1]
            topic_words = [(words[word_indices[index]], val) for index, val in zip(indices, vals)]
            if len(topic_words) < self.top_n_words:
                topic_words += [("", 0) for _ in range(self.top_n_words - len(topic_words))]
            updated_topics[topic] = Keywords(topic_words)

        return updated_topics
