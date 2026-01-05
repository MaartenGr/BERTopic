import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from typing import Mapping, List, Tuple, Union, Callable

from bertopic.representation._prompts import DEFAULT_CHAT_PROMPT
from bertopic.representation._utils import truncate_document, validate_truncate_document_parameters


class BaseRepresentation(BaseEstimator):
    """The base representation model for fine-tuning topic representations."""

    def extract_topics(
        self,
        topic_model,
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix,
        topics: Mapping[str, List[Tuple[str, float]]],
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """Extract topics.

        Each representation model that inherits this class will have
        its arguments (topic_model, documents, c_tf_idf, topics)
        automatically passed. Therefore, the representation model
        will only have access to the information about topics related
        to those arguments.

        Arguments:
            topic_model: The BERTopic model that is fitted until topic
                         representations are calculated.
            documents: A dataframe with columns "Document" and "Topic"
                       that contains all documents with each corresponding
                       topic.
            c_tf_idf: A c-TF-IDF representation that is typically
                      identical to `topic_model.c_tf_idf_` except for
                      dynamic, class-based, and hierarchical topic modeling
                      where it is calculated on a subset of the documents.
            topics: A dictionary with topic (key) and tuple of word and
                    weight (value) as calculated by c-TF-IDF. This is the
                    default topics that are returned if no representation
                    model is used.
        """
        return topic_model.topic_representations_


class LLMRepresentation(BaseRepresentation):
    """Base class for LLM-based representation models."""

    def __init__(
        self,
        prompt: str | None = None,
        nr_docs: int = 4,
        diversity: float | None = None,
        doc_length: int | None = None,
        tokenizer: Union[str, Callable] | None = None,
    ):
        """Generate a representation model that leverages LLMs.

        Arguments:
            prompt: The prompt to be used in the model. If no prompt is given,
                    `bertopic.representation._prompts.DEFAULT_CHAT_PROMPT` is used instead.
                    NOTE: Use `"[KEYWORDS]"` and `"[DOCUMENTS]"` in the prompt
                    to decide where the keywords and documents need to be
                    inserted.
            nr_docs: The number of documents to pass to Ollama if a prompt
                    with the `["DOCUMENTS"]` tag is used.
            diversity: The diversity of documents to pass to Ollama.
                    Accepts values between 0 and 1. A higher
                    values results in passing more diverse documents
                    whereas lower values passes more similar documents.
            doc_length: The maximum length of each document. If a document is longer,
                        it will be truncated. If None, the entire document is passed.
            tokenizer: The tokenizer used to calculate to split the document into segments
                    used to count the length of a document.
                        * If tokenizer is 'char', then the document is split up
                            into characters which are counted to adhere to `doc_length`
                        * If tokenizer is 'whitespace', the document is split up
                            into words separated by whitespaces. These words are counted
                            and truncated depending on `doc_length`
                        * If tokenizer is 'vectorizer', then the internal CountVectorizer
                            is used to tokenize the document. These tokens are counted
                            and truncated depending on `doc_length`
                        * If tokenizer is a callable, then that callable is used to tokenize
                            the document. These tokens are counted and truncated depending
                            on `doc_length`
        """
        self.prompt = prompt

        # Representative document extraction parameters
        self.nr_docs = nr_docs
        self.diversity = diversity

        # Document truncation
        self.doc_length = doc_length
        self.tokenizer = tokenizer
        validate_truncate_document_parameters(self.tokenizer, self.doc_length)

        # Store prompts for inspection
        self.prompts_ = []

    def _create_prompt(
        self, docs: list[str], topic: int, topics: Mapping[str, List[Tuple[str, float]]], topic_model
    ) -> str:
        """Create prompt for LLM by either using the default prompt or replacing custom tags.
        Specifically, [KEYWORDS] and [DOCUMENTS] can be used in custom prompts to insert the topic's keywords and most representative documents.

        Arguments:
            docs: The most representative documents for a given topic.
            topic: The topic for which to create the prompt.
            topics: A dictionary with topic (key) and tuple of word and
                    weight (value) as calculated by c-TF-IDF.
            topic_model: The BERTopic model that is fitted until topic
                         representations are calculated.

        Returns:
            prompt: The created prompt.
        """
        keywords = next(zip(*topics[topic]))

        # Use the Default Chat Prompt
        if self.prompt == DEFAULT_CHAT_PROMPT:
            prompt = self.prompt.replace("[KEYWORDS]", ", ".join(keywords))
            prompt = self._replace_documents(prompt, docs, topic_model)

        # Use a custom prompt that leverages keywords, documents or both using
        # custom tags, namely [KEYWORDS] and [DOCUMENTS] respectively
        else:
            prompt = self.prompt
            if "[KEYWORDS]" in prompt:
                prompt = prompt.replace("[KEYWORDS]", ", ".join(keywords))
            if "[DOCUMENTS]" in prompt:
                prompt = self._replace_documents(prompt, docs, topic_model)

        return prompt

    def _replace_documents(self, prompt: str, docs: list[str], topic_model) -> str:
        """Replace [DOCUMENTS] tag in prompt with actual documents.

        Arguments:
            prompt: The prompt containing the [DOCUMENTS] tag.
            docs: The most representative documents for a given topic.
            topic_model: The BERTopic model that is fitted until topic
                         representations are calculated.

        Returns:
            The prompt with the [DOCUMENTS] tag replaced by actual documents.
        """
        # Truncate documents if needed
        truncated_docs = [truncate_document(topic_model, self.doc_length, self.tokenizer, doc) for doc in docs]

        # Replace tag with documents
        formatted_docs = "\n".join([f"- {doc}" for doc in truncated_docs])
        return prompt.replace("[DOCUMENTS]", formatted_docs)
