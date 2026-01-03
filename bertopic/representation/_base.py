import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from typing import Mapping, List, Tuple

from bertopic.representation._prompts import DEFAULT_CHAT_PROMPT
from bertopic.representation._utils import truncate_document


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
    """An LLM-based representation model for fine-tuning topic representations."""

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
