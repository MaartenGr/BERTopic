import json
import pandas as pd
from ollama import chat
from ollama import ChatResponse
from tqdm import tqdm
from scipy.sparse import csr_matrix
from typing import Mapping, List, Tuple, Any, Union, Callable
from bertopic.representation._base import LLMRepresentation
from bertopic.representation._utils import (
    validate_truncate_document_parameters,
)
from json.decoder import JSONDecodeError
from bertopic.representation._prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_CHAT_PROMPT, DEFAULT_JSON_SCHEMA


class Ollama(LLMRepresentation):
    r"""Using the Ollama API to generate topic labels based on a local LLM.

    Arguments:
        model: Model to use within Ollama.
        generator_kwargs: Kwargs passed to `ollama.chat` for fine-tuning the output.
        prompt: The prompt to be used in the model. If no prompt is given,
                `bertopic.representation._prompts.DEFAULT_CHAT_PROMPT` is used instead.
                NOTE: Use `"[KEYWORDS]"` and `"[DOCUMENTS]"` in the prompt
                to decide where the keywords and documents need to be
                inserted.
        system_prompt: The system prompt to be used in the model. If no system prompt is given,
                       `bertopic.representation._prompts.DEFAULT_SYSTEM_PROMPT` is used instead.
        json_schema: A dictionary representing the JSON schema to enforce structured output.
                     If set to True, a default schema will be used (`bertopic.representation._prompts.DEFAULT_JSON_SCHEMA`).
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

    Usage:

    To use this, you will need to install the ollama package first:

    `pip install ollama`

    Then, you can use the Ollama representation model as follows:

    ```python
    from bertopic.representation import Ollama
    from bertopic import BERTopic

    # Create your representation model
    representation_model = Ollama("gemma3")

    # Use the representation model in BERTopic on top of the default pipeline
    topic_model = BERTopic(representation_model=representation_model)
    ```

    You can also use a custom prompt:

    ```python
    prompt = "I have the following documents: [DOCUMENTS] \nThese documents are about the following topic: '"
    representation_model = Ollama("gemma3", prompt=prompt)
    ```
    """

    def __init__(
        self,
        model: str,
        prompt: str | None = None,
        system_prompt: str | None = None,
        json_schema: Mapping[str, Any] | bool = False,
        generator_kwargs: Mapping[str, Any] = {},
        nr_docs: int = 4,
        diversity: float | None = None,
        doc_length: int | None = None,
        tokenizer: Union[str, Callable] | None = None,
        **kwargs,
    ):
        self.model = model

        # Prompts
        self.prompt = DEFAULT_CHAT_PROMPT if prompt is None else prompt
        self.system_prompt = DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt

        # JSON Schema for structured output
        self.json_schema = DEFAULT_JSON_SCHEMA if json_schema is True else json_schema

        # Representative document extraction parameters
        self.nr_docs = nr_docs
        self.diversity = diversity

        # Document truncation
        self.doc_length = doc_length
        self.tokenizer = tokenizer
        validate_truncate_document_parameters(self.tokenizer, self.doc_length)

        # Store generator kwargs
        self.generator_kwargs = generator_kwargs
        if self.generator_kwargs.get("model"):
            self.model = generator_kwargs.get("model")
            del self.generator_kwargs["model"]

        # Store prompts for inspection
        self.prompts_ = []

    def extract_topics(
        self,
        topic_model,
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix,
        topics: Mapping[str, List[Tuple[str, float]]],
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """Extract topics.

        Arguments:
            topic_model: A BERTopic model
            documents: All input documents
            c_tf_idf: The topic c-TF-IDF representation
            topics: The candidate topics as calculated with c-TF-IDF

        Returns:
            updated_topics: Updated topic representations
        """
        # Extract the top n representative documents per topic
        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
            c_tf_idf, documents, topics, 500, self.nr_docs, self.diversity
        )

        # Generate using Ollama's Language Model
        updated_topics = {}
        for topic, docs in tqdm(repr_docs_mappings.items(), disable=not topic_model.verbose):
            prompt = self._create_prompt(docs=docs, topic=topic, topics=topics, topic_model=topic_model)
            self.prompts_.append(prompt)

            # Call Ollama API
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            kwargs = {
                "model": self.model,
                "messages": messages,
                **self.generator_kwargs,
            }
            response: ChatResponse = chat(**kwargs)

            # Update labels
            if self.json_schema:
                try:
                    label = json.loads(response.message.content)["topic_label"]
                except (JSONDecodeError, KeyError):
                    label = response.message.content.strip().replace("topic: ", "")
            else:
                label = response.message.content.strip().replace("topic: ", "")
            updated_topics[topic] = [(label, 1)]

        return updated_topics
