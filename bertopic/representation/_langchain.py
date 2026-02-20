import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
from typing import Mapping, Any, Union, Callable
from bertopic.representation._base import LLMRepresentation
from bertopic.representation._prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_CHAT_PROMPT,
    DEFAULT_JSON_SCHEMA,
    DEFAULT_JSON_PROMPT,
)
from bertopic._topics import Keywords, StructuredJSON, TopicRepresentation
from bertopic._corpus import Corpus

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bertopic import BERTopic


class LangChain(LLMRepresentation):
    """Using a LangChain chat model to generate topic labels.

    Accepts any LangChain chat model (e.g. from `init_chat_model`, `ChatOpenAI`,
    `ChatAnthropic`, etc.). For structured output, uses `model.with_structured_output()`
    which automatically selects the best method (provider-native or tool-calling fallback).

    Arguments:
        model: A LangChain chat model instance (e.g. `ChatOpenAI(...)`,
               `init_chat_model("gpt-4")`, etc.).
        prompt: The prompt to be used in the model. If no prompt is given,
                `bertopic.representation._prompts.DEFAULT_CHAT_PROMPT` is used instead.
                NOTE: Use `"[KEYWORDS]"` and `"[DOCUMENTS]"` in the prompt
                to decide where the keywords and documents need to be
                inserted.
        system_prompt: The system prompt to be used in the model. If no system prompt is given,
                       `bertopic.representation._prompts.DEFAULT_SYSTEM_PROMPT` is used instead.
        json_schema: A dictionary representing the JSON schema to enforce structured output.
                     If set to True, a default schema will be used (`bertopic.representation._prompts.DEFAULT_JSON_SCHEMA`).
                     Uses LangChain's `with_structured_output()` for provider-native or
                     tool-calling structured output.
        generator_kwargs: Kwargs passed to `model.invoke()` for fine-tuning the output.
        nr_docs: The number of documents to pass to the model if a prompt
                 with the `["DOCUMENTS"]` tag is used.
        diversity: The diversity of documents to pass to the model.
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

    To use this, you will need to install the langchain package first.
    Additionally, you will need a provider package for your chosen model:

    `pip install langchain langchain-openai`

    Then, you can use the LangChain representation model as follows:

    ```python
    from langchain.chat_models import init_chat_model
    from bertopic.representation import LangChain
    from bertopic import BERTopic

    # Create your representation model
    model = init_chat_model("gpt-4")
    representation_model = LangChain(model)

    # Use the representation model in BERTopic on top of the default pipeline
    topic_model = BERTopic(representation_model=representation_model)
    ```

    You can also use a custom prompt:

    ```python
    prompt = "I have the following documents: [DOCUMENTS]. What topic do they contain?"
    representation_model = LangChain(model, prompt=prompt)
    ```

    You can also use structured output with a JSON schema:

    ```python
    representation_model = LangChain(model, json_schema=True)
    ```
    """

    def __init__(
        self,
        model,
        prompt: str | None = None,
        system_prompt: str | None = None,
        json_schema: Mapping[str, Any] | bool = False,
        generator_kwargs: Mapping[str, Any] = {},
        nr_docs: int = 4,
        diversity: float | None = None,
        doc_length: int | None = None,
        tokenizer: Union[str, Callable] | None = None,
    ):
        super().__init__(
            prompt=DEFAULT_JSON_PROMPT if json_schema else (prompt or DEFAULT_CHAT_PROMPT),
            nr_docs=nr_docs,
            diversity=diversity,
            doc_length=doc_length,
            tokenizer=tokenizer,
        )

        # LangChain specific parameters
        self.model = model
        self.system_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        self.json_schema = DEFAULT_JSON_SCHEMA if json_schema is True else json_schema
        self.generator_kwargs = generator_kwargs

    def extract_topics(
        self,
        topic_model: "BERTopic",
        corpus: Corpus,
        topic_representations: dict[int, Keywords],
        c_tf_idf: csr_matrix,
        embeddings: np.ndarray = None,
    ) -> dict[int, TopicRepresentation]:
        """Extract topics.

        Arguments:
            topic_model: A BERTopic model
            corpus: The input documents including (calculated) embeddings
            topic_representations: The candidate topic representations
            c_tf_idf: The topic c-TF-IDF representation
            embeddings: Pre-trained document embeddings (unused, for API compatibility)

        Returns:
            updated_topics: Updated topic representations
        """
        # Extract the top n representative documents per topic
        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
            c_tf_idf=c_tf_idf,
            corpus=corpus,
            nr_samples=500,
            nr_repr_docs=self.nr_docs,
            diversity=self.diversity,
        )

        # Create structured model once if json_schema is set
        llm = self.model.with_structured_output(self.json_schema) if self.json_schema else self.model

        # Generate using LangChain model
        updated_topics = {}
        for topic, docs in tqdm(repr_docs_mappings.items(), disable=not topic_model.verbose):
            prompt = self._create_prompt(
                docs=docs, topic=topic, topics=topic_representations, topic_model=topic_model
            )
            self.prompts_.append(prompt)

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            response = llm.invoke(messages, **self.generator_kwargs)

            # Update representation
            if self.json_schema:
                # with_structured_output returns a dict for JSON Schema input
                if isinstance(response, dict):
                    updated_topics[topic] = StructuredJSON(data=response)
                else:
                    updated_topics[topic] = StructuredJSON(
                        data=response.model_dump() if hasattr(response, "model_dump") else dict(response)
                    )
            else:
                updated_topics[topic] = self._parse_response(response.content.strip())

        return updated_topics
