import time
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
from bertopic._topics import Keywords, TopicRepresentation
from bertopic._corpus import Corpus

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bertopic import BERTopic


class Cohere(LLMRepresentation):
    """Use the Cohere API to generate topic labels based on their
    generative model.

    Find more about their models here:
    https://docs.cohere.ai/docs

    Arguments:
        client: A `cohere.ClientV2`
        model: Model to use within Cohere, defaults to `"command-r"`.
        prompt: The prompt to be used in the model. If no prompt is given,
                `bertopic.representation._prompts.DEFAULT_CHAT_PROMPT` is used instead.
                NOTE: Use `"[KEYWORDS]"` and `"[DOCUMENTS]"` in the prompt
                to decide where the keywords and documents need to be
                inserted.
        system_prompt: The system prompt to be used in the model. If no system prompt is given,
                       `bertopic.representation._prompts.DEFAULT_SYSTEM_PROMPT` is used instead.
        json_schema: A dictionary representing the JSON schema to enforce structured output.
                     If set to True, a default schema will be used (`bertopic.representation._prompts.DEFAULT_JSON_SCHEMA`).
                     Uses Cohere's `response_format` with `json_object` type.
        generator_kwargs: Kwargs passed to `cohere.ClientV2.chat` for fine-tuning the output.
        delay_in_seconds: The delay in seconds between consecutive prompts
                                in order to prevent RateLimitErrors.
        nr_docs: The number of documents to pass to Cohere if a prompt
                 with the `["DOCUMENTS"]` tag is used.
        diversity: The diversity of documents to pass to Cohere.
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

    To use this, you will need to install cohere first:

    `pip install cohere`

    Then, get yourself an API key and use Cohere's API as follows:

    ```python
    import cohere
    from bertopic.representation import Cohere
    from bertopic import BERTopic

    # Create your representation model
    co = cohere.ClientV2(my_api_key)
    representation_model = Cohere(co)

    # Use the representation model in BERTopic on top of the default pipeline
    topic_model = BERTopic(representation_model=representation_model)
    ```

    You can also use a custom prompt:

    ```python
    prompt = "I have the following documents: [DOCUMENTS]. What topic do they contain?"
    representation_model = Cohere(co, prompt=prompt)
    ```

    You can also use structured output with a JSON schema:

    ```python
    representation_model = Cohere(co, json_schema=True)
    ```
    """

    def __init__(
        self,
        client,
        model: str = "command-r",
        prompt: str | None = None,
        system_prompt: str | None = None,
        json_schema: Mapping[str, Any] | bool = False,
        generator_kwargs: Mapping[str, Any] = {},
        delay_in_seconds: float | None = None,
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

        # Cohere specific parameters
        self.client = client
        self.model = model
        self.system_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        self.json_schema = DEFAULT_JSON_SCHEMA if json_schema is True else json_schema
        self.generator_kwargs = generator_kwargs
        self.delay_in_seconds = delay_in_seconds
        self.generator_kwargs["response_format"] = (
            {"type": "json_object", "schema": self.json_schema} if self.json_schema else None
        )

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

        # Generate using Cohere's Language Model
        updated_topics = {}
        for topic, docs in tqdm(repr_docs_mappings.items(), disable=not topic_model.verbose):
            prompt = self._create_prompt(
                docs=docs, topic=topic, topics=topic_representations, topic_model=topic_model
            )
            self.prompts_.append(prompt)

            # Delay
            if self.delay_in_seconds:
                time.sleep(self.delay_in_seconds)

            # Call Cohere API
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            kwargs = {
                "model": self.model,
                "messages": messages,
                **self.generator_kwargs,
            }
            response = self.client.chat(**kwargs)
            response_text = response.message.content[0].text.strip()
            updated_topics[topic] = self._parse_response(response_text)

        return updated_topics
