import time
import openai
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
from typing import Mapping, Any, Union, Callable
from bertopic.representation._base import LLMRepresentation
from bertopic.representation._utils import (
    retry_with_exponential_backoff,
)
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


class OpenAI(LLMRepresentation):
    r"""Using the OpenAI Responses API to generate topic labels.

    Uses OpenAI's recommended Responses API with Structured Outputs
    for schema-adherent JSON generation.

    For an overview see:
    https://developers.openai.com/api/docs/guides/text
    https://developers.openai.com/api/docs/guides/structured-outputs

    Arguments:
        client: A `openai.OpenAI` client
        model: Model to use within OpenAI, defaults to `"gpt-4o-mini"`.
        prompt: The prompt to be used in the model. If no prompt is given,
                `bertopic.representation._prompts.DEFAULT_CHAT_PROMPT` is used instead.
                NOTE: Use `"[KEYWORDS]"` and `"[DOCUMENTS]"` in the prompt
                to decide where the keywords and documents need to be
                inserted.
        system_prompt: The system prompt to be used in the model. If no system prompt is given,
                       `bertopic.representation._prompts.DEFAULT_SYSTEM_PROMPT` is used instead.
        json_schema: A dictionary representing the JSON schema to enforce structured output.
                     If set to True, a default schema will be used (`bertopic.representation._prompts.DEFAULT_JSON_SCHEMA`).
                     Uses OpenAI's Structured Outputs via the `text.format` parameter with
                     `json_schema` type for strict schema adherence.
        generator_kwargs: Kwargs passed to `openai.responses.create`
                          for fine-tuning the output.
        delay_in_seconds: The delay in seconds between consecutive prompts
                          in order to prevent RateLimitErrors.
        exponential_backoff: Retry requests with a random exponential backoff.
                             A short sleep is used when a rate limit error is hit,
                             then the requests is retried. Increase the sleep length
                             if errors are hit until 10 unsuccessful requests.
                             If True, overrides `delay_in_seconds`.
        nr_docs: The number of documents to pass to OpenAI if a prompt
                 with the `["DOCUMENTS"]` tag is used.
        diversity: The diversity of documents to pass to OpenAI.
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

    To use this, you will need to install the openai package first:

    `pip install openai`

    Then, get yourself an API key and use OpenAI's API as follows:

    ```python
    import openai
    from bertopic.representation import OpenAI
    from bertopic import BERTopic

    # Create your representation model
    client = openai.OpenAI(api_key=MY_API_KEY)
    representation_model = OpenAI(client)

    # Use the representation model in BERTopic on top of the default pipeline
    topic_model = BERTopic(representation_model=representation_model)
    ```

    You can also use a custom prompt:

    ```python
    prompt = "I have the following documents: [DOCUMENTS] \nThese documents are about the following topic: '"
    representation_model = OpenAI(client, prompt=prompt)
    ```

    You can also use structured output with a JSON schema:

    ```python
    representation_model = OpenAI(client, json_schema=True)
    ```
    """

    def __init__(
        self,
        client,
        model: str = "gpt-4o-mini",
        prompt: str | None = None,
        system_prompt: str | None = None,
        json_schema: Mapping[str, Any] | bool = False,
        generator_kwargs: Mapping[str, Any] = {},
        delay_in_seconds: float | None = None,
        exponential_backoff: bool = False,
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

        # OpenAI specific parameters
        self.client = client
        self.model = model
        self.system_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        self.json_schema = DEFAULT_JSON_SCHEMA if json_schema is True else json_schema
        self.delay_in_seconds = delay_in_seconds
        self.exponential_backoff = exponential_backoff
        self.generator_kwargs = generator_kwargs
        if self.json_schema:
            self.generator_kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "Topic",
                    "schema": self.json_schema,
                    "strict": True,
                }
            }

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

        # Generate using OpenAI's Responses API
        updated_topics = {}
        for topic, docs in tqdm(repr_docs_mappings.items(), disable=not topic_model.verbose):
            prompt = self._create_prompt(
                docs=docs, topic=topic, topics=topic_representations, topic_model=topic_model
            )
            self.prompts_.append(prompt)

            # Delay
            if self.delay_in_seconds:
                time.sleep(self.delay_in_seconds)

            kwargs = {
                "model": self.model,
                "instructions": self.system_prompt,
                "input": [{"role": "user", "content": prompt}],
                **self.generator_kwargs,
            }
            if self.exponential_backoff:
                response = responses_create_with_backoff(self.client, **kwargs)
            else:
                response = self.client.responses.create(**kwargs)

            response_text = response.output_text.strip()
            updated_topics[topic] = self._parse_response(response_text)

        return updated_topics


def responses_create_with_backoff(client, **kwargs):
    return retry_with_exponential_backoff(
        client.responses.create,
        errors=(openai.RateLimitError,),
    )(**kwargs)
