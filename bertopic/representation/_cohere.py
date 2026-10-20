import time
from typing import Any, Callable
from bertopic.representation._base import LLMRepresentation
from bertopic.representation._prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_CHAT_PROMPT,
    DEFAULT_JSON_SCHEMA,
    DEFAULT_JSON_PROMPT,
)


class Cohere(LLMRepresentation):
    """Use the Cohere API to generate topic labels based on their
    generative model.

    Find more about their models here:
    https://docs.cohere.ai/docs

    Arguments:
        client: A `cohere.ClientV2`
        model: Model to use within Cohere, defaults to `"command-r7b-12-2024"`.
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
        model: str = "command-r7b-12-2024",
        prompt: str | None = None,
        system_prompt: str | None = None,
        json_schema: dict[str, Any] | bool = False,
        generator_kwargs: dict[str, Any] | None = None,
        delay_in_seconds: float | None = None,
        nr_docs: int = 4,
        diversity: float | None = None,
        doc_length: int | None = None,
        tokenizer: str | Callable | None = None,
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
        self.generator_kwargs = dict(generator_kwargs or {})
        self.delay_in_seconds = delay_in_seconds
        if self.json_schema:
            self.generator_kwargs["response_format"] = {"type": "json_object", "schema": self.json_schema}

    def _generate(self, prompt: str) -> str:
        """Send one prompt to Cohere's chat API."""
        if self.delay_in_seconds:
            time.sleep(self.delay_in_seconds)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        kwargs = {"model": self.model, "messages": messages, **self.generator_kwargs}
        return self.client.chat(**kwargs).message.content[0].text
