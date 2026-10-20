import time
import openai
from typing import Any, Callable
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


class OpenAI(LLMRepresentation):
    r"""Using the OpenAI Chat Completions API to generate topic labels.

    Uses OpenAI's Chat Completions API, which OpenAI-compatible servers such as vLLM,
    LM Studio and Ollama serve as well, with Structured Outputs for schema-adherent
    JSON generation.

    For an overview see:
    https://developers.openai.com/api/docs/guides/text
    https://developers.openai.com/api/docs/guides/structured-outputs

    Arguments:
        client: A `openai.OpenAI` client
        model: Model to use within OpenAI, defaults to `"gpt-6-luna"`.
        prompt: The prompt to be used in the model. If no prompt is given,
                `bertopic.representation._prompts.DEFAULT_CHAT_PROMPT` is used instead.
                NOTE: Use `"[KEYWORDS]"` and `"[DOCUMENTS]"` in the prompt
                to decide where the keywords and documents need to be
                inserted.
        system_prompt: The system prompt to be used in the model. If no system prompt is given,
                       `bertopic.representation._prompts.DEFAULT_SYSTEM_PROMPT` is used instead.
        json_schema: A dictionary representing the JSON schema to enforce structured output.
                     If set to True, a default schema will be used (`bertopic.representation._prompts.DEFAULT_JSON_SCHEMA`).
                     Uses OpenAI's Structured Outputs via the `response_format` parameter with
                     `json_schema` type for strict schema adherence.
        generator_kwargs: Kwargs passed to `client.chat.completions.create`
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
        model: str = "gpt-6-luna",
        prompt: str | None = None,
        system_prompt: str | None = None,
        json_schema: dict[str, Any] | bool = False,
        generator_kwargs: dict[str, Any] | None = None,
        delay_in_seconds: float | None = None,
        exponential_backoff: bool = False,
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

        # OpenAI specific parameters
        self.client = client
        self.model = model
        self.system_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        self.json_schema = DEFAULT_JSON_SCHEMA if json_schema is True else json_schema
        self.delay_in_seconds = delay_in_seconds
        self.exponential_backoff = exponential_backoff
        self.generator_kwargs = dict(generator_kwargs or {})
        if self.json_schema:
            # OpenAI's strict mode only accepts schemas that allow no other properties
            self.generator_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "Topic",
                    "schema": {"additionalProperties": False, **self.json_schema},
                    "strict": True,
                },
            }

    def _generate(self, prompt: str) -> str | None:
        """Send one prompt through OpenAI's Chat Completions API."""
        if self.delay_in_seconds:
            time.sleep(self.delay_in_seconds)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        kwargs = {"model": self.model, "messages": messages, **self.generator_kwargs}
        try:
            if self.exponential_backoff:
                response = chat_completions_with_backoff(self.client, **kwargs)
            else:
                response = self.client.chat.completions.create(**kwargs)
        except openai.BadRequestError as error:
            # A prompt OpenAI refuses for its content gets no label, rather than stopping the fit (#2342)
            if error.code == "content_filter":
                return None
            raise
        return response.choices[0].message.content


def chat_completions_with_backoff(client, **kwargs):
    return retry_with_exponential_backoff(
        client.chat.completions.create,
        errors=(openai.RateLimitError,),
    )(**kwargs)
