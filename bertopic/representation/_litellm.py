import time
from litellm import RateLimitError, completion
from typing import Callable
from typing import Mapping, Any
from bertopic.representation._base import LLMRepresentation
from bertopic.representation._utils import (
    retry_with_exponential_backoff,
)
from bertopic.representation._prompts import (
    DEFAULT_CHAT_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_JSON_SCHEMA,
    DEFAULT_JSON_PROMPT,
)


class LiteLLM(LLMRepresentation):
    """Using the LiteLLM API to generate topic labels.

    For an overview of models see:
    https://docs.litellm.ai/docs/providers

    Arguments:
        model: Model to use. Defaults to OpenAI's "gpt-6-luna".
        generator_kwargs: Kwargs passed to `litellm.completion`.
        prompt: The prompt to be used in the model. If no prompt is given,
                `bertopic.representation._prompts.DEFAULT_CHAT_PROMPT` is used instead.
                NOTE: Use `"[KEYWORDS]"` and `"[DOCUMENTS]"` in the prompt
                to decide where the keywords and documents need to be
                inserted.
        system_prompt: The system prompt to be used in the model. If no system prompt is given,
                       `bertopic.representation._prompts.DEFAULT_SYSTEM_PROMPT` is used instead.
        json_schema: A dictionary representing the JSON schema to enforce structured output.
                     If set to True, a default schema will be used (`bertopic.representation._prompts.DEFAULT_JSON_SCHEMA`).
                     Uses LiteLLM's `response_format` with `json_schema` type.
        delay_in_seconds: The delay in seconds between consecutive prompts
                          in order to prevent RateLimitErrors.
        exponential_backoff: Retry requests with a random exponential backoff.
                             A short sleep is used when a rate limit error is hit,
                             then the requests is retried. Increase the sleep length
                             if errors are hit until 10 unsuccesfull requests.
                             If True, overrides `delay_in_seconds`.
        nr_docs: The number of documents to pass to LiteLLM if a prompt
                 with the `["DOCUMENTS"]` tag is used.
        diversity: The diversity of documents to pass to LiteLLM.
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

    To use this, you will need to install the litellm package first:

    `pip install litellm`

    Then, get yourself an API key of any provider (for instance OpenAI) and use it as follows:

    ```python
    import os
    from bertopic.representation import LiteLLM
    from bertopic import BERTopic

    # set ENV variables
    os.environ["OPENAI_API_KEY"] = "your-openai-key"

    # Create your representation model
    representation_model = LiteLLM(model="gpt-6-luna")

    # Use the representation model in BERTopic on top of the default pipeline
    topic_model = BERTopic(representation_model=representation_model)
    ```

    You can also use a custom prompt:

    ```python
    prompt = "I have the following documents: [DOCUMENTS] \nThese documents are about the following topic: '"
    representation_model = LiteLLM(model="gpt-6-luna", prompt=prompt)
    ```

    You can also use structured output with a JSON schema:

    ```python
    representation_model = LiteLLM(model="gpt-6-luna", json_schema=True)
    ```
    """  # noqa: D301

    def __init__(
        self,
        model: str = "gpt-6-luna",
        prompt: str | None = None,
        generator_kwargs: Mapping[str, Any] | None = None,
        delay_in_seconds: float | None = None,
        exponential_backoff: bool = False,
        nr_docs: int = 4,
        diversity: float | None = None,
        doc_length: int | None = None,
        tokenizer: str | Callable | None = None,
        *,
        system_prompt: str | None = None,
        json_schema: Mapping[str, Any] | bool = False,
    ):
        super().__init__(
            prompt=DEFAULT_JSON_PROMPT if json_schema else (prompt or DEFAULT_CHAT_PROMPT),
            nr_docs=nr_docs,
            diversity=diversity,
            doc_length=doc_length,
            tokenizer=tokenizer,
        )

        # LiteLLM specific parameters
        self.model = model
        self.system_prompt = DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt
        self.json_schema = DEFAULT_JSON_SCHEMA if json_schema is True else json_schema
        self.generator_kwargs = dict(generator_kwargs or {})
        self.delay_in_seconds = delay_in_seconds
        self.exponential_backoff = exponential_backoff
        if self.json_schema:
            self.generator_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "Topic", "schema": self.json_schema},
            }

    def _generate(self, prompt: str) -> str:
        """Send one prompt through LiteLLM, retrying rate limits when asked to."""
        if self.delay_in_seconds:
            time.sleep(self.delay_in_seconds)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        kwargs = {"model": self.model, "messages": messages, **self.generator_kwargs}
        response = (
            chat_completions_with_backoff(**kwargs) if self.exponential_backoff else completion(**kwargs)
        )
        return response["choices"][0]["message"]["content"]


def chat_completions_with_backoff(**kwargs):
    return retry_with_exponential_backoff(
        completion,
        errors=(RateLimitError,),
    )(**kwargs)
