import json
from typing import Any, Callable
from bertopic.representation._base import LLMRepresentation
from bertopic.representation._prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_CHAT_PROMPT,
    DEFAULT_JSON_SCHEMA,
    DEFAULT_JSON_PROMPT,
)


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
        json_schema: dict[str, Any] | bool = False,
        generator_kwargs: dict[str, Any] = {},
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

        # LangChain specific parameters
        self.model = model
        self.system_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        self.json_schema = DEFAULT_JSON_SCHEMA if json_schema is True else json_schema
        self.generator_kwargs = generator_kwargs

    def _generate(self, prompt: str) -> str:
        """Send one prompt to the LangChain chat model."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        # A JSON schema is answered with a dict, or with a pydantic model when the schema is one
        if self.json_schema:
            structured_model = self.model.with_structured_output(self.json_schema)
            response = structured_model.invoke(messages, **self.generator_kwargs)
            return json.dumps(response if isinstance(response, dict) else response.model_dump(mode="json"))

        # Some models answer with a list of blocks, such as reasoning followed by text
        response = self.model.invoke(messages, **self.generator_kwargs)
        return response.content if isinstance(response.content, str) else response.text
