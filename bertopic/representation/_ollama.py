from ollama import chat
from typing import Any, Callable
from bertopic.representation._base import LLMRepresentation
from bertopic.representation._prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_CHAT_PROMPT,
    DEFAULT_JSON_SCHEMA,
    DEFAULT_JSON_PROMPT,
)


class Ollama(LLMRepresentation):
    r"""Using the Ollama API to generate topic labels based on a local LLM.

    Arguments:
        model: Model to use within Ollama.
        prompt: The prompt to be used in the model. If no prompt is given,
                `bertopic.representation._prompts.DEFAULT_CHAT_PROMPT` is used instead.
                NOTE: Use `"[KEYWORDS]"` and `"[DOCUMENTS]"` in the prompt
                to decide where the keywords and documents need to be
                inserted.
        system_prompt: The system prompt to be used in the model. If no system prompt is given,
                       `bertopic.representation._prompts.DEFAULT_SYSTEM_PROMPT` is used instead.
        json_schema: A dictionary representing the JSON schema to enforce structured output.
                     If set to True, a default schema will be used (`bertopic.representation._prompts.DEFAULT_JSON_SCHEMA`).
        generator_kwargs: Kwargs passed to `ollama.chat` for fine-tuning the output.
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
        json_schema: dict[str, Any] | bool = False,
        generator_kwargs: dict[str, Any] | None = None,
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

        # Ollama specific parameters
        self.model = model
        self.system_prompt = DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt
        self.json_schema = DEFAULT_JSON_SCHEMA if json_schema is True else json_schema
        self.generator_kwargs = dict(generator_kwargs or {})
        if self.json_schema:
            self.generator_kwargs["format"] = self.json_schema

    def _generate(self, prompt: str) -> str:
        """Send one prompt to the local Ollama server."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        kwargs = {"model": self.model, "messages": messages, **self.generator_kwargs}
        return chat(**kwargs).message.content
