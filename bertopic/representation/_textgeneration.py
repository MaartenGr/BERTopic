from transformers import pipeline, set_seed
from transformers.pipelines.base import Pipeline
from typing import Any, Callable, Union
from bertopic.representation._base import LLMRepresentation
from bertopic.representation._prompts import DEFAULT_COMPLETION_PROMPT


class TextGeneration(LLMRepresentation):
    """Text generation with transformers.

    Arguments:
        model: A transformers pipeline that should be initialized as "text-generation",
               for example `pipeline('text-generation', model='google/gemma-4-E2B-it')`. If a string
               is passed, a "text-generation" pipeline is created for it. A model with
               a chat template, as instruction-tuned models have, is sent the prompt as
               a chat message, so its template is applied and only its answer is kept.
        prompt: The prompt to be used in the model. If no prompt is given,
                `bertopic.representation._prompts.DEFAULT_CHAT_PROMPT` is used for a model
                with a chat template, and `DEFAULT_COMPLETION_PROMPT` for any other model.
                NOTE: Use `"[KEYWORDS]"` and `"[DOCUMENTS]"` in the prompt
                to decide where the keywords and documents need to be
                inserted.
        pipeline_kwargs: Kwargs that you can pass to the transformers.pipeline
                         when it is called.
        random_state: A random state to be passed to `transformers.set_seed`
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

    To use a model from the Hugging Face Hub:

    ```python
    from bertopic.representation import TextGeneration
    from bertopic import BERTopic

    # Create your representation model
    representation_model = TextGeneration('google/gemma-4-E2B-it')

    # Use the representation model in BERTopic on top of the default pipeline
    topic_model = BERTopic(representation_model=representation_model)
    ```

    You can use a custom prompt and decide where the keywords should
    be inserted by using the `[KEYWORDS]` or documents with the `[DOCUMENTS]` tag:

    ```python
    from transformers import pipeline
    from bertopic.representation import TextGeneration

    prompt = (
        "I have a topic described by the following keywords: [KEYWORDS]. "
        "Give a short label for this topic, and nothing else."
    )

    # Create your representation model
    generator = pipeline('text-generation', model='google/gemma-4-E2B-it')
    representation_model = TextGeneration(generator, prompt=prompt)
    ```
    """

    def __init__(
        self,
        model: Union[str, pipeline],
        prompt: str | None = None,
        pipeline_kwargs: dict[str, Any] = {},
        random_state: int = 42,
        nr_docs: int = 4,
        diversity: float | None = None,
        doc_length: int | None = None,
        tokenizer: str | Callable | None = None,
    ):
        super().__init__(
            prompt=prompt,
            nr_docs=nr_docs,
            diversity=diversity,
            doc_length=doc_length,
            tokenizer=tokenizer,
        )

        # Transformer specific parameters
        self.random_state = random_state
        set_seed(random_state)

        if isinstance(model, str):
            self.model = pipeline("text-generation", model=model)
        elif isinstance(model, Pipeline):
            self.model = model
        else:
            raise ValueError(
                "Make sure that the HF model that you"
                "pass is either a string referring to a"
                "HF model or a `transformers.pipeline` object."
            )
        self.pipeline_kwargs = pipeline_kwargs

        # A base model without a chat template completes a short prompt instead, as on 0.17
        if prompt is None and not self.model.tokenizer.chat_template:
            self.prompt = DEFAULT_COMPLETION_PROMPT

    def _generate(self, prompt: str) -> str | None:
        """Ask the transformers pipeline, as a chat when the model has a chat template."""
        # A chat model answers a conversation, which the pipeline writes in the model's own chat template
        if self.model.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]

            # The answer is a message only within the full conversation, and an empty one has no content
            output = self.model(messages, **{**self.pipeline_kwargs, "return_full_text": True})
            return output[0]["generated_text"][-1].get("content")

        # Any other model continues the prompt
        return self.model(prompt, **self.pipeline_kwargs)[0]["generated_text"].replace(prompt, "")
