import time
import openai
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Mapping, List, Tuple, Any
from bertopic.representation._base import BaseRepresentation
from bertopic.representation._utils import retry_with_exponential_backoff


DEFAULT_PROMPT = """
This is a list of texts where each collection of texts describe a topic. After each collection of texts, the name of the topic they represent is mentioned as a short-highly-descriptive title
---
Topic:
Sample texts from this topic:
- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
- Meat, but especially beef, is the word food in terms of emissions.
- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

Keywords: meat beef eat eating emissions steak food health processed chicken
Topic name: Environmental impacts of eating meat
---
Topic:
Sample texts from this topic:
- I have ordered the product weeks ago but it still has not arrived!
- The website mentions that it only takes a couple of days to deliver but I still have not received mine.
- I got a message stating that I received the monitor but that is not true!
- It took a month longer to deliver than was advised...

Keywords: deliver weeks product shipping long delivery received arrived arrive week
Topic name: Shipping and delivery issues
---
Topic:
Sample texts from this topic:
[DOCUMENTS]
Keywords: [KEYWORDS]
Topic name:"""

DEFAULT_CHAT_PROMPT = """
I have a topic that contains the following documents: 
[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]

Based on the information above, extract a short topic label in the following format:
topic: <topic label>
"""


class OpenAI(BaseRepresentation):
    """ Using the OpenAI API to generate topic labels based
    on one of their Completion of ChatCompletion models. 

    The default method is `openai.Completion` if `chat=False`. 
    The prompts will also need to follow a completion task. If you 
    are looking for a more interactive chats, use `chat=True`
    with `model=gpt-3.5-turbo`. 
    
    For an overview see:
    https://platform.openai.com/docs/models

    Arguments:
        model: Model to use within OpenAI, defaults to `"text-ada-001"`.
               NOTE: If a `gpt-3.5-turbo` model is used, make sure to set
               `chat` to True.
        generator_kwargs: Kwargs passed to `openai.Completion.create`
                          for fine-tuning the output.
        prompt: The prompt to be used in the model. If no prompt is given,
                `self.default_prompt_` is used instead.
                NOTE: Use `"[KEYWORDS]"` and `"[DOCUMENTS]"` in the prompt
                to decide where the keywords and documents need to be
                inserted.
        delay_in_seconds: The delay in seconds between consecutive prompts 
                          in order to prevent RateLimitErrors. 
        exponential_backoff: Retry requests with a random exponential backoff. 
                             A short sleep is used when a rate limit error is hit, 
                             then the requests is retried. Increase the sleep length
                             if errors are hit until 10 unsuccesfull requests. 
                             If True, overrides `delay_in_seconds`.
        chat: Set this to True if a GPT-3.5 model is used.
              See: https://platform.openai.com/docs/models/gpt-3-5
        nr_docs: The number of documents to pass to OpenAI if a prompt
                 with the `["DOCUMENTS"]` tag is used.
        diversity: The diversity of documents to pass to OpenAI.
                   Accepts values between 0 and 1. A higher 
                   values results in passing more diverse documents
                   whereas lower values passes more similar documents.

    Usage:

    To use this, you will need to install the openai package first:

    `pip install openai`

    Then, get yourself an API key and use OpenAI's API as follows:

    ```python
    import openai
    from bertopic.representation import OpenAI
    from bertopic import BERTopic

    # Create your representation model
    representation_model = OpenAI(delay_in_seconds=5)

    # Use the representation model in BERTopic on top of the default pipeline
    topic_model = BERTopic(representation_model=representation_model)
    ```

    You can also use a custom prompt:

    ```python
    prompt = "I have the following documents: [DOCUMENTS] \nThese documents are about the following topic: '"
    representation_model = OpenAI(prompt=prompt, delay_in_seconds=5)
    ```

    If you want to use OpenAI's ChatGPT model:
    
    ```python
    representation_model = OpenAI(model="gpt-3.5-turbo", delay_in_seconds=10, chat=True)
    ```
    """
    def __init__(self,
                 model: str = "text-ada-001",
                 prompt: str = None,
                 generator_kwargs: Mapping[str, Any] = {},
                 delay_in_seconds: float = None,
                 exponential_backoff: bool = False,
                 chat: bool = False,
                 nr_docs: int = 4,
                 diversity: float = None
                 ):
        self.model = model
        
        if prompt is None:
            self.prompt = DEFAULT_CHAT_PROMPT if chat else DEFAULT_PROMPT
        else:
            self.prompt = prompt

        self.default_prompt_ = DEFAULT_CHAT_PROMPT if chat else DEFAULT_PROMPT
        self.delay_in_seconds = delay_in_seconds
        self.exponential_backoff = exponential_backoff
        self.chat = chat
        self.nr_docs = nr_docs
        self.diversity = diversity

        self.generator_kwargs = generator_kwargs
        if self.generator_kwargs.get("model"):
            self.model = generator_kwargs.get("model")
        if self.generator_kwargs.get("prompt"):
            del self.generator_kwargs["prompt"]
        if not self.generator_kwargs.get("stop") and not chat:
            self.generator_kwargs["stop"] = "\n"

    def extract_topics(self,
                       topic_model,
                       documents: pd.DataFrame,
                       c_tf_idf: csr_matrix,
                       topics: Mapping[str, List[Tuple[str, float]]]
                       ) -> Mapping[str, List[Tuple[str, float]]]:
        """ Extract topics

        Arguments:
            topic_model: A BERTopic model
            documents: All input documents
            c_tf_idf: The topic c-TF-IDF representation
            topics: The candidate topics as calculated with c-TF-IDF

        Returns:
            updated_topics: Updated topic representations
        """
        # Extract the top n representative documents per topic
        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(c_tf_idf, documents, topics, 500, self.nr_docs, self.diversity)

        # Generate using OpenAI's Language Model
        updated_topics = {}
        for topic, docs in repr_docs_mappings.items():
            prompt = self._create_prompt(docs, topic, topics)

            # Delay
            if self.delay_in_seconds:
                time.sleep(self.delay_in_seconds)

            if self.chat:
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                kwargs = {"model": self.model, "messages": messages, **self.generator_kwargs}
                if self.exponential_backoff:
                    response = chat_completions_with_backoff(**kwargs)
                else:
                    response = openai.ChatCompletion.create(**kwargs)
                label = response["choices"][0]["message"]["content"].strip().replace("topic: ", "")
            else:
                if self.exponential_backoff:
                    response = completions_with_backoff(model=self.model, prompt=prompt, **self.generator_kwargs)
                else:
                    response = openai.Completion.create(model=self.model, prompt=prompt, **self.generator_kwargs)
                label = response["choices"][0]["text"].strip()

            updated_topics[topic] = [(label, 1)]

        return updated_topics

    def _create_prompt(self, docs, topic, topics):
        keywords = list(zip(*topics[topic]))[0]

        # Use the Default Chat Prompt
        if self.prompt == DEFAULT_CHAT_PROMPT or self.prompt == DEFAULT_PROMPT:
            prompt = self.prompt.replace("[KEYWORDS]", " ".join(keywords))
            prompt = self._replace_documents(prompt, docs)

        # Use a custom prompt that leverages keywords, documents or both using
        # custom tags, namely [KEYWORDS] and [DOCUMENTS] respectively
        else:
            prompt = self.prompt
            if "[KEYWORDS]" in prompt:
                prompt = prompt.replace("[KEYWORDS]", " ".join(keywords))
            if "[DOCUMENTS]" in prompt:
                prompt = self._replace_documents(prompt, docs)

        return prompt

    @staticmethod
    def _replace_documents(prompt, docs):
        to_replace = ""
        for doc in docs:
            to_replace += f"- {doc[:255]}\n"
        prompt = prompt.replace("[DOCUMENTS]", to_replace)
        return prompt


def completions_with_backoff(**kwargs):
    return retry_with_exponential_backoff(openai.Completion.create, errors=(openai.error.RateLimitError,))(**kwargs)


def chat_completions_with_backoff(**kwargs):
    return retry_with_exponential_backoff(openai.ChatCompletion.create, errors=(openai.error.RateLimitError,))(**kwargs)
