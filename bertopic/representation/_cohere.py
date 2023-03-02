import time
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Mapping, List, Tuple
from bertopic.representation._base import BaseRepresentation


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


class Cohere(BaseRepresentation):
    """ Use the Cohere API to generate topic labels based on their
    generative model.

    Find more about their models here:
    https://docs.cohere.ai/docs

    Arguments:
        client: A cohere.Client
        model: Model to use within Cohere, defaults to `"xlarge"`.
        prompt: The prompt to be used in the model. If no prompt is given,
                `self.default_prompt_` is used instead.
                NOTE: Use `"[KEYWORDS]"` and `"[DOCUMENTS]"` in the prompt
                to decide where the keywords and documents need to be
                inserted.
        delay_in_seconds: The delay in seconds between consecutive prompts 
                                in order to prevent RateLimitErrors. 

    Usage:

    To use this, you will need to install cohere first:

    `pip install cohere`

    Then, get yourself an API key and use Cohere's API as follows:

    ```python
    import cohere
    from bertopic.representation import Cohere
    from bertopic import BERTopic

    # Create your representation model
    co = cohere.Client(my_api_key)
    representation_model = Cohere(co)

    # Use the representation model in BERTopic on top of the default pipeline
    topic_model = BERTopic(representation_model=representation_model)
    ```

    You can also use a custom prompt:

    ```python
    prompt = "I have the following documents: [DOCUMENTS]. What topic do they contain?"
    representation_model = Cohere(co, prompt=prompt)
    ```
    """
    def __init__(self,
                 client,
                 model: str = "xlarge",
                 prompt: str = None,
                 delay_in_seconds: float = None,
                 ):
        self.client = client
        self.model = model
        self.prompt = prompt if prompt is not None else DEFAULT_PROMPT
        self.default_prompt_ = DEFAULT_PROMPT
        self.delay_in_seconds = delay_in_seconds

    def extract_topics(self,
                       topic_model,
                       documents: pd.DataFrame,
                       c_tf_idf: csr_matrix,
                       topics: Mapping[str, List[Tuple[str, float]]]
                       ) -> Mapping[str, List[Tuple[str, float]]]:
        """ Extract topics

        Arguments:
            topic_model: Not used
            documents: Not used
            c_tf_idf: Not used
            topics: The candidate topics as calculated with c-TF-IDF

        Returns:
            updated_topics: Updated topic representations
        """
        # Extract the top 4 representative documents per topic
        repr_docs_mappings, _, _ = topic_model._extract_representative_docs(c_tf_idf, documents, topics, 500, 4)

        # Generate using Cohere's Language Model
        updated_topics = {}
        for topic, docs in repr_docs_mappings.items():
            prompt = self._create_prompt(docs, topic, topics)

            # Delay
            if self.delay_in_seconds:
                time.sleep(self.delay_in_seconds)

            request = self.client.generate(model=self.model,
                                           prompt=prompt,
                                           max_tokens=50,
                                           num_generations=1,
                                           stop_sequences=["\n"])
            label = request.generations[0].text.strip()
            updated_topics[topic] = [(label, 1)] + [("", 0) for _ in range(9)]

        return updated_topics
    
    def _create_prompt(self, docs, topic, topics):
        keywords = list(zip(*topics[topic]))[0]

        # Use the Default Chat Prompt
        if self.prompt == self.prompt == DEFAULT_PROMPT:
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
