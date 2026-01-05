import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from llama_cpp import Llama
from typing import Mapping, List, Tuple, Any, Union, Callable
from bertopic.representation._base import LLMRepresentation
from bertopic.representation._prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_CHAT_PROMPT


class LlamaCPP(LLMRepresentation):
    """A llama.cpp implementation to use as a representation model.

    Arguments:
        model: Either a string pointing towards a local LLM or a
                `llama_cpp.Llama` object.
        prompt: The prompt to be used in the model. If no prompt is given,
                `bertopic.representation._prompts.DEFAULT_CHAT_PROMPT` is used instead.
                NOTE: Use `"[KEYWORDS]"` and `"[DOCUMENTS]"` in the prompt
                to decide where the keywords and documents need to be
                inserted.
        system_prompt: The system prompt to be used in the model. If no system prompt is given,
                       `bertopic.representation._prompts.DEFAULT_SYSTEM_PROMPT` is used instead.
        pipeline_kwargs: Kwargs that you can pass to the `llama_cpp.Llama`
                         when it is called such as `max_tokens` to be generated.
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
                       * If tokenizer is 'whitespace', the the document is split up
                         into words separated by whitespaces. These words are counted
                         and truncated depending on `doc_length`
                       * If tokenizer is 'vectorizer', then the internal CountVectorizer
                         is used to tokenize the document. These tokens are counted
                         and truncated depending on `doc_length`
                       * If tokenizer is a callable, then that callable is used to tokenize
                         the document. These tokens are counted and truncated depending
                         on `doc_length`

    Usage:

    To use a llama.cpp, first download the LLM:

    ```bash
    wget https://huggingface.co/TheBloke/zephyr-7B-alpha-GGUF/resolve/main/zephyr-7b-alpha.Q4_K_M.gguf
    ```

    Then, we can now use the model the model with BERTopic in just a couple of lines:

    ```python
    from bertopic import BERTopic
    from bertopic.representation import LlamaCPP

    # Use llama.cpp to load in a 4-bit quantized version of Zephyr 7B Alpha
    representation_model = LlamaCPP("zephyr-7b-alpha.Q4_K_M.gguf")

    # Create our BERTopic model
    topic_model = BERTopic(representation_model=representation_model, verbose=True)
    ```

    If you want to have more control over the LLMs parameters, you can run it like so:

    ```python
    from bertopic import BERTopic
    from bertopic.representation import LlamaCPP
    from llama_cpp import Llama

    # Use llama.cpp to load in a 4-bit quantized version of Zephyr 7B Alpha
    llm = Llama(model_path="zephyr-7b-alpha.Q4_K_M.gguf", n_gpu_layers=-1, n_ctx=4096, stop="Q:")
    representation_model = LlamaCPP(llm)

    # Create our BERTopic model
    topic_model = BERTopic(representation_model=representation_model, verbose=True)
    ```
    """

    def __init__(
        self,
        model: Union[str, Llama],
        prompt: str | None = None,
        system_prompt: str | None = None,
        pipeline_kwargs: Mapping[str, Any] = {},
        nr_docs: int = 4,
        diversity: float | None = None,
        doc_length: int | None = None,
        tokenizer: Union[str, Callable] | None = None,
    ):
        super().__init__(
            prompt=prompt if prompt is not None else DEFAULT_CHAT_PROMPT,
            nr_docs=nr_docs,
            diversity=diversity,
            doc_length=doc_length,
            tokenizer=tokenizer,
        )

        # Llama.cpp specific parameters
        if isinstance(model, str):
            self.model = Llama(model_path=model, n_gpu_layers=-1, stop="\n", chat_format="ChatML")
        elif isinstance(model, Llama):
            self.model = model
        else:
            raise ValueError(
                "Make sure that the model that you"
                "pass is either a string referring to a"
                "local LLM or a ` llama_cpp.Llama` object."
            )
        self.system_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        self.pipeline_kwargs = pipeline_kwargs

    def extract_topics(
        self,
        topic_model,
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix,
        topics: Mapping[str, List[Tuple[str, float]]],
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """Extract topic representations and return a single label.

        Arguments:
            topic_model: A BERTopic model
            documents: Not used
            c_tf_idf: Not used
            topics: The candidate topics as calculated with c-TF-IDF

        Returns:
            updated_topics: Updated topic representations
        """
        # Extract the top 4 representative documents per topic
        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
            c_tf_idf, documents, topics, 500, self.nr_docs, self.diversity
        )

        updated_topics = {}
        for topic, docs in tqdm(repr_docs_mappings.items(), disable=not topic_model.verbose):
            # Prepare prompt
            prompt = self._create_prompt(docs=docs, topic=topic, topics=topics, topic_model=topic_model)
            self.prompts_.append(prompt)

            # Extract result from generator and use that as label
            # topic_description = self.model(prompt, **self.pipeline_kwargs)["choices"]
            topic_description = self.model.create_chat_completion(
                messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}],
                **self.pipeline_kwargs,
            )
            label = topic_description["choices"][0]["message"]["content"].strip()
            updated_topics[topic] = [(label, 1)] + [("", 0) for _ in range(9)]

        return updated_topics
