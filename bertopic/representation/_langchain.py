import pandas as pd
from langchain.docstore.document import Document
from langchain.schema.runnable import Runnable, RunnableConfig
from scipy.sparse import csr_matrix
from typing import Any, Dict, Mapping, List, Optional, Tuple, Union

from bertopic.representation._base import BaseRepresentation

DEFAULT_PROMPT = "What are these documents about? Please give a single label."


class LangChain(BaseRepresentation):
    """ Using chains in langchain to generate topic labels.

    Currently, only chains from question answering is implemented. See:
    https://langchain.readthedocs.io/en/latest/modules/chains/combine_docs_examples/question_answering.html

    Arguments:
        chain: A langchain chain that has two input parameters, `input_documents` and `query`.
        prompt: The prompt to be used in the model. If no prompt is given,
                `self.default_prompt_` is used instead.

    Usage:

    To use this, you will need to install the langchain package first.
    Additionally, you will need an underlying LLM to support langchain,
    like openai:

    `pip install langchain`
    `pip install openai`

    Then, you can create your chain as follows:

    ```python
    from langchain.chains.question_answering import load_qa_chain
    from langchain.llms import OpenAI
    chain = load_qa_chain(OpenAI(temperature=0, openai_api_key=my_openai_api_key), chain_type="stuff")
    ```

    Finally, you can pass the chain to BERTopic as follows:

    ```python
    from bertopic.representation import LangChain

    # Create your representation model
    representation_model = LangChain(chain)

    # Use the representation model in BERTopic on top of the default pipeline
    topic_model = BERTopic(representation_model=representation_model)
    ```

    You can also use a custom prompt:

    ```python
    prompt = "What are these documents about? Please give a single label."
    representation_model = LangChain(chain, prompt=prompt)
    ```
    """
    # TODO: update docstring
    def __init__(self,
                 chain: Runnable,
                 prompt: Optional[str] = None,
                 config: Union[RunnableConfig, Dict[str, Any], None] = None,
                 nr_samples: int = 500,
                 nr_repr_docs: int = 4,
                 diversity: Optional[float] = None,
                 truncate_len: Union[int, None] = 1000,
                 ):
        self.chain = chain
        self.prompt = prompt if prompt is not None else DEFAULT_PROMPT
        self.default_prompt_ = DEFAULT_PROMPT
        self.config = config
        self.nr_samples = nr_samples
        self.nr_repr_docs = nr_repr_docs
        self.diversity = diversity
        self.truncate_len = truncate_len

    def extract_topics(self,
                       topic_model,
                       documents: pd.DataFrame,
                       c_tf_idf: csr_matrix,
                       topics: Mapping[str, List[Tuple[str, float]]]
                       ) -> Mapping[str, List[Tuple[str, int]]]:
        """ Extract topics

        Arguments:
            topic_model: A BERTopic model
            documents: All input documents
            c_tf_idf: The topic c-TF-IDF representation
            topics: The candidate topics as calculated with c-TF-IDF

        Returns:
            updated_topics: Updated topic representations
        """
        # Extract the top `nr_repr_docs` representative documents per topic
        # Use all documents if `nr_repr_docs` is None
        nr_repr_docs = self.nr_repr_docs if self.nr_repr_docs is not None else len(documents)
        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
            c_tf_idf=c_tf_idf,
            documents=documents,
            topics=topics,
            nr_samples=self.nr_samples,
            nr_repr_docs=nr_repr_docs,
            diversity=self.diversity,
        )

        # Generate label using langchain's batch functionality
        chain_docs: List[List[Document]] = [
            [
                Document(
                    page_content=doc if self.truncate_len is None else doc[:self.truncate_len]
                )
                for doc in docs
            ]
            for docs in repr_docs_mappings.values()
        ]

        # `self.chain` must take `input_documents` and `question` as input keys
        inputs: List[Dict[str, Union[List[str], str]]] = [
            {"input_documents": docs, "question": self.prompt}
            for docs in chain_docs
        ]

        # `self.chain` must return a dict with an `output_text` key
        outputs: List[Dict[str, str]] = self.chain.batch(inputs=inputs, config=self.config)
        # same output key as the `StuffDocumentsChain` returned by `load_qa_chain`
        labels: List[str] = [output["output_text"].strip() for output in outputs]

        updated_topics = {
            topic: [(label, 1)] + [("", 0) for _ in range(9)]
            for topic, label in zip(repr_docs_mappings.keys(), labels)
        }

        return updated_topics
