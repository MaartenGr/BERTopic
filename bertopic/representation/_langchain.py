import pandas as pd
from scipy.sparse import csr_matrix
from typing import Mapping, List, Tuple
from langchain.docstore.document import Document
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
    def __init__(self,
                 chain,
                 prompt: str = None,
                 ):
        self.chain = chain
        self.prompt = prompt if prompt is not None else DEFAULT_PROMPT
        self.default_prompt_ = DEFAULT_PROMPT

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
        # Extract the top 4 representative documents per topic
        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(c_tf_idf, documents, topics, 500, 4)

        # Generate label using langchain
        updated_topics = {}
        for topic, docs in repr_docs_mappings.items():
            chain_docs = [Document(page_content=doc[:1000]) for doc in docs]
            label = self.chain.run(input_documents=chain_docs, question=self.prompt).strip()
            updated_topics[topic] = [(label, 1)] + [("", 0) for _ in range(9)]

        return updated_topics
