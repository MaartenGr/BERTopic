import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from typing import Mapping, List, Tuple, Union, Callable
from langchain.docstore.document import Document
from bertopic.representation._base import BaseRepresentation
from bertopic.representation._utils import truncate_document


DEFAULT_PROMPT = "What are these documents about? Please give a single label."


class LangChain(BaseRepresentation):
    """ Using chains in langchain to generate topic labels.

    Currently, only chains from question answering is implemented. See:
    https://langchain.readthedocs.io/en/latest/modules/chains/combine_docs_examples/question_answering.html

    Arguments:
        chain: A langchain chain that has two input parameters, `input_documents` and `query`.
        prompt: The prompt to be used in the model. If no prompt is given,
                `self.default_prompt_` is used instead.
        nr_docs: The number of documents to pass to LangChain if a prompt
                 with the `["DOCUMENTS"]` tag is used.
        diversity: The diversity of documents to pass to LangChain.
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
                         and trunctated depending on `doc_length`. They are decoded with 
                         whitespaces.
                       * If tokenizer is a callable, then that callable is used to tokenize
                         the document. These tokens are counted and truncated depending
                         on `doc_length`
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
                 nr_docs: int = 4,
                 diversity: float = None,
                 doc_length: int = None,
                 tokenizer: Union[str, Callable] = None
                 ):
        self.chain = chain
        self.prompt = prompt if prompt is not None else DEFAULT_PROMPT
        self.default_prompt_ = DEFAULT_PROMPT
        self.nr_docs = nr_docs
        self.diversity = diversity
        self.doc_length = doc_length
        self.tokenizer = tokenizer

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
        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
            c_tf_idf=c_tf_idf,
            documents=documents,
            topics=topics,
            nr_samples=500,
            nr_repr_docs=self.nr_docs,
            diversity=self.diversity
        )

        # Generate label using langchain
        updated_topics = {}
        for topic, docs in tqdm(repr_docs_mappings.items(), disable=not topic_model.verbose):
            truncated_docs = [truncate_document(topic_model, self.doc_length, self.tokenizer, doc) for doc in docs]
            chain_docs = [Document(page_content=doc) for doc in truncated_docs]
            label = self.chain.run(input_documents=chain_docs, question=self.prompt).strip()
            updated_topics[topic] = [(label, 1)] + [("", 0) for _ in range(9)]

        return updated_topics
