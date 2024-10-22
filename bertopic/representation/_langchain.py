import pandas as pd
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from scipy.sparse import csr_matrix
from typing import Callable, Mapping, List, Tuple, Union

from bertopic.representation._base import BaseRepresentation
from bertopic.representation._utils import truncate_document


DEFAULT_PROMPT = ChatPromptTemplate.from_template(
    "What are these documents about? {DOCUMENTS} Here are keywords related to them {KEYWORDS}. Your output is a single label without any formatting."
)


class LangChain(BaseRepresentation):
    """Using chains in langchain to generate topic labels.

    You can use chains or Runnables such as those composed using the LangChain Expression Language
    as long as their schema respects the conditions defined below.

    Arguments:
        chain: The langchain chain or Runnable with a `batch` method.
               Input keys must be `DOCUMENTS` (mandatory) and `KEYWORDS` (optional).
        nr_docs: The number of documents to pass to LangChain
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
                       * If tokenizer is 'whitespace', the document is split up
                         into words separated by whitespaces. These words are counted
                         and truncated depending on `doc_length`
                       * If tokenizer is 'vectorizer', then the internal CountVectorizer
                         is used to tokenize the document. These tokens are counted
                         and truncated depending on `doc_length`. They are decoded with
                         whitespaces.
                       * If tokenizer is a callable, then that callable is used to tokenize
                         the document. These tokens are counted and truncated depending
                         on `doc_length`
        chain_config: The configuration for the langchain chain. Can be used to set options
                      like max_concurrency to avoid rate limiting errors.
    Usage:

    To use this, you will need to install the langchain package first.
    Additionally, you will need an underlying LLM to support langchain,
    like openai:

    `pip install langchain`
    `pip install langchain_openai`

    Then, you can create your chain as follows:

    ```python
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain.chains.combine_documents import create_stuff_documents_chain

    chat_model = ChatOpenAI(model=..., api_key=...)

    # For simple prompts
    prompt = ChatPromptTemplate.from_template("What are these documents about? {documents}. Here are some keywords about them {keywords} Please give a single label.")

    # For multi-message prompts
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are provided with a list of documents and are asked to provide a single label for the topic."),
            ("human", "Here is the list of documents: {documents}"),
        ]
    )

    chain = RunnablePassthrough.assign(representation=create_stuff_documents_chain(chat_model, prompt, document_variable_name="documents"))
    ```

    Finally, you can pass the chain to BERTopic as follows:

    ```python
    from bertopic.representation import LangChain

    # Create your representation model
    representation_model = LangChain(chain)

    # Use the representation model in BERTopic on top of the default pipeline
    topic_model = BERTopic(representation_model=representation_model)
    ```

    You can also use a Runnable instead of a chain.
    The example below uses the LangChain Expression Language:

    ```python
    from bertopic.representation import LangChain
    from langchain.chat_models import ChatAnthropic
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_experimental.data_anonymizer.presidio import PresidioReversibleAnonymizer

    prompt = ...

    chat_model = ...

    # We will construct a special privacy-preserving chain using Microsoft Presidio

    pii_handler = PresidioReversibleAnonymizer(analyzed_fields=["PERSON"])

    chain = (
        {
            "documents": (
                lambda inp: [
                    Document(
                        page_content=pii_handler.anonymize(
                            d.page_content,
                            language="en",
                        ),
                    )
                    for d in inp["documents"]
                ]
            ),
            "keywords": RunnablePassthrough(),
        }
        | create_stuff_documents_chain(chat_model, prompt, document_variable_name="documents")
        | (lambda output: {"representation": pii_handler.deanonymize(output["representation"])})
    )

    representation_model = LangChain(chain)
    ```
    """

    def __init__(
        self,
        chain,
        nr_docs: int = 4,
        diversity: float = None,
        doc_length: int = None,
        tokenizer: Union[str, Callable] = None,
        chain_config=None,
    ):
        self.chain = chain
        self.chain_config = chain_config
        self.nr_docs = nr_docs
        self.diversity = diversity
        self.doc_length = doc_length
        self.tokenizer = tokenizer

    def extract_topics(
        self,
        topic_model,
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix,
        topics: Mapping[str, List[Tuple[str, float]]],
    ) -> Mapping[str, List[Tuple[str, int]]]:
        """Extract topics.

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
            diversity=self.diversity,
        )

        # Generate label using langchain's batch functionality
        chain_docs: List[List[Document]] = [
            [
                Document(page_content=truncate_document(topic_model, self.doc_length, self.tokenizer, doc))
                for doc in docs
            ]
            for docs in repr_docs_mappings.values()
        ]

        # `self.chain` must take `documents` as a mandatory input key and `keywords` as an optional input key
        formatted_keywords_list = []
        for topic in topics:
            keywords = list(zip(*topics[topic]))[0]
            formatted_keywords_list.append(", ".join(keywords))

        # Documents are passed as a list of langchain Document objects, it is up to the chain to format them into a str
        inputs = [
            {"DOCUMENTS": docs, "KEYWORDS": formatted_keywords}
            for docs, formatted_keywords in zip(chain_docs, formatted_keywords_list)
        ]

        # `self.chain` must return a dict with an `representation` key
        outputs = self.chain.batch(inputs=inputs, config=self.chain_config)
        labels = [output.strip() for output in outputs]

        updated_topics = {
            topic: [(label, 1)] + [("", 0) for _ in range(9)] for topic, label in zip(repr_docs_mappings.keys(), labels)
        }

        return updated_topics
