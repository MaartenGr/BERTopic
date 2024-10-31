import pandas as pd
from langchain.docstore.document import Document
from scipy.sparse import csr_matrix
from typing import Callable, Mapping, List, Tuple, Union
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from bertopic.representation._base import BaseRepresentation
from bertopic.representation._utils import truncate_document


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


class LangChain(BaseRepresentation):
    """Using chains in langchain to generate topic labels.

    The classic example uses `langchain.chains.question_answering.load_qa_chain`.
    This returns a chain that takes a list of documents and a question as input.

    You can also use Runnables such as those composed using the LangChain Expression Language.

    Arguments:
        llm: The language model to use for creating a basic langchain chain.
             This parameter is used to create a default chain if no custom chain
             is provided. If a custom chain is provided via the `chain` parameter,
             this parameter is ignored.
        prompt: A string containing placeholders `[DOCUMENTS]` and `[KEYWORDS]` that will be
                replaced with the actual documents and keywords during processing. If not provided,
                the default prompt defined in DEFAULT_PROMPT will be used. Note that the prompt is
                only used in the basic LangChain stuff documents chain (used when `llm` is provided).
        chain: A custom LangChain chain to be used instead of the basic LangChain stuff documents chain that
                is created using `llm` and `prompt`. The chain must accept the input key `DOCUMENTS` and optionally
                the input key `KEYWORDS`. It should output either a string or a list directly (not as a dict).
                If provided, `llm` and `prompt` are ignored.
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

    You can also use a Runnable instead of a chain.
    The example below uses the LangChain Expression Language:

    ```python
    from bertopic.representation import LangChain
    from langchain.chains.question_answering import load_qa_chain
    from langchain.chat_models import ChatAnthropic
    from langchain.schema.document import Document
    from langchain.schema.runnable import RunnablePassthrough
    from langchain_experimental.data_anonymizer.presidio import PresidioReversibleAnonymizer

    prompt = ...
    llm = ...

    # We will construct a special privacy-preserving chain using Microsoft Presidio

    pii_handler = PresidioReversibleAnonymizer(analyzed_fields=["PERSON"])

    chain = (
        {
            "input_documents": (
                lambda inp: [
                    Document(
                        page_content=pii_handler.anonymize(
                            d.page_content,
                            language="en",
                        ),
                    )
                    for d in inp["input_documents"]
                ]
            ),
            "question": RunnablePassthrough(),
        }
        | load_qa_chain(representation_llm, chain_type="stuff")
        | (lambda output: {"output_text": pii_handler.deanonymize(output["output_text"])})
    )

    representation_model = LangChain(chain, prompt=representation_prompt)
    ```
    """

    def __init__(
        self,
        llm: LanguageModelLike = None,
        prompt: str = DEFAULT_PROMPT,
        chain: Runnable = None,
        nr_docs: int = 4,
        diversity: float = None,
        doc_length: int = None,
        tokenizer: Union[str, Callable] = None,
        chain_config: dict = None,
    ):
        if chain is not None:
            self.chain = chain
        elif llm is not None:
            # Convert prompt placeholders to the LangChain format
            langchain_prompt = prompt.replace("[DOCUMENTS]", "{DOCUMENTS}").replace("[KEYWORDS]", "{KEYWORDS}")

            # Create ChatPromptTemplate
            chat_prompt = ChatPromptTemplate.from_template(langchain_prompt)

            # Create a basic LangChain chain using create_stuff_documents_chain
            self.chain = create_stuff_documents_chain(llm, chat_prompt, document_variable_name="DOCUMENTS")
        else:
            raise ValueError("Either `llm` or `chain` must be provided")

        self.nr_docs = nr_docs
        self.diversity = diversity
        self.doc_length = doc_length
        self.tokenizer = tokenizer
        self.chain_config = chain_config

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

        # `self.chain` must take `input_documents` and `question` as input keys
        # Use a custom prompt that leverages keywords, using the tag: [KEYWORDS]
        if "[KEYWORDS]" in self.prompt:
            prompts = []
            for topic in topics:
                keywords = list(zip(*topics[topic]))[0]
                prompt = self.prompt.replace("[KEYWORDS]", ", ".join(keywords))
                prompts.append(prompt)

            inputs = [{"input_documents": docs, "question": prompt} for docs, prompt in zip(chain_docs, prompts)]

        else:
            inputs = [{"input_documents": docs, "question": self.prompt} for docs in chain_docs]

        # `self.chain` must return a dict with an `output_text` key
        # same output key as the `StuffDocumentsChain` returned by `load_qa_chain`
        outputs = self.chain.batch(inputs=inputs, config=self.chain_config)
        labels = [output["output_text"].strip() for output in outputs]

        updated_topics = {
            topic: [(label, 1)] + [("", 0) for _ in range(9)] for topic, label in zip(repr_docs_mappings.keys(), labels)
        }

        return updated_topics
