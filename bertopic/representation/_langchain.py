import pandas as pd
from langchain_core.documents import Document
from scipy.sparse import csr_matrix
from typing import Callable, Mapping, List, Tuple, Union
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from bertopic.representation._base import BaseRepresentation
from bertopic.representation._utils import truncate_document

DEFAULT_PROMPT = """
This is a list of texts where each collection of texts describes a topic. After each collection of texts, the name of the topic they represent is mentioned as a short, highly descriptive title.
---
Topic:
Sample texts from this topic:
Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial-style meat production and factory farming, meat has become a staple food.
Meat, but especially beef, is the worst food in terms of emissions.
Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

Keywords: meat, beef, eat, eating, emissions, steak, food, health, processed, chicken
Topic name: Environmental impacts of eating meat
---
Topic:
Sample texts from this topic:
I have ordered the product weeks ago but it still has not arrived!
The website mentions that it only takes a couple of days to deliver but I still have not received mine.
I got a message stating that I received the monitor but that is not true!
It took a month longer to deliver than was advised...

Keywords: deliver, weeks, product, shipping, long, delivery, received, arrived, arrive, week
Topic name: Shipping and delivery issues
---
Topic:
Sample texts from this topic:
[DOCUMENTS]

Keywords: [KEYWORDS]
Topic name:"""


class LangChain(BaseRepresentation):
    """This representation model uses LangChain to generate descriptive topic labels.

    It supports two main usage patterns:
    1. Basic usage with a language model and optional custom prompt
    2. Advanced usage with a custom LangChain chain for full control over the generation process

    Arguments:
        llm: A LangChain text model or chat model used to generate representations, only needed for basic usage.
             Examples include ChatOpenAI or ChatAnthropic. Ignored if a custom chain is provided.
        prompt: A string template containing the placeholder [DOCUMENTS] and optionally [KEYWORDS], only needed for basic usage.
                Defaults to a pre-defined prompt defined in DEFAULT_PROMPT. Ignored if a custom chain is provided.
        chain: A custom LangChain chain to generate representations, only needed for advanced usage.
               The chain must be a LangChain Runnable that implements the batch method and accepts these input keys:
               - DOCUMENTS: (required) A list of LangChain Document objects
               - KEYWORDS: (optional) A list of topic keywords
               The chain must directly output either a string label or a list of strings.
               If provided, llm and prompt are ignored.
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
        chain_config: The configuration for the LangChain chain. Can be used to set options like max_concurrency to avoid rate limiting errors.

    Usage:

        To use this representation, you will need to install the LangChain package first.

        `pip install langchain`

        There are two ways to use the LangChain representation:

        1. Use a default LangChain chain that is created using an underlying language model and a prompt.

            You will first need to install the package for the underlying model. For example, if you want to use OpenAI:

            `pip install langchain_openai`

            ```python
            from bertopic.representation import LangChain
            from langchain_openai import ChatOpenAI

            chat_model = ChatOpenAI(temperature=0, openai_api_key=my_openai_api_key)

            # Create your representation model with the pre-defined prompt
            representation_model = LangChain(llm=chat_model)

            # Create your representation model with a custom prompt
            prompt = "What are these documents about? [DOCUMENTS] Here are keywords related to them [KEYWORDS]."
            representation_model = LangChain(llm=chat_model, prompt=prompt)

            # Use the representation model in BERTopic on top of the default pipeline
            topic_model = BERTopic(representation_model=representation_model)
            ```

        2. Use a custom LangChain chain for full control over the generation process:

            Remember that the chain will receive two inputs: `DOCUMENTS` and `KEYWORDS` and that it must return directly a string label
            or a list of strings.

            ```python
            from bertopic.representation import LangChain
            from langchain_anthropic import ChatAnthropic
            from langchain_core.documents import Document
            from langchain_core.prompts import ChatPromptTemplate
            from langchain.chains.combine_documents import create_stuff_documents_chain
            from langchain_experimental.data_anonymizer.presidio import PresidioReversibleAnonymizer

            prompt = ...

            chat_model = ...

            # We will construct a special privacy-preserving chain using Microsoft Presidio

            pii_handler = PresidioReversibleAnonymizer(analyzed_fields=["PERSON"])

            chain = (
                {
                    "DOCUMENTS": (
                        lambda inp: [
                            Document(
                                page_content=pii_handler.anonymize(
                                    d.page_content,
                                    language="en",
                                ),
                            )
                            for d in inp["DOCUMENTS"]
                        ]
                    ),
                    "KEYWORDS": lambda keywords: keywords["KEYWORDS"],
                }
                | create_stuff_documents_chain(chat_model, prompt, document_variable_name="DOCUMENTS")
            )

            representation_model = LangChain(chain=chain)
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
        self.prompt = prompt

        if chain is not None:
            self.chain = chain
        elif llm is not None:
            # Check that the prompt contains the necessary placeholder
            if "[DOCUMENTS]" not in prompt:
                raise ValueError("The prompt must contain the placeholder [DOCUMENTS]")

            # Convert prompt placeholders to the LangChain format
            langchain_prompt = prompt.replace("[DOCUMENTS]", "{DOCUMENTS}").replace("[KEYWORDS]", "{KEYWORDS}")

            # Create ChatPromptTemplate
            chat_prompt = ChatPromptTemplate.from_template(langchain_prompt)

            # Create a basic LangChain chain using create_stuff_documents_chain
            self.chain = create_stuff_documents_chain(
                llm,
                chat_prompt,
                document_variable_name="DOCUMENTS",
                document_separator="\n",
            )
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

        # Extract keywords from the topics and format them as a string
        formatted_keywords_list = []
        for topic in topics:
            keywords = list(zip(*topics[topic]))[0]
            formatted_keywords_list.append(", ".join(keywords))

        # self.chain must accept DOCUMENTS as a mandatory input key and KEYWORDS as an optional input key
        # We always pass both keys to the chain, and the chain can choose to use them or not
        # Documents are passed as a list of LangChain Document objects, it is up to the chain to format them into a string
        inputs = [
            {"DOCUMENTS": docs, "KEYWORDS": formatted_keywords}
            for docs, formatted_keywords in zip(chain_docs, formatted_keywords_list)
        ]

        # self.chain must return a string label or a list of string labels for each input
        outputs = self.chain.batch(inputs=inputs, config=self.chain_config)

        # Process outputs from the chain - can be either strings or lists of strings
        updated_topics = {}
        for topic, output in zip(repr_docs_mappings.keys(), outputs):
            # Each output can be either:
            # - A single string representing the main topic label
            # - A list of strings representing multiple related labels
            if isinstance(output, str):
                # For string output: use it as the main label (weight=1)
                # and pad with 9 empty strings (weight=0)
                labels = [(output.strip(), 1)] + [("", 0) for _ in range(9)]
            else:
                # For list output:
                # 1. Convert all elements to stripped strings
                # 2. Take up to 10 elements
                # 3. Assign decreasing weights from 1.0 to 0.1
                # 4. Pad with empty strings if needed to always have 10 elements
                clean_outputs = [str(label).strip() for label in output]
                top_labels = clean_outputs[:10]

                # Create (label, weight) pairs with decreasing weights
                labels = [(label, 1.0 - (i * 0.1)) for i, label in enumerate(top_labels)]

                # Pad with empty strings if we have less than 10 labels
                if len(labels) < 10:
                    labels.extend([("", 0.0) for _ in range(10 - len(labels))])

            updated_topics[topic] = labels

        return updated_topics
