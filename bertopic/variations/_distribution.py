"""Approximate topic distributions across documents.
See: https://maartengr.github.io/BERTopic/getting_started/distribution/distribution.html.
"""

import math
import numpy as np
from typing import TYPE_CHECKING
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from bertopic._utils import MyLogger

if TYPE_CHECKING:
    from bertopic import BERTopic

logger = MyLogger()
logger.configure("WARNING")


def approximate_distribution(
    topic_model: "BERTopic",
    documents: str | list[str],
    window: int = 4,
    stride: int = 1,
    min_similarity: float = 0.1,
    batch_size: int = 1000,
    padding: bool = False,
    use_embedding_model: bool = False,
    calculate_tokens: bool = False,
    separator: str = " ",
) -> tuple[np.ndarray, list[np.ndarray] | None]:
    """A post-hoc approximation of topic distributions across documents.

    In order to perform this approximation, each document is split into tokens
    according to the provided tokenizer in the `CountVectorizer`. Then, a
    sliding window is applied on each document creating subsets of the document.
    For example, with a window size of 3 and stride of 1, the sentence:

    `Solving the right problem is difficult.`

    can be split up into `solving the right`, `the right problem`, `right problem is`,
    and `problem is difficult`. These are called tokensets. For each of these
    tokensets, we calculate their c-TF-IDF representation and find out
    how similar they are to the previously generated topics. Then, the
    similarities to the topics for each tokenset are summed up in order to
    create a topic distribution for the entire document.

    We can also dive into this a bit deeper by then splitting these tokensets
    up into individual tokens and calculate how much a word, in a specific sentence,
    contributes to the topics found in that document. This can be enabled by
    setting `calculate_tokens=True` which can be used for visualization purposes
    in `topic_model.visualize_approximate_distribution`.

    The main output, `topic_distributions`, can also be used directly in
    `.visualize_distribution(topic_distributions[index])` by simply selecting
    a single distribution.

    Arguments:
        topic_model: The BERTopic instance
        documents: A single document or a list of documents for which we
                    approximate their topic distributions
        window: Size of the moving window which indicates the number of
                tokens being considered.
        stride: How far the window should move at each step.
        min_similarity: The minimum similarity of a document's tokenset
                        with respect to the topics.
        batch_size: The number of documents to process at a time. If None,
                    then all documents are processed at once.
                    NOTE: With a large number of documents, it is not
                    advised to process all documents at once.
        padding: Whether to pad the beginning and ending of a document with
                    empty tokens.
        use_embedding_model: Whether to use the topic model's embedding
                                model to calculate the similarity between
                                tokensets and topics instead of using c-TF-IDF.
        calculate_tokens: Calculate the similarity of tokens with all topics.
                            NOTE: This is computation-wise more expensive and
                            can require more memory. Using this over batches of
                            documents might be preferred.
        separator: The separator used to merge tokens into tokensets.

    Returns:
        topic_distributions: A `n` x `m` matrix containing the topic distributions
                                for all input documents with `n` being the documents
                                and `m` the topics.
        topic_token_distributions: A list of `t` x `m` arrays with `t` being the
                                    number of tokens for the respective document
                                    and `m` the topics.

    Examples:
    After fitting the model, the topic distributions can be calculated regardless
    of the clustering model and regardless of whether the documents were previously
    seen or not:

    ```python
    topic_distr, _ = topic_model.approximate_distribution(docs)
    ```

    As a result, the topic distributions are calculated in `topic_distr` for the
    entire document based on a token set with a specific window size and stride.

    If you want to calculate the topic distributions on a token-level:

    ```python
    topic_distr, topic_token_distr = topic_model.approximate_distribution(docs, calculate_tokens=True)
    ```

    The `topic_token_distr` then contains, for each token, the best fitting topics.
    As with `topic_distr`, it can contain multiple topics for a single token.
    """
    if isinstance(documents, str):
        documents = [documents]

    if batch_size is None:
        batch_size = len(documents)
        batches = 1
    else:
        batches = math.ceil(len(documents) / batch_size)

    topic_distributions = []
    topic_token_distributions = []

    for i in tqdm(range(batches), disable=not topic_model.verbose):
        doc_set = documents[i * batch_size : (i + 1) * batch_size]

        # Extract tokens
        analyzer = topic_model.vectorizer_model.build_tokenizer()
        tokens = [analyzer(document) for document in doc_set]

        # Extract token sets
        all_sentences = []
        all_indices = [0]
        all_token_sets_ids = []

        for tokenset in tokens:
            if len(tokenset) < window:
                token_sets = [tokenset]
                token_sets_ids = [list(range(len(tokenset)))]
            else:
                # Extract tokensets using window and stride parameters
                stride_indices = list(range(len(tokenset)))[::stride]
                token_sets = []
                token_sets_ids = []
                for stride_index in stride_indices:
                    selected_tokens = tokenset[stride_index : stride_index + window]

                    if padding or len(selected_tokens) == window:
                        token_sets.append(selected_tokens)
                        token_sets_ids.append(
                            list(
                                range(
                                    stride_index,
                                    stride_index + len(selected_tokens),
                                )
                            )
                        )

                # Add empty tokens at the beginning and end of a document
                if padding:
                    padded = []
                    padded_ids = []
                    t = math.ceil(window / stride) - 1
                    for i in range(math.ceil(window / stride) - 1):
                        padded.append(tokenset[: window - ((t - i) * stride)])
                        padded_ids.append(list(range(0, window - ((t - i) * stride))))

                    token_sets = padded + token_sets
                    token_sets_ids = padded_ids + token_sets_ids

            # Join the tokens
            sentences = [separator.join(token) for token in token_sets]
            all_sentences.extend(sentences)
            all_token_sets_ids.extend(token_sets_ids)
            all_indices.append(all_indices[-1] + len(sentences))

        # Calculate similarity between embeddings of token sets and the topics
        if use_embedding_model:
            embeddings = topic_model._extract_embeddings(all_sentences, verbose=True)
            similarity = cosine_similarity(embeddings, topic_model.topic_embeddings_[topic_model._outliers :])

        # Calculate similarity between c-TF-IDF of token sets and the topics
        else:
            bow_doc = topic_model.vectorizer_model.transform(all_sentences)
            c_tf_idf_doc = topic_model.ctfidf_model.transform(bow_doc)
            similarity = cosine_similarity(c_tf_idf_doc, topic_model.c_tf_idf_[topic_model._outliers :])

        # Only keep similarities that exceed the minimum
        similarity[similarity < min_similarity] = 0

        # Aggregate results on an individual token level
        if calculate_tokens:
            topic_distribution = []
            topic_token_distribution = []
            for index, token in enumerate(tokens):
                start = all_indices[index]
                end = all_indices[index + 1]

                if start == end:
                    end = end + 1

                # Assign topics to individual tokens
                token_id = [i for i in range(len(token))]
                token_val = {index: [] for index in token_id}
                for sim, token_set in zip(similarity[start:end], all_token_sets_ids[start:end]):
                    for token in token_set:
                        if token in token_val:
                            token_val[token].append(sim)

                matrix = []
                for _, value in token_val.items():
                    matrix.append(np.add.reduce(value))

                # Take empty documents into account
                matrix = np.array(matrix)
                if len(matrix.shape) == 1:
                    matrix = np.zeros((1, len(topic_model.topic_labels_) - topic_model._outliers))

                topic_token_distribution.append(np.array(matrix))
                topic_distribution.append(np.add.reduce(matrix))

            topic_distribution = normalize(topic_distribution, norm="l1", axis=1)

        # Aggregate on a tokenset level indicated by the window and stride
        else:
            topic_distribution = []
            for index in range(len(all_indices) - 1):
                start = all_indices[index]
                end = all_indices[index + 1]

                if start == end:
                    end = end + 1
                group = similarity[start:end].sum(axis=0)
                topic_distribution.append(group)
            topic_distribution = normalize(np.array(topic_distribution), norm="l1", axis=1)
            topic_token_distribution = None

        # Combine results
        topic_distributions.append(topic_distribution)
        if topic_token_distribution is None:
            topic_token_distributions = None
        else:
            topic_token_distributions.extend(topic_token_distribution)

    topic_distributions = np.vstack(topic_distributions)

    return topic_distributions, topic_token_distributions
