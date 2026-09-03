import numpy as np
import narwhals.stable.v2 as nw

from bertopic._config import get_output

try:
    from great_tables import GT, loc, style  # noqa: F401

    HAS_GREAT_TABLES = True
except (ModuleNotFoundError, ImportError):
    HAS_GREAT_TABLES = False


def visualize_approximate_distribution(
    topic_model,
    document: str,
    topic_token_distribution: np.ndarray,
    normalize: bool = False,
):
    """Visualize the topic distribution calculated by `.approximate_topic_distribution`
    on a token level. Thereby indicating the extend to which a certain word or phrases belong
    to a specific topic. The assumption here is that a single word can belong to multiple
    similar topics and as such give information about the broader set of topics within
    a single document.

    Note:
    This function will return a styled GT table if Great Tables is installed. If not,
    it will return a plain polars DataFrame. To install great_tables:

    `pip install great_tables`

    Arguments:
        topic_model: A fitted BERTopic instance.
        document: The document for which you want to visualize
                  the approximated topic distribution.
        topic_token_distribution: The topic-token distribution of the document as
                                  extracted by `.approximate_topic_distribution`
        normalize: Whether to normalize, between 0 and 1 (summing to 1), the
                   topic distribution values.

    Returns:
        df: A styled GT table or polars DataFrame indicating the best fitting topics
            for each token.

    Examples:
    ```python
    # Calculate the topic distributions on a token level
    # Note that we need to have `calculate_token_level=True`
    topic_distr, topic_token_distr = topic_model.approximate_distribution(
            docs, calculate_token_level=True
    )

    # Visualize the approximated topic distributions
    df = topic_model.visualize_approximate_distribution(docs[0], topic_token_distr[0])
    df
    ```
    """
    # Tokenize document
    analyzer = topic_model.vectorizer_model.build_tokenizer()
    tokens = analyzer(document)

    if len(tokens) == 0:
        raise ValueError("Make sure that your document contains at least 1 token.")

    # Prepare dataframe with results
    if normalize:
        data = (topic_token_distribution / topic_token_distribution.sum()).T
    else:
        data = topic_token_distribution.T

    columns = [f"{token}{' ' * i}" for i, token in enumerate(tokens)]
    topic_labels = list(topic_model.topic_labels_.values())[topic_model._outliers :]

    # Drop topics with no weight on any token before building, rather than filtering
    # after: a pandas filter keeps the original row labels, leaving gaps in the index.
    keep = data.sum(axis=1) != 0
    data = data[keep]
    topic_labels = [label for label, keep_label in zip(topic_labels, keep) if keep_label]

    if len(topic_labels) == 0:
        return nw.from_dict({}, backend=get_output()).to_native()

    df = nw.from_dict(
        {"Topic": topic_labels, **{column: data[:, index] for index, column in enumerate(columns)}},
        backend=get_output(),
    )

    # Style the resulting dataframe using Great Tables
    if HAS_GREAT_TABLES:
        max_val = df.select(nw.max_horizontal(columns).alias("max"))["max"].max()
        return (
            GT(df.to_native())
            .tab_stub(rowname_col="Topic")
            .fmt_number(columns=columns, decimals=3)
            .data_color(columns=columns, palette="Blues", domain=[0, max_val])
        )

    return df.to_native()
