import numpy as np
import polars as pl

try:
    from great_tables import loc, style  # noqa: F401

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

    df = pl.from_numpy(data, schema=columns)
    df = df.insert_column(0, pl.Series("Topic", topic_labels))

    # Filter rows where all token values are 0
    value_cols = [c for c in df.columns if c != "Topic"]
    df = df.filter(pl.sum_horizontal(value_cols) != 0)

    if len(df) == 0:
        return df

    # Style the resulting dataframe using Great Tables
    if HAS_GREAT_TABLES:
        max_val = df.select(value_cols).max_horizontal().max()
        return (
            df.style.tab_stub(rowname_col="Topic")
            .fmt_number(columns=value_cols, decimals=3)
            .data_color(columns=value_cols, palette="Blues", domain=[0, max_val])
        )

    return df
