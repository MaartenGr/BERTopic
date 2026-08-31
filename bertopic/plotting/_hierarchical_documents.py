import numpy as np
import polars as pl
import plotly.graph_objects as go
import math

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bertopic import BERTopic


def visualize_hierarchical_documents(
    topic_model: "BERTopic",
    docs: list[str],
    hierarchical_topics: pl.DataFrame | None = None,
    topics: list[int] | None = None,
    embeddings: np.ndarray = None,
    reduced_embeddings: np.ndarray = None,
    sample: float | int | None = None,
    hide_annotations: bool = False,
    hide_document_hover: bool = True,
    nr_levels: int = 10,
    level_scale: str = "linear",
    custom_labels: bool | str = False,
    title: str = "<b>Hierarchical Documents and Topics</b>",
    width: int = 1200,
    height: int = 750,
) -> go.Figure:
    """Visualize documents and their topics in 2D at different levels of hierarchy.

    Arguments:
        topic_model: A fitted BERTopic instance.
        docs: The documents you used when calling either `fit` or `fit_transform`
        hierarchical_topics: A dataframe that contains a hierarchy of topics
                             represented by their parents and their children
        topics: A selection of topics to visualize.
                Not to be confused with the topics that you get from `.fit_transform`.
                For example, if you want to visualize only topics 1 through 5:
                `topics = [1, 2, 3, 4, 5]`.
        embeddings: The embeddings of all documents in `docs`.
        reduced_embeddings: The 2D reduced embeddings of all documents in `docs`.
        sample: The percentage of documents in each topic that you would like to keep.
                Value can be between 0 and 1. Setting this value to, for example,
                0.1 (10% of documents in each topic) makes it easier to visualize
                millions of documents as a subset is chosen.
        hide_annotations: Hide the names of the traces on top of each cluster.
        hide_document_hover: Hide the content of the documents when hovering over
                             specific points. Helps to speed up generation of visualizations.
        nr_levels: The number of levels to be visualized in the hierarchy. First, the distances
                   in `hierarchical_topics.Distance` are split in `nr_levels` lists of distances.
                   Then, for each list of distances, the merged topics are selected that have a
                   distance less or equal to the maximum distance of the selected list of distances.
                   NOTE: To get all possible merged steps, make sure that `nr_levels` is equal to
                   the length of `hierarchical_topics`.
        level_scale: Whether to apply a linear or logarithmic (log) scale levels of the distance
                     vector. Linear scaling will perform an equal number of merges at each level
                     while logarithmic scaling will perform more mergers in earlier levels to
                     provide more resolution at higher levels (this can be used for when the number
                     of topics is large).
        custom_labels: If bool, whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
                       NOTE: Custom labels are only generated for the original
                       un-merged topics.
        title: Title of the plot.
        width: The width of the figure.
        height: The height of the figure.

    Examples:
    To visualize the topics simply run:

    ```python
    topic_model.visualize_hierarchical_documents(docs, hierarchical_topics)
    ```

    Do note that this re-calculates the embeddings and reduces them to 2D.
    The advised and preferred pipeline for using this function is as follows:

    ```python
    from sklearn.datasets import fetch_20newsgroups
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from umap import UMAP

    # Prepare embeddings
    docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(docs, show_progress_bar=False)

    # Train BERTopic and extract hierarchical topics
    topic_model = BERTopic().fit(docs, embeddings)
    hierarchical_topics = topic_model.hierarchical_topics(docs)

    # Reduce dimensionality of embeddings, this step is optional
    # reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)

    # Run the visualization with the original embeddings
    topic_model.visualize_hierarchical_documents(docs, hierarchical_topics, embeddings=embeddings)

    # Or, if you have reduced the original embeddings already:
    topic_model.visualize_hierarchical_documents(docs, hierarchical_topics, reduced_embeddings=reduced_embeddings)
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_hierarchical_documents(docs, hierarchical_topics, reduced_embeddings=reduced_embeddings)
    fig.write_html("path/to/file.html")
    ```

    Note:
        This visualization was inspired by the scatter plot representation of Doc2Map:
        https://github.com/louisgeisler/Doc2Map

    <iframe src="../../getting_started/visualization/hierarchical_documents.html"
    style="width:1000px; height: 770px; border: 0px;""></iframe>
    """
    if not isinstance(hierarchical_topics, pl.DataFrame):
        hierarchical_topics = topic_model._hierarchy.to_polars()
    topic_per_doc = topic_model.topics_

    # Sample the data to optimize for visualization and dimensionality reduction
    if sample is None or sample > 1:
        sample = 1

    indices = []
    for topic in set(topic_per_doc):
        s = np.where(np.array(topic_per_doc) == topic)[0]
        size = len(s) if len(s) < 100 else int(len(s) * sample)
        indices.extend(np.random.choice(s, size=size, replace=False))
    indices = np.array(indices)

    df = pl.DataFrame(
        {
            "topic": [topic_per_doc[index] for index in indices],
            "doc": [docs[index] for index in indices],
        }
    )

    # Extract embeddings if not already done
    if sample is None:
        if embeddings is None and reduced_embeddings is None:
            embeddings_to_reduce = topic_model._extract_embeddings(df["doc"].to_list())
        else:
            embeddings_to_reduce = embeddings
    else:
        if embeddings is not None:
            embeddings_to_reduce = embeddings[indices]
        elif embeddings is None and reduced_embeddings is None:
            embeddings_to_reduce = topic_model._extract_embeddings(df["doc"].to_list())

    # Reduce input embeddings
    if reduced_embeddings is None:
        try:
            from umap import UMAP

            umap_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric="cosine").fit(
                embeddings_to_reduce
            )
            embeddings_2d = umap_model.embedding_
        except (ImportError, ModuleNotFoundError):
            raise ModuleNotFoundError(
                "UMAP is required if the embeddings are not yet reduced in dimensionality. Please install it using `pip install umap-learn`."
            )
    elif sample is not None and reduced_embeddings is not None:
        embeddings_2d = reduced_embeddings[indices]
    elif sample is None and reduced_embeddings is not None:
        embeddings_2d = reduced_embeddings

    # Combine data
    df = df.with_columns(
        pl.Series("x", embeddings_2d[:, 0]),
        pl.Series("y", embeddings_2d[:, 1]),
    )

    # Create topic list for each level, levels are created by calculating the distance
    distances = hierarchical_topics["Distance"].to_list()
    if level_scale == "log" or level_scale == "logarithmic":
        log_indices = (
            np.round(
                np.logspace(
                    start=math.log(1, 10),
                    stop=math.log(len(distances) - 1, 10),
                    num=nr_levels,
                )
            )
            .astype(int)
            .tolist()
        )
        log_indices.reverse()
        max_distances = [distances[i] for i in log_indices]
    elif level_scale == "lin" or level_scale == "linear":
        max_distances = [
            distances[indices[-1]] for indices in np.array_split(range(len(hierarchical_topics)), nr_levels)
        ][::-1]
    else:
        raise ValueError("level_scale needs to be one of 'log' or 'linear'")

    for index, max_distance in enumerate(max_distances):
        # Get topics below `max_distance`
        mapping = {topic: topic for topic in df["topic"].unique().to_list()}
        selection = hierarchical_topics.filter(pl.col("Distance") <= max_distance)
        selection = selection.with_columns(pl.col("Parent_ID").cast(pl.Int64))
        selection = selection.sort("Parent_ID")

        for row in selection.iter_rows(named=True):
            for topic in row["Topics"]:
                mapping[topic] = row["Parent_ID"]

        # Make sure the mappings are mapped 1:1
        mappings = [True for _ in mapping]
        while any(mappings):
            for i, (key, value) in enumerate(mapping.items()):
                if value in mapping.keys() and key != value:
                    mapping[key] = mapping[value]
                else:
                    mappings[i] = False

        # Create new column
        df = df.with_columns(pl.col("topic").replace(mapping).cast(pl.Int64).alias(f"level_{index + 1}"))

    # Prepare topic names of original and merged topics
    trace_names = []
    topic_names = {}
    for topic in range(hierarchical_topics["Parent_ID"].cast(pl.Int64).max()):
        if topic < hierarchical_topics["Parent_ID"].cast(pl.Int64).min():
            if topic_model.get_topic(topic):
                if isinstance(custom_labels, str):
                    trace_name = f"{topic}_" + "_".join(
                        next(zip(*topic_model.topic_aspects_[custom_labels][topic]))[:3]
                    )
                elif topic_model.custom_labels_ is not None and custom_labels:
                    trace_name = topic_model.custom_labels_[topic + topic_model._outliers]
                else:
                    trace_name = f"{topic}_" + "_".join(
                        [word[:20] for word, _ in topic_model.get_topic(topic)][:3]
                    )
                topic_names[topic] = {
                    "trace_name": trace_name[:40],
                    "plot_text": trace_name[:40],
                }
                trace_names.append(trace_name)
        else:
            trace_name = (
                f"{topic}_" + hierarchical_topics.filter(pl.col("Parent_ID") == str(topic))["Parent_Name"][0]
            )
            plot_text = "_".join([name[:20] for name in trace_name.split("_")[:3]])
            topic_names[topic] = {
                "trace_name": trace_name[:40],
                "plot_text": plot_text[:40],
            }
            trace_names.append(trace_name)

    # Prepare traces
    all_traces = []
    for level in range(len(max_distances)):
        traces = []

        # Outliers
        if topic_model._outliers:
            outlier_df = df.filter(pl.col(f"level_{level + 1}") == -1)
            traces.append(
                go.Scattergl(
                    x=outlier_df["x"],
                    y=outlier_df["y"],
                    mode="markers+text",
                    name="other",
                    hoverinfo="text",
                    hovertext=outlier_df["doc"] if not hide_document_hover else None,
                    showlegend=False,
                    marker=dict(color="#CFD8DC", size=5, opacity=0.5),
                )
            )

        # Selected topics
        if topics:
            selection = df.filter(pl.col("topic").is_in(topics))
            unique_topics = sorted(
                [int(topic) for topic in selection[f"level_{level + 1}"].unique().to_list()]
            )
        else:
            unique_topics = sorted([int(topic) for topic in df[f"level_{level + 1}"].unique().to_list()])

        for topic in unique_topics:
            if topic != -1:
                if topics:
                    selection = df.filter(
                        (pl.col(f"level_{level + 1}") == topic) & (pl.col("topic").is_in(topics))
                    )
                else:
                    selection = df.filter(pl.col(f"level_{level + 1}") == topic)

                if not hide_annotations:
                    selection = selection.with_columns(pl.lit("").alias("text"))
                    annotation_data = {col: [None] for col in selection.columns}
                    annotation_data["x"] = [selection["x"].mean()]
                    annotation_data["y"] = [selection["y"].mean()]
                    annotation_data["text"] = [topic_names[int(topic)]["plot_text"]]
                    selection = pl.concat(
                        [selection, pl.DataFrame(annotation_data, schema=selection.schema)],
                        how="vertical",
                    )

                traces.append(
                    go.Scattergl(
                        x=selection["x"],
                        y=selection["y"],
                        text=selection["text"] if not hide_annotations else None,
                        hovertext=selection["doc"] if not hide_document_hover else None,
                        hoverinfo="text",
                        name=topic_names[int(topic)]["trace_name"],
                        mode="markers+text",
                        marker=dict(size=5, opacity=0.5),
                    )
                )

        all_traces.append(traces)

    # Track and count traces
    nr_traces_per_set = [len(traces) for traces in all_traces]
    trace_indices = [(0, nr_traces_per_set[0])]
    for index, nr_traces in enumerate(nr_traces_per_set[1:]):
        start = trace_indices[index][1]
        end = nr_traces + start
        trace_indices.append((start, end))

    # Visualization
    fig = go.Figure()
    for traces in all_traces:
        for trace in traces:
            fig.add_trace(trace)

    for index in range(len(fig.data)):
        if index >= nr_traces_per_set[0]:
            fig.data[index].visible = False

    # Create and add slider
    steps = []
    for index, indices in enumerate(trace_indices):
        step = dict(
            method="update",
            label=str(index),
            args=[{"visible": [False] * len(fig.data)}],
        )
        for index in range(indices[1] - indices[0]):
            step["args"][0]["visible"][index + indices[0]] = True
        steps.append(step)

    sliders = [dict(currentvalue={"prefix": "Level: "}, pad={"t": 20}, steps=steps)]

    # Add grid in a 'plus' shape
    x_range = (
        df["x"].min() - abs((df["x"].min()) * 0.15),
        df["x"].max() + abs((df["x"].max()) * 0.15),
    )
    y_range = (
        df["y"].min() - abs((df["y"].min()) * 0.15),
        df["y"].max() + abs((df["y"].max()) * 0.15),
    )
    fig.add_shape(
        type="line",
        x0=sum(x_range) / 2,
        y0=y_range[0],
        x1=sum(x_range) / 2,
        y1=y_range[1],
        line=dict(color="#CFD8DC", width=2),
    )
    fig.add_shape(
        type="line",
        x0=x_range[0],
        y0=sum(y_range) / 2,
        x1=x_range[1],
        y1=sum(y_range) / 2,
        line=dict(color="#9E9E9E", width=2),
    )
    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)

    # Stylize layout
    fig.update_layout(
        sliders=sliders,
        template="simple_white",
        title={
            "text": f"{title}",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=22, color="Black"),
        },
        width=width,
        height=height,
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig
