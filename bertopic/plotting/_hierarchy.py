import numpy as np
from scipy.cluster.hierarchy import linkage
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

import plotly.graph_objects as go
import plotly.figure_factory as ff


def visualize_hierarchy(topic_model,
                        orientation: str = "left",
                        topics: List[int] = None,
                        top_n_topics: int = None,
                        width: int = 1000,
                        height: int = 600,
                        optimal_ordering: bool = False) -> go.Figure:
    """ Visualize a hierarchical structure of the topics

    A ward linkage function is used to perform the
    hierarchical clustering based on the cosine distance
    matrix between topic embeddings.

    Arguments:
        topic_model: A fitted BERTopic instance.
        orientation: The orientation of the figure.
                     Either 'left' or 'bottom'
        topics: A selection of topics to visualize
        top_n_topics: Only select the top n most frequent topics
        width: The width of the figure. Only works if orientation is set to 'left'
        height: The height of the figure. Only works if orientation is set to 'bottom'
        optimal_ordering: If True, the linkage matrix will be reordered so that the distance
            between successive leaves is minimal. This results in a more intuitive
            tree structure when the data are visualized. defaults to False, because
            this algorithm can be slow, particularly on large datasets. See
            also the `linkage` function fun `scipy`.
    Returns:
        fig: A plotly figure

    Usage:

    To visualize the hierarchical structure of
    topics simply run:

    ```python
    topic_model.visualize_hierarchy()
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_hierarchy()
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/hierarchy.html"
    style="width:1000px; height: 680px; border: 0px;""></iframe>
    """

    # Select topic embeddings
    if topic_model.topic_embeddings is not None:
        embeddings = np.array(topic_model.topic_embeddings)
    else:
        embeddings = topic_model.c_tf_idf

    # Select topics based on top_n and topics args
    if topics is not None:
        topics = sorted(list(topics))
    elif top_n_topics is not None:
        topics = sorted(topic_model.get_topic_freq().Topic.to_list()[1:top_n_topics + 1])
    else:
        topics = sorted(list(topic_model.get_topics().keys()))

    # Select embeddings
    all_topics = sorted(list(topic_model.get_topics().keys()))
    indices = np.array([all_topics.index(topic) for topic in topics])
    embeddings = embeddings[indices]

    # Create dendogram
    distance_matrix = 1 - cosine_similarity(embeddings)
    fig = ff.create_dendrogram(distance_matrix,
                               orientation=orientation,
                               linkagefun=lambda x: linkage(x, "ward",
                                                            optimal_ordering=optimal_ordering),
                               color_threshold=1)

    # Create nicer labels
    axis = "yaxis" if orientation == "left" else "xaxis"
    new_labels = [[[str(topics[int(x)]), None]] + topic_model.get_topic(topics[int(x)])
                  for x in fig.layout[axis]["ticktext"]]
    new_labels = ["_".join([label[0] for label in labels[:4]]) for labels in new_labels]
    new_labels = [label if len(label) < 30 else label[:27] + "..." for label in new_labels]

    # Stylize layout
    fig.update_layout(
        plot_bgcolor='#ECEFF1',
        template="plotly_white",
        title={
            'text': "<b>Hierarchical Clustering",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )

    # Stylize orientation
    if orientation == "left":
        fig.update_layout(height=200 + (15 * len(topics)),
                          width=width,
                          yaxis=dict(tickmode="array",
                                     ticktext=new_labels))

        # Fix empty space on the bottom of the graph
        y_max = max([trace['y'].max() + 5 for trace in fig['data']])
        y_min = min([trace['y'].min() - 5 for trace in fig['data']])
        fig.update_layout(yaxis=dict(range=[y_min, y_max]))

    else:
        fig.update_layout(width=200 + (15 * len(topics)),
                          height=height,
                          xaxis=dict(tickmode="array",
                                     ticktext=new_labels))
    return fig
