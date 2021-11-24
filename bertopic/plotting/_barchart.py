import numpy as np
from typing import List

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualize_barchart(topic_model,
                       topics: List[int] = None,
                       top_n_topics: int = 8,
                       n_words: int = 5,
                       columns: int = 4) -> go.Figure:
    """ Visualize a barchart of selected topics

    Arguments:
        topic_model: A fitted BERTopic instance.
        topics: A selection of topics to visualize.
        top_n_topics: Only select the top n most frequent topics.
        n_words: Number of words to show in a topic
        columns: Number of columns to plot.

    Returns:
        fig: A plotly figure

    Usage:

    To visualize the barchart of selected topics
    simply run:

    ```python
    topic_model.visualize_barchart()
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_barchart()
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../tutorial/visualization/bar_chart.html"
    style="width:1100px; height: 660px; border: 0px;""></iframe>
    """
    # Select topics based on top_n and topics args
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = topic_model.get_topic_freq().Topic.to_list()[1:top_n_topics + 1]
    else:
        topics = topic_model.get_topic_freq().Topic.to_list()[1:7]

    # Initialize figure
    subplot_titles = [f"Topic {topic}" for topic in topics]
    columns = 4
    rows = int(np.ceil(len(topics) / columns))
    fig = make_subplots(rows=rows,
                        cols=columns,
                        shared_xaxes=True,
                        horizontal_spacing=.15,
                        vertical_spacing=.2/rows,
                        subplot_titles=subplot_titles)

    # Add barchart for each topic
    row = 1
    column = 1
    for topic in topics:
        words = [word + "  " for word, _ in topic_model.get_topic(topic)][:n_words][::-1]
        scores = [score for _, score in topic_model.get_topic(topic)][:n_words][::-1]

        fig.add_trace(
            go.Bar(x=scores,
                   y=words,
                   orientation='h'),
            row=row, col=column)

        if column == columns:
            column = 1
            row += 1
        else:
            column += 1

    # Stylize graph
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        title={
            'text': "<b>Topic Word Scores",
            'font': dict(
                size=22,
                color="Black")
        },
        width=220*columns,
        autosize=False,
        height=200+25*top_n_topics,
        uniformtext={"mode": "show", "minsize": 8},
        margin_t=100,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )
    
    fig.update_xaxes(showgrid=True,automargin=True,showticklabels=False)
    fig.update_yaxes(showgrid=True,automargin=True,nticks=n_words)

    return fig
  
