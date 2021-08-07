import numpy as np
import plotly.graph_objects as go


def visualize_distribution(topic_model,
                           probabilities: np.ndarray,
                           min_probability: float = 0.015,
                           width: int = 800,
                           height: int = 600) -> go.Figure:
    """ Visualize the distribution of topic probabilities

    Arguments:
        topic_model: A fitted BERTopic instance.
        probabilities: An array of probability scores
        min_probability: The minimum probability score to visualize.
                         All others are ignored.
        width: The width of the figure.
        height: The height of the figure.

    Usage:

    Make sure to fit the model before and only input the
    probabilities of a single document:

    ```python
    topic_model.visualize_distribution(probabilities[0])
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_distribution(probabilities[0])
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../tutorial/visualization/probabilities.html"
    style="width:1000px; height: 500px; border: 0px;""></iframe>
    """
    if len(probabilities.shape) != 2:
        raise ValueError("This visualization cannot be used if you have set `calculate_probabilities` to False "
                         "as it uses the topic probabilities of all topics. ")
    if len(probabilities[probabilities > min_probability]) == 0:
        raise ValueError("There are no values where `min_probability` is higher than the "
                         "probabilities that were supplied. Lower `min_probability` to prevent this error.")
    if not topic_model.calculate_probabilities:
        raise ValueError("This visualization cannot be used if you have set `calculate_probabilities` to False "
                         "as it uses the topic probabilities. ")

    # Get values and indices equal or exceed the minimum probability
    labels_idx = np.argwhere(probabilities >= min_probability).flatten()
    vals = probabilities[labels_idx].tolist()

    # Create labels
    labels = []
    for idx in labels_idx:
        words = topic_model.get_topic(idx)
        if words:
            label = [word[0] for word in words[:5]]
            label = f"<b>Topic {idx}</b>: {'_'.join(label)}"
            label = label[:40] + "..." if len(label) > 40 else label
            labels.append(label)
        else:
            vals.remove(probabilities[idx])

    # Create Figure
    fig = go.Figure(go.Bar(
        x=vals,
        y=labels,
        marker=dict(
            color='#C8D2D7',
            line=dict(
                color='#6E8484',
                width=1),
        ),
        orientation='h')
    )

    fig.update_layout(
        xaxis_title="Probability",
        title={
            'text': "<b>Topic Probability Distribution",
            'y': .95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        template="simple_white",
        width=width,
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )

    return fig
