import numpy as np
import plotly.graph_objects as go

from typing import List, Union


def visualize_representative_images(
    topic_model,
    topics: List[int] | None = None,
    aspect: str = "Visual_Aspect",
    custom_labels: Union[bool, str] = False,
    title: str = "<b>Representative Images per Topic</b>",
    width: int = 1200,
    height: int = 800,
):
    """Visualize the representative images of each topic with a slider for topic selection.

    When BERTopic is fitted on images with a `VisualRepresentation` aspect, each topic
    is represented by a collage of its most representative images. This visualization
    shows those collages at their original resolution, one topic at a time, with a
    slider to step through the topics.

    Arguments:
        topic_model: A fitted BERTopic instance.
        topics: A selection of topics to visualize.
                For example, if you want to visualize only topics 1 through 5:
                `topics = [1, 2, 3, 4, 5]`.
        aspect: The name of the topic aspect that contains the representative images
                as created by `bertopic.representation.VisualRepresentation`.
        custom_labels: If bool, whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
        title: Title of the plot.
        width: The width of the figure.
        height: The height of the figure.

    Examples:
    To visualize the representative images of each topic, make sure to fit BERTopic
    with a `VisualRepresentation` aspect first:

    ```python
    from bertopic import BERTopic
    from bertopic.representation import VisualRepresentation

    # Additional representation of topics as a collage of images
    representation_model = {"Visual_Aspect": VisualRepresentation()}
    topic_model = BERTopic(embedding_model="clip-ViT-B-32", representation_model=representation_model)
    topics, probs = topic_model.fit_transform(documents=None, images=images)

    # Run the visualization
    topic_model.visualize_representative_images()
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_representative_images()
    fig.write_html("path/to/file.html")
    ```
    """
    if aspect not in topic_model.topic_aspects_:
        raise ValueError(
            f"The aspect '{aspect}' could not be found in `topic_model.topic_aspects_`. "
            "Make sure to fit BERTopic with a `VisualRepresentation` aspect, for example: "
            '`BERTopic(representation_model={"Visual_Aspect": VisualRepresentation()})`.'
        )
    image_topics = topic_model.topic_aspects_[aspect]

    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is None:
        topics = sorted(freq_df.Topic.to_list())
    topics = [topic for topic in topics if topic in image_topics]
    if not topics:
        raise ValueError(f"None of the selected topics have images in the '{aspect}' aspect.")

    # Prepare topic names
    if isinstance(custom_labels, str):
        names = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in topics]
        names = ["_".join([label[0] for label in labels[:4]]) for labels in names]
        names = [label if len(label) < 30 else label[:27] + "..." for label in names]
    elif topic_model.custom_labels_ is not None and custom_labels:
        names = [topic_model.custom_labels_[topic + topic_model._outliers] for topic in topics]
    else:
        names = [f"{topic}_" + "_".join([word for word, value in topic_model.get_topic(topic)][:3]) for topic in topics]

    # Visualize
    fig = go.Figure()
    for index, topic in enumerate(topics):
        fig.add_trace(go.Image(z=np.asarray(image_topics[topic]), hoverinfo="skip", visible=index == 0))

    # Create a slider for topic selection
    steps = [
        dict(
            label=f"Topic {topic}",
            method="update",
            args=[
                {"visible": [index == i for i in range(len(topics))]},
                {"title.text": f"{title}<br><sup>{name}</sup>"},
            ],
        )
        for index, (topic, name) in enumerate(zip(topics, names))
    ]
    sliders = [dict(active=0, pad={"t": 50}, steps=steps)]

    # Stylize layout
    fig.update_layout(
        template="simple_white",
        title={
            "text": f"{title}<br><sup>{names[0]}</sup>",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=22, color="Black"),
        },
        width=width,
        height=height,
        sliders=sliders,
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig
