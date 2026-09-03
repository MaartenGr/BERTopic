from typing import TYPE_CHECKING

import narwhals.stable.v2 as nw
from narwhals.stable.v2.typing import IntoDataFrameT

from bertopic._config import get_output

if TYPE_CHECKING:
    from bertopic import BERTopic


def with_annotation(selection: IntoDataFrameT, label: str) -> IntoDataFrameT:
    """Append one extra point at the centre of a cluster, carrying its label.

    The row is copied from an existing one so that every column keeps its type, then x, y
    and text are overridden. A row of nulls would not do: pandas has no null for an integer
    column, and the explicit schema keeps x and y at the width they were built with.

    A cluster with no documents has no centre to label, so it is handed back untouched.
    """
    if len(selection) == 0:
        return selection

    annotation = {column: [selection[column][0]] for column in selection.columns}
    annotation["x"] = [selection["x"].mean()]
    annotation["y"] = [selection["y"].mean()]
    annotation["text"] = [label]

    extra_row = nw.from_dict(annotation, backend=get_output(), schema=selection.schema)
    return nw.concat([selection, extra_row], how="vertical")


def select_topics(
    topic_model: "BERTopic", topics: list[int] | None = None, top_n_topics: int | None = None
) -> list[int]:
    """Select topics based on a given list of topics and which .

    Arguments:
        topic_model: A fitted BERTopic instance.
        topics: A list of topics to select. If None, all topics are selected.
        top_n_topics: The number of top topics to select based on frequency. If None, all topics are selected.

    Returns:
        topics: A list of selected topics.
    """
    all_topic_ids = topic_model._topics.topic_ids(outliers=False)
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        frequencies = topic_model._topics.frequencies()
        top_n = sorted(all_topic_ids, key=lambda t: frequencies.get(t, 0), reverse=True)[:top_n_topics]
        topics = sorted(top_n)
    else:
        topics = sorted(all_topic_ids)
    return topics
