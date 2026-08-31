from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bertopic import BERTopic


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
