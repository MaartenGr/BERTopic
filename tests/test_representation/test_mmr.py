import logging
from types import SimpleNamespace

from bertopic.representation import MaximalMarginalRelevance


def _candidate_topics(nr_words=30):
    return {
        0: [(f"word_{i}", 1 / (i + 1)) for i in range(nr_words)],
        1: [(f"term_{i}", 1 / (i + 1)) for i in range(nr_words)],
    }


def test_mmr_without_embedding_model_respects_top_n_words(caplog):
    topic_model = SimpleNamespace(embedding_model=None)
    mmr = MaximalMarginalRelevance(top_n_words=10)
    topics = _candidate_topics()

    with caplog.at_level(logging.WARNING, logger="BERTopic"):
        logging.getLogger("BERTopic").addHandler(caplog.handler)
        try:
            updated_topics = mmr.extract_topics(topic_model, None, None, topics)
        finally:
            logging.getLogger("BERTopic").removeHandler(caplog.handler)

    assert all(len(words) == 10 for words in updated_topics.values())
    assert updated_topics[0] == topics[0][:10]
    assert "MaximalMarginalRelevance" in caplog.text
