from bertopic import BERTopic
import pytest


@pytest.fixture(scope="module")
def base_bertopic():
    model = BERTopic(language="english",
                     verbose=True,
                     min_topic_size=5)
    return model
