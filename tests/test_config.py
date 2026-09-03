"""Contract tests for the dataframe library BERTopic returns.

BERTopic builds a dataframe only at the very end, so which library it builds with is a
presentation choice. Pandas is the default because that is what every existing notebook
expects, and polars is one call away.
"""

import pandas as pd
import pytest

import bertopic

# Polars is a test dependency rather than a runtime one, so skip rather than fail
# collection when the suite is run without the `test` extra
pl = pytest.importorskip("polars")


@pytest.fixture(autouse=True)
def restore_default_output():
    """The setting is global, so no test may leak its choice into the next one."""
    yield
    bertopic.set_output("pandas")


def test_the_default_is_pandas():
    assert bertopic.get_output() == "pandas"


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_supported_backends_are_accepted(backend):
    bertopic.set_output(backend)

    assert bertopic.get_output() == backend


def test_an_unknown_backend_is_rejected():
    """A typo should fail here rather than at the next `get_topic_info`."""
    with pytest.raises(ValueError, match="pandas"):
        bertopic.set_output("pandsa")


def test_returns_pandas_by_default(base_topic_model, documents):
    assert isinstance(base_topic_model.get_topic_info(), pd.DataFrame)
    assert isinstance(base_topic_model.get_topic_freq(), pd.DataFrame)
    assert isinstance(base_topic_model.get_document_info(documents), pd.DataFrame)


def test_returns_polars_once_asked(base_topic_model, documents):
    bertopic.set_output("polars")

    assert isinstance(base_topic_model.get_topic_info(), pl.DataFrame)
    assert isinstance(base_topic_model.get_topic_freq(), pl.DataFrame)
    assert isinstance(base_topic_model.get_document_info(documents), pl.DataFrame)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_variations_follow_the_setting(backend, base_topic_model, documents):
    """Every method that builds a dataframe honours the same choice."""
    bertopic.set_output(backend)
    expected = pd.DataFrame if backend == "pandas" else pl.DataFrame
    timestamps = [index % 10 for index in range(len(documents))]

    assert isinstance(base_topic_model.topics_over_time(documents, timestamps), expected)
    assert isinstance(base_topic_model.hierarchical_topics(documents), expected)
