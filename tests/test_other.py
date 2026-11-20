import builtins
import importlib
import sys

import pytest

from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic._utils import MockPlotlyModule
import bertopic._bertopic as bertopic_module

try:
    from plotly.graph_objects import Figure
except ImportError:
    Figure = None


def test_load_save_model():
    model = BERTopic(language="Dutch", embedding_model=None)
    model.save("test", serialization="pickle")
    loaded_model = BERTopic.load("test")
    assert type(model) is type(loaded_model)
    assert model.language == loaded_model.language
    assert model.embedding_model == loaded_model.embedding_model
    assert model.top_n_words == loaded_model.top_n_words


def test_get_params():
    model = BERTopic()
    params = model.get_params()
    assert not params["embedding_model"]
    assert not params["low_memory"]
    assert not params["nr_topics"]
    assert params["n_gram_range"] == (1, 1)
    assert params["min_topic_size"] == 10
    assert params["language"] == "english"


def test_no_plotly():
    model = BERTopic(
        language="Dutch",
        embedding_model=None,
        min_topic_size=2,
        top_n_words=1,
        umap_model=BaseDimensionalityReduction(),
    )
    model.fit(["hello", "hi", "goodbye", "goodbye", "whats up"] * 10)

    try:
        out = model.visualize_topics()
        assert isinstance(out, Figure) if Figure else False
    except ImportError as e:
        assert "Plotly is required to use" in str(e)
        assert "bertopic[plotting]" in str(e)


def test_missing_hdbscan_dependency(monkeypatch):
    monkeypatch.setattr(bertopic_module, "HAS_HDBSCAN", False)
    monkeypatch.setattr(bertopic_module, "SK_HDBSCAN", None)

    with pytest.raises(ModuleNotFoundError, match=r"pip install bertopic\[hdbscan\]"):
        BERTopic(embedding_model=None, umap_model=BaseDimensionalityReduction())


def test_missing_sentence_transformers_dependency(monkeypatch):
    module_name = "bertopic.backend._sentencetransformers"
    original_module = sys.modules.pop(module_name, None)
    original_import = builtins.__import__

    def import_without_sentence_transformers(name, *args, **kwargs):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ModuleNotFoundError("No module named 'sentence_transformers'", name="sentence_transformers")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_sentence_transformers)
    try:
        with pytest.raises(ModuleNotFoundError, match=r"pip install bertopic\[sentence-transformers\]"):
            importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module


def test_mock_plotly_dependency_message():
    with pytest.raises(ImportError, match=r"pip install bertopic\[plotting\]"):
        MockPlotlyModule().visualize_topics()
