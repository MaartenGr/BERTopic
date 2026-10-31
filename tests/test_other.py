import pytest
import copy

from bertopic import BERTopic

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


@pytest.mark.parametrize(
    "pickled",
    [
        # A pickle from 0.17.4 holds its TopicMapper, a class that 1.0 no longer has,
        b"cbertopic._bertopic\nTopicMapper\n.",
        # and can hold a VisualRepresentation, from a module that 1.0 no longer has
        b"cbertopic.representation._visual\nVisualRepresentation\n.",
    ],
)
def test_load_a_model_pickled_before_1_0(tmp_path, pickled):
    path = tmp_path / "model"
    path.write_bytes(pickled)
    with pytest.raises(ValueError, match="safetensors"):
        BERTopic.load(path)


def test_get_params():
    model = BERTopic()
    params = model.get_params()
    assert not params["embedding_model"]
    assert not params["low_memory"]
    assert not params["nr_topics"]
    assert params["n_gram_range"] == (1, 1)
    assert params["min_topic_size"] == 10
    assert params["language"] == "english"


@pytest.mark.parametrize("model", [("base_topic_model")])
def test_no_plotly(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    try:
        out = topic_model.visualize_topics()
        assert isinstance(out, Figure) if Figure else False
    except ImportError as e:
        assert "Plotly is required to use" in str(e)
