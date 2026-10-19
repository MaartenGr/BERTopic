import copy
import sys
import types

import pytest
from transformers.pipelines.base import Pipeline

from bertopic.representation import LangChain, TextGeneration


@pytest.mark.parametrize(
    "name, build",
    [
        ("_openai", lambda module, **kwargs: module.OpenAI(None, **kwargs)),
        ("_ollama", lambda module, **kwargs: module.Ollama("gemma4:e4b-it-qat", **kwargs)),
        ("_cohere", lambda module, **kwargs: module.Cohere(None, **kwargs)),
        ("_litellm", lambda module, **kwargs: module.LiteLLM("gpt-4o-mini", **kwargs)),
        ("_llamacpp", lambda module, **kwargs: module.LlamaCPP(module.Llama(), **kwargs)),
    ],
)
def test_a_json_schema_stays_with_its_own_instance(name, build, stand_in_clients):
    module = stand_in_clients(f"bertopic.representation.{name}")
    plain = build(module)
    build(module, json_schema=True)

    # The plain instance sends no schema, although one was built with it afterwards
    assert "topic_label" not in str(plain.generator_kwargs)


def test_ollama_sends_a_format_only_when_given_one(stand_in_clients):
    module = stand_in_clients("bertopic.representation._ollama")

    # ollama-python rejects False as a format, so none is sent without a schema, and a user's own is kept
    assert "format" not in module.Ollama("gemma4:e4b-it-qat").generator_kwargs
    own_format = module.Ollama("gemma4:e4b-it-qat", generator_kwargs={"format": "json"}).generator_kwargs
    assert own_format["format"] == "json"


def test_openai_strict_schema_allows_no_other_properties(stand_in_clients):
    module = stand_in_clients("bertopic.representation._openai")
    text_format = module.OpenAI(None, json_schema=True).generator_kwargs["text"]["format"]
    assert text_format["strict"]
    assert text_format["schema"]["additionalProperties"] is False


@Pipeline.register
class Continuation:
    """A text-generation pipeline that continues every prompt with the same label."""

    def __call__(self, prompt, **kwargs):
        return [{"generated_text": f"{prompt} space exploration"}]


def test_text_generation_prompts_with_keywords_only(kmeans_pca_topic_model, documents):
    topic_model = copy.deepcopy(kmeans_pca_topic_model)
    representation_model = TextGeneration(Continuation())
    topic_model.update_topics(documents, representation_model=representation_model)

    # A completion model such as gpt2 reads 1,024 positions, so the default prompt leaves the documents out
    assert all(len(prompt.split()) < 100 for prompt in representation_model.prompts_)
    assert {topic_model.get_topic(topic)[0][0] for topic in topic_model.topic_sizes_} == {"space exploration"}


class ReasoningChatModel:
    """A chat model that answers with a reasoning block before its text, as LangChain messages hold them."""

    def invoke(self, messages, **kwargs):
        content = [
            {"type": "reasoning", "reasoning": "These are about space."},
            {"type": "text", "text": "topic: Space"},
        ]
        return types.SimpleNamespace(content=content, text="topic: Space")


def test_langchain_reads_the_text_of_content_blocks(kmeans_pca_topic_model, documents):
    topic_model = copy.deepcopy(kmeans_pca_topic_model)
    topic_model.update_topics(documents, representation_model=LangChain(ReasoningChatModel()))
    assert {topic_model.get_topic(topic)[0][0] for topic in topic_model.topic_sizes_} == {"Space"}


def test_litellm_retries_a_rate_limit(stand_in_clients, monkeypatch):
    module = stand_in_clients("bertopic.representation._litellm")
    attempts = []

    def completion(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise sys.modules["litellm"].RateLimitError("Too many requests")
        return {"choices": [{"message": {"content": "topic: Space"}}]}

    monkeypatch.setattr(module, "completion", completion)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    response = module.chat_completions_with_backoff(model="gpt-4o-mini", messages=[])

    assert len(attempts) == 2
    assert response["choices"][0]["message"]["content"] == "topic: Space"
