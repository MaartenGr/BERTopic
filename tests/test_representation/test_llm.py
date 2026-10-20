import copy
import sys
import types

import pytest
from transformers.pipelines.base import Pipeline

from bertopic.representation import LangChain, TextGeneration
from bertopic.representation._prompts import DEFAULT_CHAT_PROMPT


@pytest.mark.parametrize(
    "name, build",
    [
        ("_openai", lambda module, **kwargs: module.OpenAI(None, **kwargs)),
        ("_ollama", lambda module, **kwargs: module.Ollama("gemma4:e4b-it-qat", **kwargs)),
        ("_cohere", lambda module, **kwargs: module.Cohere(None, **kwargs)),
        ("_litellm", lambda module, **kwargs: module.LiteLLM("gpt-6-luna", **kwargs)),
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
    json_schema = module.OpenAI(None, json_schema=True).generator_kwargs["response_format"]["json_schema"]
    assert json_schema["strict"]
    assert json_schema["schema"]["additionalProperties"] is False


@Pipeline.register
class Continuation:
    """A text-generation pipeline that continues every prompt with the same label."""

    tokenizer = types.SimpleNamespace(chat_template=None)

    def __call__(self, prompt, **kwargs):
        return [{"generated_text": f"{prompt} space exploration"}]


def test_text_generation_prompts_with_keywords_only(kmeans_pca_topic_model, documents):
    topic_model = copy.deepcopy(kmeans_pca_topic_model)
    representation_model = TextGeneration(Continuation())
    topic_model.update_topics(documents, representation_model=representation_model)

    # A completion model such as gpt2 reads 1,024 positions, so the default prompt leaves the documents out
    assert all(len(prompt.split()) < 100 for prompt in representation_model.prompts_)
    assert {topic_model.get_topic(topic)[0][0] for topic in topic_model.topic_sizes_} == {"space exploration"}


@Pipeline.register
class ChatPipeline:
    """A text-generation pipeline for a chat model, which answers a conversation as transformers does."""

    tokenizer = types.SimpleNamespace(chat_template="{{ messages }}")

    def __init__(self, answer="Space"):
        self.answer = answer
        self.conversations = []

    def __call__(self, messages, return_full_text=True, **kwargs):
        self.conversations.append(messages)

        # The answer is a message only within the full conversation, and an empty one has no content
        reply = {"role": "assistant", "content": self.answer} if self.answer else {"role": "assistant"}
        return [{"generated_text": [*messages, reply] if return_full_text else self.answer}]


@pytest.mark.parametrize("pipeline_kwargs", [{}, {"return_full_text": False}])
def test_text_generation_sends_a_chat_model_messages(pipeline_kwargs, kmeans_pca_topic_model, documents):
    topic_model = copy.deepcopy(kmeans_pca_topic_model)
    generator = ChatPipeline()
    representation_model = TextGeneration(generator, pipeline_kwargs=pipeline_kwargs)
    topic_model.update_topics(documents, representation_model=representation_model)

    # A chat model is asked like the other LLMs, and the pipeline writes the prompt in its chat template
    assert representation_model.prompt == DEFAULT_CHAT_PROMPT
    sent = [conversation[0]["content"] for conversation in generator.conversations]
    assert sent == representation_model.prompts_
    assert {topic_model.get_topic(topic)[0][0] for topic in topic_model.topic_sizes_} == {"Space"}


def test_text_generation_marks_a_chat_answer_without_a_label(kmeans_pca_topic_model, documents):
    topic_model = copy.deepcopy(kmeans_pca_topic_model)
    with pytest.warns(UserWarning, match="no label"):
        topic_model.update_topics(documents, representation_model=TextGeneration(ChatPipeline(answer="")))
    assert {topic_model.get_topic(topic)[0][0] for topic in topic_model.topic_sizes_} == {"No label returned"}


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
    response = module.chat_completions_with_backoff(model="gpt-6-luna", messages=[])

    assert len(attempts) == 2
    assert response["choices"][0]["message"]["content"] == "topic: Space"


@Pipeline.register
class StandInLLM:
    """Answers each prompt in the shape every client library uses, and records the prompts it was sent.

    `answer` maps a prompt's position to its answer, so a test can leave one topic without a label.
    """

    tokenizer = types.SimpleNamespace(chat_template=None)

    def __init__(self, answer=lambda index: "Space"):
        self.answer = answer
        self.prompts = []

    def _reply(self, prompt):
        self.prompts.append(prompt)
        return self.answer(len(self.prompts) - 1)

    @property
    def openai_client(self):
        """An OpenAI client whose chat.completions.create answers through this stand-in."""
        completions = types.SimpleNamespace(create=self.openai_create)
        return types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))

    def openai_create(self, messages, **kwargs):
        message = types.SimpleNamespace(content=self._reply(messages[-1]["content"]))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    def chat(self, messages, **kwargs):  # Cohere
        text = types.SimpleNamespace(text=self._reply(messages[-1]["content"]))
        return types.SimpleNamespace(message=types.SimpleNamespace(content=[text]))

    def ollama_chat(self, messages, **kwargs):
        message = types.SimpleNamespace(content=self._reply(messages[-1]["content"]))
        return types.SimpleNamespace(message=message)

    def completion(self, messages, **kwargs):  # LiteLLM, and llama.cpp's create_chat_completion
        return {"choices": [{"message": {"content": self._reply(messages[-1]["content"])}}]}

    create_chat_completion = completion

    def invoke(self, messages, **kwargs):  # LangChain
        return types.SimpleNamespace(content=self._reply(messages[-1]["content"]))

    def with_structured_output(self, schema):  # LangChain with a JSON schema
        return types.SimpleNamespace(
            invoke=lambda messages, **kwargs: {"topic_label": self.invoke(messages).content}
        )

    def __call__(self, prompt, **kwargs):  # a transformers pipeline
        return [{"generated_text": prompt + self._reply(prompt)}]


LLM_REPRESENTATIONS = ["OpenAI", "Ollama", "Cohere", "LiteLLM", "LlamaCPP", "LangChain", "TextGeneration"]


def build(name, load, llm, **kwargs):
    """An LLM representation sending its requests to the stand-in, wherever its library would send them."""
    module = load(f"bertopic.representation._{name.lower()}")
    vars(module).update(chat=llm.ollama_chat, completion=llm.completion, Llama=StandInLLM)
    clients = {"OpenAI": llm.openai_client, "Ollama": "gemma4:e4b-it-qat", "LiteLLM": "gpt-6-luna"}
    return getattr(module, name)(clients.get(name, llm), **kwargs)


@pytest.mark.parametrize(
    "name, kwargs", [(name, {}) for name in LLM_REPRESENTATIONS] + [("LangChain", {"json_schema": True})]
)
def test_every_llm_representation_labels_its_topics(
    name, kwargs, stand_in_clients, kmeans_pca_topic_model, documents
):
    llm = StandInLLM()
    representation_model = build(name, stand_in_clients, llm, **kwargs)
    topic_model = copy.deepcopy(kmeans_pca_topic_model)
    topic_model.update_topics(documents, representation_model=representation_model)

    # Each topic is asked for once, with the prompt that was built for it, and labelled with the answer
    assert llm.prompts == representation_model.prompts_
    assert len(llm.prompts) == len(topic_model.topic_sizes_)
    assert {topic_model.get_topic(topic)[0][0] for topic in topic_model.topic_sizes_} == {"Space"}


@pytest.mark.parametrize("name", LLM_REPRESENTATIONS)
def test_an_answer_without_a_label_is_marked(name, stand_in_clients, kmeans_pca_topic_model, documents):
    # The first topic's answer holds nothing, as when a provider withholds it
    llm = StandInLLM(answer=lambda index: "" if index == 0 else "Space")
    topic_model = copy.deepcopy(kmeans_pca_topic_model)
    with pytest.warns(UserWarning, match="no label"):
        topic_model.update_topics(documents, representation_model=build(name, stand_in_clients, llm))

    labels = [topic_model.get_topic(topic)[0][0] for topic in topic_model.topic_sizes_]
    assert sorted(labels) == ["No label returned"] + ["Space"] * (len(labels) - 1)


def refusal(code):
    """An answer that refuses the first prompt with an OpenAI error carrying this code."""

    def answer(index):
        if index == 0:
            error = sys.modules["openai"].BadRequestError("The prompt was refused")
            error.code = code
            raise error
        return "Space"

    return answer


def test_openai_marks_a_prompt_refused_for_its_content(stand_in_clients, kmeans_pca_topic_model, documents):
    topic_model = copy.deepcopy(kmeans_pca_topic_model)
    representation_model = build("OpenAI", stand_in_clients, StandInLLM(refusal("content_filter")))
    with pytest.warns(UserWarning, match="no label"):
        topic_model.update_topics(documents, representation_model=representation_model)
    assert "No label returned" in {topic_model.get_topic(topic)[0][0] for topic in topic_model.topic_sizes_}

    # Any other bad request still stops the fit
    representation_model = build("OpenAI", stand_in_clients, StandInLLM(refusal("invalid_request")))
    with pytest.raises(sys.modules["openai"].BadRequestError):
        topic_model.update_topics(documents, representation_model=representation_model)


def test_documents_are_fetched_only_when_the_prompt_uses_them(
    stand_in_clients, kmeans_pca_topic_model, documents, monkeypatch
):
    topic_model = copy.deepcopy(kmeans_pca_topic_model)
    representation_model = build(
        "OpenAI", stand_in_clients, StandInLLM(), prompt="Keywords: [KEYWORDS]", nr_docs=7
    )
    extract, requested = topic_model._extract_representative_docs, []

    def recording_extract(**kwargs):
        requested.append(kwargs["nr_repr_docs"])
        return extract(**kwargs)

    monkeypatch.setattr(topic_model, "_extract_representative_docs", recording_extract)
    topic_model.update_topics(documents, representation_model=representation_model)

    # Only BERTopic's own representative documents were extracted, not seven per topic for the prompt
    assert 7 not in requested


@pytest.mark.parametrize("name", ["Cohere", "LiteLLM", "LlamaCPP"])
def test_a_response_format_without_a_schema_is_kept(name, stand_in_clients):
    response_format = {"type": "text"}
    representation_model = build(
        name, stand_in_clients, StandInLLM(), generator_kwargs={"response_format": response_format}
    )
    assert representation_model.generator_kwargs["response_format"] == response_format
