import copy
import subprocess
import sys

import pytest
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.backend import MultiModalBackend
from bertopic.backend._utils import select_backend
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline


@pytest.mark.parametrize(
    "model",
    [
        ("kmeans_pca_topic_model"),
        ("base_topic_model"),
        ("custom_topic_model"),
        ("merged_topic_model"),
        ("reduced_topic_model"),
        ("online_topic_model"),
    ],
)
def test_extract_embeddings(model, request):
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    single_embedding = topic_model.embedding_model.embed_documents(["a document"])
    multiple_embeddings = topic_model.embedding_model.embed_documents(
        ["something different", "another document"]
    )
    sim_matrix = cosine_similarity(single_embedding, multiple_embeddings)[0]

    assert single_embedding.shape[0] == 1
    assert single_embedding.shape[1] == 384
    assert np.min(single_embedding) > -5
    assert np.max(single_embedding) < 5

    assert multiple_embeddings.shape[0] == 2
    assert multiple_embeddings.shape[1] == 384
    assert np.min(multiple_embeddings) > -5
    assert np.max(multiple_embeddings) < 5

    assert sim_matrix[0] < 0.5
    assert sim_matrix[1] > 0.5


@pytest.mark.parametrize(
    "model",
    [
        ("kmeans_pca_topic_model"),
        ("base_topic_model"),
        ("custom_topic_model"),
        ("merged_topic_model"),
        ("reduced_topic_model"),
        ("online_topic_model"),
    ],
)
def test_extract_embeddings_compare(model, embedding_model, request):
    docs = ["some document"]
    topic_model = copy.deepcopy(request.getfixturevalue(model))
    bertopic_embeddings = topic_model.embedding_model.embed_documents(docs)

    assert isinstance(bertopic_embeddings, np.ndarray)
    assert bertopic_embeddings.shape == (1, 384)

    sentence_embeddings = embedding_model.encode(docs, show_progress_bar=False)
    assert np.array_equal(bertopic_embeddings, sentence_embeddings)


def test_extract_incorrect_embeddings():
    with pytest.raises(ValueError):
        model = BERTopic(language="Unknown language")
        model.fit(["some document"])


def test_sklearn_embedder_with_sparse_output(documents):
    topic_model = BERTopic(
        embedding_model=make_pipeline(TfidfVectorizer()),
        umap_model=TruncatedSVD(n_components=5, random_state=42),
        hdbscan_model=KMeans(n_clusters=5, random_state=42),
    )
    topic_model.fit(documents)
    topics, _ = topic_model.transform(documents[:5])

    assert len(topics) == 5


def test_a_broken_optional_package_does_not_break_the_import():
    # openai<1.0 has no OpenAI class, which an empty module stands in for
    code = (
        "import sys, types; sys.modules['openai'] = types.ModuleType('openai'); "
        "import bertopic.backend as backend; "
        "print(type(backend.OpenAIBackend).__name__, 'no attribute' in backend.OpenAIBackend.msg)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    # The placeholder names the import error rather than asking to install what is installed
    assert result.stdout.strip() == "NotInstalled True", result.stderr[-500:]


def test_select_backend_rejects_an_unrecognised_model():
    with pytest.raises(TypeError, match="not supported"):
        select_backend(lambda documents: documents)


def test_the_default_is_the_text_model_of_embeddinggemma_2(monkeypatch):
    # Downloaded in CI on purpose: a fresh install resolves the default's dependencies anew
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    backend = select_backend(None, language="english")
    model = backend.embedding_model

    # Text only: the image and audio encoders are not loaded, and media is not accepted
    assert model[0].model.config.vision_config is None and model[0].model.config.audio_config is None
    assert not isinstance(backend, MultiModalBackend)

    # Set up for clustering, cut at 1,024 tokens since the model sets no length limit of its own, and in
    # float32 on a CPU, where its bfloat16 weights would run several times slower without hardware support
    assert model.default_prompt_name == "Clustering"
    assert model.max_seq_length == 1024
    assert next(model.parameters()).dtype == torch.float32
    assert backend.embed_documents(["a rocket launch", "a recipe for cake"]).shape == (2, 768)


def test_a_model_that_embeds_media_is_wrapped_to_embed_it(embedding_model, monkeypatch):
    assert not isinstance(select_backend(embedding_model), MultiModalBackend)

    # MiniLM stands in for a model that also takes images
    monkeypatch.setattr(SentenceTransformer, "modalities", property(lambda model: ["text", "image"]))
    backend = select_backend("sentence-transformers/all-MiniLM-L6-v2")

    assert isinstance(backend, MultiModalBackend)
    assert backend._hf_model == "sentence-transformers/all-MiniLM-L6-v2"


@pytest.mark.parametrize(
    "module, backend, default_model",
    [("_openai", "OpenAIBackend", "text-embedding-ada-002"), ("_cohere", "CohereBackend", "large")],
)
def test_embedding_backends_keep_their_own_model(module, backend, default_model, stand_in_clients):
    embedder = getattr(stand_in_clients(f"bertopic.backend.{module}"), backend)

    # A backend given another model must not pass it on to the next one
    embedder(None, "a-newer-model")
    assert embedder(None).embedding_model == default_model
