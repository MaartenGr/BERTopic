from ._base import BaseEmbedder

# Imports for light-weight variant of BERTopic
from bertopic.backend._sklearn import SklearnEmbedder
from bertopic._utils import MyLogger
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline as ScikitPipeline

logger = MyLogger()
logger.configure("WARNING")

languages = [
    "arabic",
    "bulgarian",
    "catalan",
    "czech",
    "danish",
    "german",
    "greek",
    "english",
    "spanish",
    "estonian",
    "persian",
    "finnish",
    "french",
    "canadian french",
    "galician",
    "gujarati",
    "hebrew",
    "hindi",
    "croatian",
    "hungarian",
    "armenian",
    "indonesian",
    "italian",
    "japanese",
    "georgian",
    "korean",
    "kurdish",
    "lithuanian",
    "latvian",
    "macedonian",
    "mongolian",
    "marathi",
    "malay",
    "burmese",
    "norwegian bokmal",
    "dutch",
    "polish",
    "portuguese",
    "brazilian portuguese",
    "romanian",
    "russian",
    "slovak",
    "slovenian",
    "albanian",
    "serbian",
    "swedish",
    "thai",
    "turkish",
    "ukrainian",
    "urdu",
    "vietnamese",
    "chinese (simplified)",
    "chinese (traditional)",
]


def select_backend(embedding_model, language: str | None = None, verbose: bool = False) -> BaseEmbedder:
    """Select an embedding model based on language or a specific provided model.
    Without a model, we choose the text model of EmbeddingGemma 2, which supports 100+ languages.
    A sentence-transformers model that also embeds images, audio or video is wrapped to embed them.
    If sentence-transformers is not installed, in the case of a lightweight installation,
    a scikit-learn backend is default.

    Returns:
        model: The selected model backend.
    """
    logger.set_level("INFO" if verbose else "WARNING")

    # BERTopic language backend
    if isinstance(embedding_model, BaseEmbedder):
        return embedding_model

    # Scikit-learn backend
    if isinstance(embedding_model, ScikitPipeline):
        return SklearnEmbedder(embedding_model)

    # Flair word embeddings
    if "flair" in str(type(embedding_model)):
        from bertopic.backend._flair import FlairBackend

        return FlairBackend(embedding_model)

    # Spacy embeddings
    if "spacy" in str(type(embedding_model)):
        from bertopic.backend._spacy import SpacyBackend

        return SpacyBackend(embedding_model)

    # Gensim embeddings
    if "gensim" in str(type(embedding_model)):
        from bertopic.backend._gensim import GensimBackend

        return GensimBackend(embedding_model)

    # USE embeddings
    if "tensorflow" and "saved_model" in str(type(embedding_model)):
        from bertopic.backend._use import USEBackend

        return USEBackend(embedding_model)

    # Sentence Transformer embeddings
    if "sentence_transformers" in str(type(embedding_model)) or isinstance(embedding_model, str):
        from ._sentencetransformers import SentenceTransformerBackend

        backend = SentenceTransformerBackend(embedding_model)

        # Multimodal model
        if {"image", "audio", "video"} & set(backend.embedding_model.modalities):
            from ._multimodal import MultiModalBackend

            multimodal_backend = MultiModalBackend(backend.embedding_model)
            multimodal_backend._hf_model = backend._hf_model
            return multimodal_backend
        return backend

    # Hugging Face embeddings
    if "transformers" and "pipeline" in str(type(embedding_model)):
        from ._hftransformers import HFTransformerBackend

        return HFTransformerBackend(embedding_model)

    # Model2Vec embeddings
    if "model2vec" in str(type(embedding_model)):
        from ._model2vec import Model2VecBackend

        return Model2VecBackend(embedding_model)

    # FastEmbed word embeddings
    if "fastembed" in str(type(embedding_model)):
        from bertopic.backend._fastembed import FastEmbedBackend

        return FastEmbedBackend(embedding_model)

    # Anything else is not a model BERTopic can embed with, rather than one to silently replace
    if embedding_model is not None:
        raise TypeError(
            f"An embedding_model of type {type(embedding_model).__name__} is not supported. Pass a model name, "
            "a SentenceTransformer or a subclass of bertopic.backend.BaseEmbedder, or see the embeddings "
            "guide for the other libraries BERTopic supports."
        )

    # One model covers every supported language, so the language is only checked
    if language and language.lower() not in [*languages, "en", "multilingual"]:
        raise ValueError(
            f"{language} is currently not supported. However, you can "
            f"create any embeddings yourself and pass it through fit_transform(docs, embeddings)\n"
            "Else, please select a language from the following list:\n"
            f"{languages}"
        )

    try:
        from ._sentencetransformers import SentenceTransformerBackend

        # EmbeddingGemma 2's text model, without the image and audio encoders that only media needs
        return SentenceTransformerBackend(
            "google/embeddinggemma-2", config_kwargs={"vision_config": None, "audio_config": None}
        )

    # A ModuleNotFoundError might be a lightweight installation
    except ModuleNotFoundError as e:
        if e.name != "sentence_transformers":
            # Error occurred in a downstream module, probably not a lightweight install
            raise e
        # Whole sentence_transformers module is missing, probably a lightweight install
        if verbose:
            logger.info(
                "Automatically selecting lightweight scikit-learn embedding backend as sentence-transformers appears to not be installed."
            )
        pipe = make_pipeline(TfidfVectorizer(), TruncatedSVD(100))
        return SklearnEmbedder(pipe)
