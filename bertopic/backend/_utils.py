from bertopic.backend._base import BaseEmbedder
from bertopic.backend._sentencetransformers import SentenceTransformerBackend, languages


def select_backend(embedding_model,
                   language: str = None) -> BaseEmbedder:
    """ Select an embedding model based on language or a specific sentence transformer models.
    When selecting a language, we choose distilbert-base-nli-stsb-mean-tokens for English and
    xlm-r-bert-base-nli-stsb-mean-tokens for all other languages as it support 100+ languages.

    Returns:
        model: Either a Sentence-Transformer or Flair model
    """
    # BERTopic language backend
    if isinstance(embedding_model, BaseEmbedder):
        return embedding_model

    # Sentence Transformer embeddings
    if "sentence_transformers" in str(type(embedding_model)):
        return SentenceTransformerBackend(embedding_model)

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

    # Create a Sentence Transformer model based on a string
    if isinstance(embedding_model, str):
        return SentenceTransformerBackend(embedding_model)

    # Select embedding model based on language
    if language:
        if language.lower() in ["English", "english", "en"]:
            return SentenceTransformerBackend("distilbert-base-nli-stsb-mean-tokens")
        elif language.lower() in languages or language == "multilingual":
            return SentenceTransformerBackend("xlm-r-bert-base-nli-stsb-mean-tokens")
        else:
            raise ValueError(f"{language} is currently not supported. However, you can "
                             f"create any embeddings yourself and pass it through fit_transform(docs, embeddings)\n"
                             "Else, please select a language from the following list:\n"
                             f"{languages}")

    return SentenceTransformerBackend("xlm-r-bert-base-nli-stsb-mean-tokens")
