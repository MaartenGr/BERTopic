from ._base import BaseEmbedder
from ._sentencetransformers import SentenceTransformerBackend

languages = ['afrikaans', 'albanian', 'amharic', 'arabic', 'armenian', 'assamese',
             'azerbaijani', 'basque', 'belarusian', 'bengali', 'bengali romanize',
             'bosnian', 'breton', 'bulgarian', 'burmese', 'burmese zawgyi font', 'catalan',
             'chinese (simplified)', 'chinese (traditional)', 'croatian', 'czech', 'danish',
             'dutch', 'english', 'esperanto', 'estonian', 'filipino', 'finnish', 'french',
             'galician', 'georgian', 'german', 'greek', 'gujarati', 'hausa', 'hebrew', 'hindi',
             'hindi romanize', 'hungarian', 'icelandic', 'indonesian', 'irish', 'italian', 'japanese',
             'javanese', 'kannada', 'kazakh', 'khmer', 'korean', 'kurdish (kurmanji)', 'kyrgyz',
             'lao', 'latin', 'latvian', 'lithuanian', 'macedonian', 'malagasy', 'malay', 'malayalam',
             'marathi', 'mongolian', 'nepali', 'norwegian', 'oriya', 'oromo', 'pashto', 'persian',
             'polish', 'portuguese', 'punjabi', 'romanian', 'russian', 'sanskrit', 'scottish gaelic',
             'serbian', 'sindhi', 'sinhala', 'slovak', 'slovenian', 'somali', 'spanish', 'sundanese',
             'swahili', 'swedish', 'tamil', 'tamil romanize', 'telugu', 'telugu romanize', 'thai',
             'turkish', 'ukrainian', 'urdu', 'urdu romanize', 'uyghur', 'uzbek', 'vietnamese',
             'welsh', 'western frisian', 'xhosa', 'yiddish']


def select_backend(embedding_model,
                   language: str = None) -> BaseEmbedder:
    """ Select an embedding model based on language or a specific sentence transformer models.
    When selecting a language, we choose paraphrase-MiniLM-L6-v2 for English and
    paraphrase-multilingual-MiniLM-L12-v2 for all other languages as it support 100+ languages.

    Returns:
        model: Either a Sentence-Transformer or Flair model
    """
    # BERTopic language backend
    if isinstance(embedding_model, BaseEmbedder):
        return embedding_model

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
    if "sentence_transformers" in str(type(embedding_model)):
        return SentenceTransformerBackend(embedding_model)

    # Create a Sentence Transformer model based on a string
    if isinstance(embedding_model, str):
        return SentenceTransformerBackend(embedding_model)

    # Select embedding model based on language
    if language:
        if language.lower() in ["English", "english", "en"]:
            return SentenceTransformerBackend("paraphrase-MiniLM-L6-v2")
        elif language.lower() in languages or language == "multilingual":
            return SentenceTransformerBackend("paraphrase-multilingual-MiniLM-L12-v2")
        else:
            raise ValueError(f"{language} is currently not supported. However, you can "
                             f"create any embeddings yourself and pass it through fit_transform(docs, embeddings)\n"
                             "Else, please select a language from the following list:\n"
                             f"{languages}")

    return SentenceTransformerBackend("paraphrase-MiniLM-L6-v2")
