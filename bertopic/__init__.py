from bertopic._bertopic import BERTopic
from bertopic._ctfidf import ClassTFIDF
from bertopic._embeddings import languages, embedding_models

__version__ = "0.4.0"

__all__ = [
    "BERTopic",
    "ClassTFIDF",
    "languages",
    "embedding_models",
]
