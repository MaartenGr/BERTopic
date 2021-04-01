from bertopic._utils import NotInstalled
from ._base import BaseEmbedder
from ._sentencetransformers import SentenceTransformerBackend


try:
    from ._flair import FlairBackend
except ModuleNotFoundError as e:
    FlairBackend = NotInstalled("Flair", "flair")

try:
    from ._spacy import SpacyBackend
except (ModuleNotFoundError, ImportError) as e:
    SpacyBackend = NotInstalled("Spacy", "spacy")


__all__ = [
    "BaseEmbedder",
    "SentenceTransformerBackend",
    "FlairBackend",
    "SpacyBackend"
]
