from bertopic.backend._base import BaseEmbedder
from bertopic.backend._sentencetransformers import SentenceTransformerBackend, languages

from bertopic._utils import NotInstalled

try:
    from ._flair import FlairBackend
except ModuleNotFoundError as e:
    FlairBackend = NotInstalled("Flair", "flair")

try:
    from ._spacy import SpacyBackend
except (ModuleNotFoundError, ImportError) as e:
    SpacyBackend = NotInstalled("Spacy", "spacy")

try:
    from ._use import USEBackend
except (ModuleNotFoundError, ImportError) as e:
    USEBackend = NotInstalled("USE", "use")

try:
    from ._gensim import GensimBackend
except (ModuleNotFoundError, ImportError) as e:
    GensimBackend = NotInstalled("Gensim", "gensim")

__all__ = [
    "BaseEmbedder",
    "SentenceTransformerBackend",
    "FlairBackend",
    "SpacyBackend",
    "USEBackend",
    "GensimBackend",
    "languages"
]
