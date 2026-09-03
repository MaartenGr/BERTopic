from importlib.metadata import version

from bertopic._bertopic import BERTopic
from bertopic._config import get_output, set_output

__version__ = version("bertopic")

__all__ = [
    "BERTopic",
    "get_output",
    "set_output",
]
