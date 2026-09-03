"""Which dataframe library BERTopic hands back.

BERTopic computes in its own data classes and only builds a dataframe at the very end,
so the library it builds with is a presentation choice rather than an engine. Pandas is
the default because that is what every existing notebook expects.
"""

from typing import Literal

Backend = Literal["pandas", "polars"]

SUPPORTED_BACKENDS = ("pandas", "polars")

_output: Backend = "pandas"


def set_output(backend: Backend) -> None:
    """Choose the dataframe library that BERTopic returns.

    The chosen library has to be installed. Only pandas is a BERTopic dependency, so
    reaching for polars means installing it yourself.

    Arguments:
        backend: Either "pandas" (the default) or "polars".

    Examples:
    ```python
    import bertopic

    bertopic.set_output("polars")
    topic_model.get_topic_info()  # now a polars DataFrame
    ```
    """
    global _output
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"`backend` should be one of {SUPPORTED_BACKENDS}, not {backend!r}.")
    _output = backend


def get_output() -> Backend:
    """Get the dataframe library that BERTopic currently returns."""
    return _output
