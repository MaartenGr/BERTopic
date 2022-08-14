
import pytest
import numpy as np
from bertopic._mmr import mmr


@pytest.mark.parametrize("words,diversity",
                         [(['stars', 'star', 'starry', 'astronaut', 'astronauts'], 0),
                          (['stars', 'spaceship', 'nasa', 'skies', 'sky'], 1)])
def test_mmr(words, diversity):
    """ Test MMR

    Testing both low and high diversity when selecing candidates.
    In the parameters, you can see that low diversity leads to very
    similar words/vectors to be selected, whereas a high diversity
    leads to a selection of candidates that, albeit similar to the input
    document, are less similar to each other.
    """
    candidates = mmr(doc_embedding=np.array([5, 5, 5, 5]).reshape(1, -1),
                     word_embeddings=np.array([[1, 1, 2, 2],
                                               [1, 2, 4, 7],
                                               [4, 4, 4, 4],
                                               [4, 4, 4, 4],
                                               [4, 4, 4, 4],
                                               [1, 1, 9, 3],
                                               [5, 3, 5, 8],
                                               [6, 6, 6, 6],
                                               [6, 6, 6, 6],
                                               [5, 8, 7, 2]]),
                     words=['space', 'nasa', 'stars', 'star', 'starry', 'spaceship',
                            'sky', 'astronaut', 'astronauts', 'skies'],
                     diversity=diversity)
    assert candidates == words
