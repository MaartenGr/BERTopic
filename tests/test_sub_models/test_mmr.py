
import pytest
import numpy as np
from bertopic._mmr import mmr


@pytest.mark.parametrize("words,diversity",
                         [(['stars', 'star', 'starry', 'astronaut', 'astronauts'], 0),
                          (['stars', 'spaceship', 'nasa', 'skies', 'sky'], 1)])
def test_mmr(words, diversity):
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
