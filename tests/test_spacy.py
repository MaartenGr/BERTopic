import numpy as np
from bertopic.backend._spacy import SpacyBackend

def test_spacy_backend(spacy_nlp_model):
    spacy_model = SpacyBackend(spacy_nlp_model)
    output = spacy_model.embed(["sanity", "testing", "one two three"])
    assert len(output.shape) > 1, "should have rows and columns"
    assert output.shape[0] == 3, "should have 3 rows"
    assert output.shape[1] > 0, "should have a non zero number of columns"
    assert isinstance(output, np.ndarray), "should be a numpy array"
