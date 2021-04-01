import spacy
import numpy as np
from tqdm import tqdm
from thinc.api import set_gpu_allocator, require_gpu
from ._base import BaseEmbedder


class SpacyBackend(BaseEmbedder):
    def __init__(self, embedding_model):
        super().__init__()

        if "spacy" in str(type(embedding_model)):
            self.embedding_model = embedding_model
        else:
            raise ValueError("Please select a correct Spacy model by either using a string such as 'en_core_web_md' "
                             "or create a nlp model using: `nlp = spacy.load('en_core_web_md')")

        if "transformer" in self.embedding_model.component_names:
            set_gpu_allocator("pytorch")
            require_gpu(0)

    def embed(self, documents, verbose):
        if "transformer" in self.embedding_model.component_names:
            embeddings = np.array([self.embedding_model(doc)._.trf_data.tensors[-1][0] for
                                   doc in tqdm(documents, disable=not verbose)])
        else:
            embeddings = np.array([self.embedding_model(doc).vector for doc in tqdm(documents, disable=not verbose)])

        return embeddings
