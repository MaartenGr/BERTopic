import spacy
import numpy as np
from tqdm import tqdm
from ._base import BaseEmbedder


class SpacyBackend(BaseEmbedder):
    def __init__(self, embedding_model):
        super().__init__()

        if "spacy" in str(type(embedding_model)):
            self.embedding_model = embedding_model
        else:
            raise ValueError("Please select a correct Spacy model by either using a string such as 'en_core_web_md' "
                             "or create a nlp model using: `nlp = spacy.load('en_core_web_md')")

    def embed(self, documents, verbose):
        if "transformer" in self.embedding_model.component_names:
            embeddings = []
            for doc in tqdm(documents, position=0, leave=True, disable=not verbose):
                try:
                    embedding = self.embedding_model(doc)._.trf_data.tensors[-1][0]
                except:
                    embedding = self.embedding_model("An empty document")._.trf_data.tensors[-1][0]
                embeddings.append(embedding)
            embeddings = np.array(embeddings)
        else:
            embeddings = []
            for doc in tqdm(documents, position=0, leave=True, disable=not verbose):
                try:
                    vector = self.embedding_model(doc).vector
                except ValueError:
                    vector = self.embedding_model("An empty document").vector
                embeddings.append(vector)
            embeddings = np.array(embeddings)

        return embeddings
