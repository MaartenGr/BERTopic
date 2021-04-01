from ._base import BaseEmbedder
from flair.embeddings import DocumentEmbeddings, TokenEmbeddings, DocumentPoolEmbeddings
from flair.data import Sentence
from tqdm import tqdm
import numpy as np


class FlairBackend(BaseEmbedder):
    def __init__(self, embedding_model):
        super().__init__()

        # Flair word embeddings
        if isinstance(embedding_model, TokenEmbeddings):
            self.embedding_model = DocumentPoolEmbeddings([embedding_model])

        # Flair document embeddings + disable fine tune to prevent CUDA OOM
        # https://github.com/flairNLP/flair/issues/1719
        elif isinstance(embedding_model, DocumentEmbeddings):
            if "fine_tune" in embedding_model.__dict__:
                embedding_model.fine_tune = False
            self.embedding_model = embedding_model

        else:
            raise ValueError("Please select a correct Flair model by either using preparing a token or document "
                             "embedding model: \n"
                             "`from flair.embeddings import TransformerDocumentEmbeddings`"
                             "`roberta = TransformerDocumentEmbeddings('roberta-base')`")

    def embed(self, documents, verbose):
        embeddings = []
        for index, document in tqdm(enumerate(documents), disable=not verbose):
            try:
                sentence = Sentence(document) if document else Sentence("an empty document")
                self.embedding_model.embed(sentence)
            except RuntimeError:
                sentence = Sentence("an empty document")
                self.embedding_model.embed(sentence)
            embedding = sentence.embedding.detach().cpu().numpy()
            embeddings.append(embedding)
        embeddings = np.asarray(embeddings)
        return embeddings
