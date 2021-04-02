from ._base import BaseEmbedder
from sentence_transformers import SentenceTransformer


class SentenceTransformerBackend(BaseEmbedder):
    def __init__(self, embedding_model):
        super().__init__()

        if isinstance(embedding_model, SentenceTransformer):
            self.embedding_model = embedding_model
        elif isinstance(embedding_model, str):
            self.embedding_model = SentenceTransformer(embedding_model)

    def embed(self, documents, verbose):
        embeddings = self.embedding_model.encode(documents, show_progress_bar=verbose)
        return embeddings


languages = ['afrikaans', 'albanian', 'amharic', 'arabic', 'armenian', 'assamese',
             'azerbaijani', 'basque', 'belarusian', 'bengali', 'bengali romanize',
             'bosnian', 'breton', 'bulgarian', 'burmese', 'burmese zawgyi font', 'catalan',
             'chinese (simplified)', 'chinese (traditional)', 'croatian', 'czech', 'danish',
             'dutch', 'english', 'esperanto', 'estonian', 'filipino', 'finnish', 'french',
             'galician', 'georgian', 'german', 'greek', 'gujarati', 'hausa', 'hebrew', 'hindi',
             'hindi romanize', 'hungarian', 'icelandic', 'indonesian', 'irish', 'italian', 'japanese',
             'javanese', 'kannada', 'kazakh', 'khmer', 'korean', 'kurdish (kurmanji)', 'kyrgyz',
             'lao', 'latin', 'latvian', 'lithuanian', 'macedonian', 'malagasy', 'malay', 'malayalam',
             'marathi', 'mongolian', 'nepali', 'norwegian', 'oriya', 'oromo', 'pashto', 'persian',
             'polish', 'portuguese', 'punjabi', 'romanian', 'russian', 'sanskrit', 'scottish gaelic',
             'serbian', 'sindhi', 'sinhala', 'slovak', 'slovenian', 'somali', 'spanish', 'sundanese',
             'swahili', 'swedish', 'tamil', 'tamil romanize', 'telugu', 'telugu romanize', 'thai',
             'turkish', 'ukrainian', 'urdu', 'urdu romanize', 'uyghur', 'uzbek', 'vietnamese',
             'welsh', 'western frisian', 'xhosa', 'yiddish']
