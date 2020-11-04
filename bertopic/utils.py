import numpy as np
import logging
from collections.abc import Iterable


def check_documents_type(documents):
    """ Check whether the input documents are indeed a list of strings """
    if isinstance(documents, Iterable) and not isinstance(documents, str):
        if not any([isinstance(doc, str) for doc in documents]):
            raise TypeError("Make sure that the iterable only contains strings.")

    else:
        raise TypeError("Make sure that the documents variable is an iterable containing strings only.")


def check_embeddings_shape(embeddings, docs):
    """ Check if the embeddings have the correct shape """
    if embeddings is not None:
        if not isinstance(embeddings, np.ndarray):
            raise ValueError("Make sure to input embeddings as a numpy array. ")
        else:
            if embeddings.shape[0] != len(docs):
                raise ValueError("Make sure that the embeddings are a numpy array with shape: "
                                 "(len(docs), vector_dim) where vector_dim is the dimensionality "
                                 "of the vector embeddings. ")


def create_logger():
    """ Initialize logger """
    logger = logging.getLogger('BERTopic')
    logger.setLevel(logging.WARNING)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
    logger.addHandler(sh)
    return logger


def sentence_models():
    """ All models, as of 23/09/2020, pretrained for sentence transformers """
    models = ['average_word_embeddings_glove.6B.300d.zip',
              'average_word_embeddings_glove.840B.300d.zip',
              'average_word_embeddings_komninos.zip',
              'average_word_embeddings_levy_dependency.zip',
              'bert-base-nli-cls-token.zip',
              'bert-base-nli-max-tokens.zip',
              'bert-base-nli-mean-tokens.zip',
              'bert-base-nli-stsb-mean-tokens.zip',
              'bert-base-nli-stsb-wkpooling.zip',
              'bert-base-nli-wkpooling.zip',
              'bert-base-wikipedia-sections-mean-tokens.zip',
              'bert-large-nli-cls-token.zip',
              'bert-large-nli-max-tokens.zip',
              'bert-large-nli-mean-tokens.zip',
              'bert-large-nli-stsb-mean-tokens.zip',
              'distilbert-base-nli-max-tokens.zip',
              'distilbert-base-nli-mean-tokens.zip',
              'distilbert-base-nli-stsb-mean-tokens.zip',
              'distilbert-base-nli-stsb-quora-ranking.zip',
              'distilbert-base-nli-stsb-wkpooling.zip',
              'distilbert-base-nli-wkpooling.zip',
              'distilbert-multilingual-nli-stsb-quora-ranking.zip',
              'distiluse-base-multilingual-cased.zip',
              'roberta-base-nli-mean-tokens.zip',
              'roberta-base-nli-stsb-mean-tokens.zip',
              'roberta-base-nli-stsb-wkpooling.zip',
              'roberta-base-nli-wkpooling.zip',
              'roberta-large-nli-mean-tokens.zip',
              'roberta-large-nli-stsb-mean-tokens.zip',
              'xlm-r-100langs-bert-base-nli-mean-tokens.zip',
              'xlm-r-100langs-bert-base-nli-stsb-mean-tokens.zip',
              'xlm-r-base-en-ko-nli-ststb.zip',
              'xlm-r-large-en-ko-nli-ststb.zip']
    return models
