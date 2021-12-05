# Embedding Models
In this tutorial, we will be going through the embedding models that can be used in BERTopic. 
Having the option to choose embedding models allows you to leverage pre-trained embeddings that suit your use case. 
Moreover, it helps to create a topic when you have little data available.

### **Sentence Transformers**
You can select any model from sentence-transformers [here](https://www.sbert.net/docs/pretrained_models.html) 
and pass it through BERTopic with `embedding_model`:

```python
from bertopic import BERTopic
topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2")
```

Or select a SentenceTransformer model with your parameters:

```python
from sentence_transformers import SentenceTransformer

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
topic_model = BERTopic(embedding_model=sentence_model)
```

!!! tip "Tip!"
    This embedding back-end was put here first for a reason, sentence-transformers works amazing out-of-the-box! Playing around with different models can give you great results. Also, make sure to frequently visit [this](https://www.sbert.net/docs/pretrained_models.html) page as new models are often released. 


### **Flair**
[Flair](https://github.com/flairNLP/flair) allows you to choose almost any embedding model that 
is publicly available. Flair can be used as follows:

```python
from flair.embeddings import TransformerDocumentEmbeddings

roberta = TransformerDocumentEmbeddings('roberta-base')
topic_model = BERTopic(embedding_model=roberta)
```

You can select any 🤗 transformers model [here](https://huggingface.co/models).

Moreover, you can also use Flair to use word embeddings and pool them to create document embeddings. 
Under the hood, Flair simply averages all word embeddings in a document. Then, we can easily 
pass it to BERTopic in order to use those word embeddings as document embeddings: 

```python
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings

glove_embedding = WordEmbeddings('crawl')
document_glove_embeddings = DocumentPoolEmbeddings([glove_embedding])

topic_model = BERTopic(embedding_model=document_glove_embeddings)
```

### **Spacy**
[Spacy](https://github.com/explosion/spaCy) is an amazing framework for processing text. There are 
many models available across many languages for modeling text. 
 
To use Spacy's non-transformer models in BERTopic:

```python
import spacy

nlp = spacy.load("en_core_web_md", exclude=['tagger', 'parser', 'ner', 
                                            'attribute_ruler', 'lemmatizer'])

topic_model = BERTopic(embedding_model=nlp)
```

Using spacy-transformer models:

```python
import spacy

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf", exclude=['tagger', 'parser', 'ner', 
                                             'attribute_ruler', 'lemmatizer'])

topic_model = BERTopic(embedding_model=nlp)
```

If you run into memory issues with spacy-transformer models, try:

```python
import spacy
from thinc.api import set_gpu_allocator, require_gpu

nlp = spacy.load("en_core_web_trf", exclude=['tagger', 'parser', 'ner', 
                                             'attribute_ruler', 'lemmatizer'])
set_gpu_allocator("pytorch")
require_gpu(0)

topic_model = BERTopic(embedding_model=nlp)
```

### **Universal Sentence Encoder (USE)**
The Universal Sentence Encoder encodes text into high-dimensional vectors that are used here 
for embedding the documents. The model is trained and optimized for greater-than-word length text, 
such as sentences, phrases, or short paragraphs.

Using USE in BERTopic is rather straightforward:

```python
import tensorflow_hub
embedding_model = tensorflow_hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
topic_model = BERTopic(embedding_model=embedding_model)
```

### **Gensim**
BERTopic supports the `gensim.downloader` module, which allows it to download any word embedding model supported by Gensim. 
Typically, these are Glove, Word2Vec, or FastText embeddings:

```python
import gensim.downloader as api
ft = api.load('fasttext-wiki-news-subwords-300')
topic_model = BERTopic(embedding_model=ft)
```

!!! tip "Tip!"
    Gensim is primarily used for Word Embedding models. This works typically best for short documents since the word embeddings are pooled.