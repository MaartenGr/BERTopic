# Embedding Models
In this tutorial we will be going through the embedding models that can be used in BERTopic. Having the option to choose embedding models allow you to leverage pre-trained embeddings that suit your use-case. Moreover, it helps creating a topic when you have little data to your availability.

### **Sentence Transformers**
You can select any model from sentence-transformers [here](https://www.sbert.net/docs/pretrained_models.html) 
and pass it through BERTopic with `embedding_model`:

```python
from bertopic import BERTopic
topic_model = BERTopic(embedding_model="xlm-r-bert-base-nli-stsb-mean-tokens")
```

Or select a SentenceTransformer model with your own parameters:

```python
from sentence_transformers import SentenceTransformer

sentence_model = SentenceTransformer("distilbert-base-nli-mean-tokens", device="cuda")
topic_model = BERTopic(embedding_model=sentence_model)
```

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
 
 allows you to choose almost any embedding model that 
is publicly available. Flair can be used as follows:

To use Spacy's non-transformer models in BERTopic:

```python
import spacy

nlp = spacy.load("en_core_web_md", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])

topic_model = BERTopic(embedding_model=nlp)
```

Using spacy-transformer models:

```python
import spacy

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])

topic_model = BERTopic(embedding_model=nlp)
```

If you run into memory issues with spacy-transformer models, try:

```python
import spacy
from thinc.api import set_gpu_allocator, require_gpu

nlp = spacy.load("en_core_web_trf", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
set_gpu_allocator("pytorch")
require_gpu(0)

topic_model = BERTopic(embedding_model=nlp)
```

### **Universal Sentence Encoder (USE)**
The Universal Sentence Encoder encodes text into high dimensional vectors that are used here 
for embedding the documents. The model is trained and optimized for greater-than-word length text, 
such as sentences, phrases or short paragraphs.

Using USE in BERTopic is rather straightforward:

```python
import tensorflow_hub
embedding_model = tensorflow_hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
topic_model = BERTopic(embedding_model=embedding_model)
```

### **Gensim**
For Gensim, BERTopic supports its `gensim.downloader` module. Here, we can download any model word embedding model 
to be used in BERTopic. Note that Gensim is primarily used for Word Embedding models. This works typically 
best for short documents since the word embeddings are pooled.

```python
import gensim.downloader as api
ft = api.load('fasttext-wiki-news-subwords-300')
topic_model = BERTopic(embedding_model=ft)
```

## **Customization**
Over the last years, many new embedding models have been released that could be interesting to use as a 
backend in BERTopic. It is not always feasible to implement them all as there are simply too many to follow.

In order to still allow to use those embeddings, BERTopic knows several ways to add these embeddings while still 
allowing for full functionality of BERTopic.

Moreover, there are several customization options that allow for a bit more control over which embedding to use when.

### **Word + Document Embeddings**
You might want to be using different language models for creating document- and word-embeddings. For example, 
while SentenceTransformers might be great in embedding sentences and documents, you might prefer to use 
FastText to create the word embeddings.

```python
from bertopic.backend import WordDocEmbedder
import gensim.downloader as api
from sentence_transformers import SentenceTransformer

# Word embedding model
ft = api.load('fasttext-wiki-news-subwords-300')

# Document embedding model
distilbert = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")

# Create a model that uses both language models and pass it through BERTopic
word_doc_embedder = WordDocEmbedder(embedding_model=distilbert, word_embedding_model=ft)
topic_model = BERTopic(embedding_model=word_doc_embedder)
```

### **Custom Backend**
If your backend or model cannot be found in the ones currently available, you can use the `bertopic.backend.BaseEmbedder` class to 
create your own backend. Below, you will find an example of creating a SentenceTransformer backend for BERTopic:

```python
from bertopic.backend import BaseEmbedder
from sentence_transformers import SentenceTransformer

class CustomEmbedder(BaseEmbedder):
    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model

    def embed(self, documents, verbose=False):
        embeddings = self.embedding_model.encode(documents, show_progress_bar=verbose)
        return embeddings 

# Create custom backend
distilbert = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")
custom_embedder = CustomEmbedder(embedding_model=distilbert)

# Pass custom backend to bertopic
topic_model = BERTopic(embedding_model=custom_embedder)
```

### **Custom Embeddings**
The base models in BERTopic are BERT-based models that work well with document similarity tasks. Your documents, 
however, might be too specific for a general pre-trained model to be used. Fortunately, you can use embedding 
model in BERTopic in order to create document features.   

You only need to prepare the document embeddings yourself and pass them through `fit_transform` of BERTopic:
```python
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer

# Prepare embeddings
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
sentence_model = SentenceTransformer("distilbert-base-nli-mean-tokens")
embeddings = sentence_model.encode(docs, show_progress_bar=False)

# Create topic model and use the custom embeddings
topic_model = BERTopic()
topics, _ = topic_model.fit_transform(docs, embeddings)
```

As you can see above, we used a SentenceTransformer model to create the embedding. You could also have used 
`🤗 transformers`, `Doc2Vec`, or any other embedding method. 

#### **TF-IDF**
As mentioned above, any embedding technique can be used. However, when running umap, the typical distance metric is 
`cosine` which does not work quite well for a TF-IDF matrix. Instead, BERTopic will recognize that a sparse matrix 
is passed and use `hellinger` instead which works quite well for the similarity between probability distributions. 

We simply create a TF-IDF matrix and use them as embeddings in our `fit_transform` method:

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF sparse matrix
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
vectorizer = TfidfVectorizer(min_df=5)
embeddings = vectorizer.fit_transform(docs)

# 
topic_model = BERTopic(stop_words="english")
topics, _ = topic_model.fit_transform(docs, embeddings)
```

Here, you will probably notice that creating the embeddings is quite fast whereas `fit_transform` is quite slow. 
This is to be expected as reducing dimensionality of a large sparse matrix takes some time. The inverse of using 
transformer embeddings is true: creating the embeddings is slow whereas `fit_transform` is quite fast. 

You can play around with different models until you find the best suiting model for you.   
