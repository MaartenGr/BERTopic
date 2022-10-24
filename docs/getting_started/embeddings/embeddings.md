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

### 🤗 Hugging Face Transformers
To use a Hugging Face transformers model, load in a pipeline and point 
to any model found on their model hub (https://huggingface.co/models):

```python
from transformers.pipelines import pipeline

embedding_model = pipeline("feature-extraction", model="distilbert-base-cased")
topic_model = BERTopic(embedding_model=embedding_model)
```

!!! tip "Tip!"
    These transformers also work quite well using `sentence-transformers` which has a number of 
    optimizations tricks that make using it a bit faster. 

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
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create a model that uses both language models and pass it through BERTopic
word_doc_embedder = WordDocEmbedder(embedding_model=embedding_model, word_embedding_model=ft)
topic_model = BERTopic(embedding_model=word_doc_embedder)
```

!!! note
    The word embeddings are only created when applying MMR. In other words, 
    to use this feature, you will have to select a value for `diversity`  between 0 and 1 when
    instantiating BERTopic. 

### **Custom Backend**
If your backend or model cannot be found in the ones currently available, you can use the `bertopic.backend.BaseEmbedder` class to 
create your backend. Below, you will find an example of creating a SentenceTransformer backend for BERTopic:

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
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
custom_embedder = CustomEmbedder(embedding_model=embedding_model)

# Pass custom backend to bertopic
topic_model = BERTopic(embedding_model=custom_embedder)
```

### **Custom Embeddings**
The base models in BERTopic are BERT-based models that work well with document similarity tasks. Your documents, 
however, might be too specific for a general pre-trained model to be used. Fortunately, you can use embedding 
model in BERTopic to create document features.   

You only need to prepare the document embeddings yourself and pass them through `fit_transform` of BERTopic:
```python
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer

# Prepare embeddings
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = sentence_model.encode(docs, show_progress_bar=False)

# Train our topic model using our pre-trained sentence-transformers embeddings
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs, embeddings)
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

# Train our topic model using TF-IDF vectors
topic_model = BERTopic(stop_words="english")
topics, probs = topic_model.fit_transform(docs, embeddings)
```

Here, you will probably notice that creating the embeddings is quite fast whereas `fit_transform` is quite slow. 
This is to be expected as reducing the dimensionality of a large sparse matrix takes some time. The inverse of using 
transformer embeddings is true: creating the embeddings is slow whereas `fit_transform` is quite fast. 

#### **Scikit-Learn Embeddings**
Scikit-Learn is a framework for more than just machine learning. 
It offers many preprocessing tools, some of which can be used to create representations 
for text. Many of these tools are relatively lightweight and don't require a GPU. 
While the representations may be less expressive as many BERT models, the fact that 
it runs much faster can make it a relevant candidate to consider. 

If you have a scikit-learn compatible pipeline that you'd like to use to embed
text then you can use the `SklearnEmbedder` helper class. An example is shown 
below.

```python
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from bertopic.backend import SklearnEmbedder

pipe = make_pipeline(
    TfidfVectorizer(),
    TruncatedSVD(100)
)

sklearn_embedder = SklearnEmbedder(pipe)
topic_model = BERTopic(embedding_model=sklearn_embedder)
```
