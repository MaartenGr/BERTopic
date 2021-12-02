Over the last years, many new embedding models have been released that could be interesting to use as a 
backend in BERTopic. Although not all might be implemented, BERTopic knows several ways to add these embeddings while still 
allowing for full functionality of BERTopic. As a result, you can use any language model as a back-end with just a few lines of code. 
Here, we will go through several ways of approaching this. 

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