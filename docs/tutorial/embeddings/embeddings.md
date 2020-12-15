## **Transformer Models**
The base models in BERTopic are both BERT-based models that work well with document similarity tasks. You documents, 
however, might be too specific for a general pre-trained model to be used. Fortunately, you can use embedding 
model in BERTopic in order to create document features.   

You only need to prepare the document embeddings yourself and pass them through `fit_transform` of BERTopic:
```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer

# Prepare embeddings
docs = fetch_20newsgroups(subset='all')['data']
sentence_model = SentenceTransformer("distilbert-base-nli-mean-tokens")
embeddings = sentence_model.encode(docs, show_progress_bar=False)

# Create topic model
model = BERTopic()
topics, probabilities = model.fit_transform(docs, embeddings)
```

As you can see above, we used a SentenceTransformer model to create the embedding. You could also have used 
`🤗 transformers`, `Doc2Vec`, or any other embedding method. 

Due to the stochastisch nature of UMAP, the results from BERTopic might differ even if you run the same code
multiple times. Using your own embeddings allows you to try out BERTopic several times until you find the 
topics that suit you best. You only need to generate the embeddings itself once and run BERTopic several times
with different parameters. 


## **TF-IDF**
As mentioned above, any embedding technique can be used. However, when running umap, the typical distance metric is 
`cosine` which does not work quite well for a TF-IDF matrix. Instead, BERTopic will recognize that a sparse matrix 
is passed and use `hellinger` instead which works quite well for the similarity between probability distributions. 

We simply create a TF-IDF matrix and use them as embeddings in our `fit_transform` method:

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF sparse matrix
docs = fetch_20newsgroups(subset='all')['data']
vectorizer = TfidfVectorizer(min_df=5)
embeddings = vectorizer.fit_transform(docs)

# 
model = BERTopic(stop_words="english")
topics, probabilities = model.fit_transform(docs, embeddings)
```

Here, you will probably notice that creating the embeddings is quite fast whereas `fit_transform` is quite slow. 
This is to be expected as reducing dimensionality of a large sparse matrix takes some time. The inverse of using 
transformer embeddings is true: creating the embeddings is slow whereas `fit_transform` is quite fast. 

You can play around with different models until you find the best suiting model for you.   
