## **Sentence Transformers**
You can select any model from sentence-transformers [here](https://www.sbert.net/docs/pretrained_models.html) 
and pass it through BERTopic with `embedding_model`:

```python
from bertopic import BERTopic
topic_model = BERTopic(embedding_model="xlm-r-bert-base-nli-stsb-mean-tokens")
```

Or select a SentenceTransformer model with your own parameters:

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

sentence_model = SentenceTransformer("distilbert-base-nli-mean-tokens", device="cuda")
topic_model = BERTopic(embedding_model=sentence_model)
```

## **Flair**
[Flair](https://github.com/flairNLP/flair) allows you to choose almost any embedding model that 
is publicly available. Flair can be used as follows:

```python
from bertopic import BERTopic
from flair.embeddings import TransformerDocumentEmbeddings

roberta = TransformerDocumentEmbeddings('roberta-base')
topic_model = BERTopic(embedding_model=roberta)
```

You can select any 🤗 transformers model [here](https://huggingface.co/models).

Moreover, you can also use Flair to use word embeddings and pool them to create document embeddings. 
Under the hood, Flair simply averages all word embeddings in a document. Then, we can easily 
pass it to BERTopic in order to use those word embeddings as document embeddings: 

```python
from bertopic import BERTopic
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings

glove_embedding = WordEmbeddings('crawl')
document_glove_embeddings = DocumentPoolEmbeddings([glove_embedding])

topic_model = BERTopic(embedding_model=document_glove_embeddings)
```

## **Custom Embeddings**
The base models in BERTopic are both BERT-based models that work well with document similarity tasks. You documents, 
however, might be too specific for a general pre-trained model to be used. Fortunately, you can use embedding 
model in BERTopic in order to create document features.   

You only need to prepare the document embeddings yourself and pass them through `fit_transform` of BERTopic:
```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer

# Prepare embeddings
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
sentence_model = SentenceTransformer("distilbert-base-nli-mean-tokens")
embeddings = sentence_model.encode(docs, show_progress_bar=False)

# Create topic model
topic_model = BERTopic()
topics, probabilities = topic_model.fit_transform(docs, embeddings)
```

As you can see above, we used a SentenceTransformer model to create the embedding. You could also have used 
`🤗 transformers`, `Doc2Vec`, or any other embedding method. 

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
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
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
