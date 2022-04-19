# Tips & Tricks


## **Document length**
As a default, we are using sentence-transformers to embed our documents. However, as the name implies, the embedding model works best for either sentences or paragraphs. This means that whenever you have a set of documents, where each documents contains several paragraphs, BERTopic will struggle getting accurately extracting a topic from that document. Several paragraphs typically means several topics and BERTopic will assign only one topic to a document. 

Therefore, it is advised to split up longer documents into either sentences or paragraphs before embedding them. That way, BERTopic will have a much easier job identifying topics in isolation. 

## **Removing stop words**
At times, stop words might end up in our topic representations. This is something we typically want to avoid as they contribute little to the interpretation of the topics. However, removing stop words as a preprocessing step is not advised as the transformer-based embedding models that we use need the full context in order to create accurate embeddings. 

Instead, we can use the `CountVectorizer` to preprocess our documents **after** having generated embeddings and clustered 
our documents. Personally, I have found almost no disadvantages to using the `CountVectorizer` to remove stopwords and 
it is something I would strongly advise to try out:

```python
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

vectorizer_model = CountVectorizer(stop_words="english")
topic_model = BERTopic(vectorizer_model=vectorizer_model)
```

## **Diversify topic representation**
After having calculated our top *n* words per topic there might be many words that essentially 
mean the same thing. As a little bonus, we can use the `diversity` parameter in BERTopic to 
diversity words in each topic such that we limit the number of duplicate words we find in each topic. 
This is done using an algorithm called Maximal Marginal Relevance which compares word embeddings 
with the topic embedding. 

We do this by specifying a value between 0 and 1, with 0 being not at all diverse and 1 being completely diverse:

```python
from bertopic import BERTopic
topic_model = BERTopic(diversity=0.2)
```

Since MMR is using word embeddings to diversify the topic representations, it is necessary to pass the embedding model to BERTopic if you are using pre-computed embeddings:
    
```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = sentence_model.encode(docs, show_progress_bar=False)
topic_model = BERTopic(embedding_model=sentence_model, diversity=0.2)
```


## **Topic-term matrix**
Although BERTopic focuses on clustering our documents, the end result does contain a topic-term matrix. 
This topic-term matrix is calculated using c-TF-IDF, a TF-IDF procedure optimized for class-based analyses. 

To extract the topic-term matrix (or c-TF-IDF matrix) with the corresponding words, we can simply do the following:

```python
topic_term_matrix = topic_model.c_tf_idf
words = topic_model.vectorizer_model.get_feature_names()
```

!!! note
    This only works if you have set `diversity=None`, for all other values the top *n* are 
    further optimized using MMR which is not represented in the topic-term matrix as it does 
    not optimize the entire matrix. 


## **Pre-compute embeddings**
Typically, we want to iterate fast over different versions of our BERTopic model whilst we are trying to optimize it to a specific use case. To speed up this process, we can pre-compute the embeddings, save them, 
and pass them to BERTopic so it does not need to calculate the embeddings each time: 

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

## **Speed up UMAP**
At times, UMAP may take a while to fit on the embeddings that you have. This often happens when you have
the embeddings millions of documents that you want to reduce in dimensionality. There is a trick that 
can speed up this process somewhat: Initializing UMAP with rescaled PCA embeddings. 

Without going in too much detail (look [here](https://github.com/lmcinnes/umap/issues/771#issuecomment-931886015) for more information), you can reduce the embeddings using PCA 
and use that as a starting point. This can speed up the dimensionality reduction a bit: 


```python
import numpy as np
from umap import UMAP
from bertopic import BERTopic
from sklearn.decomposition import PCA


def rescale(x, inplace=False):
    """ Rescale an embedding so optimization will not have convergence issues.
    """
    if not inplace:
        x = np.array(x, copy=True)

    x /= np.std(x[:, 0]) * 10000

    return x


# Initialize and rescale PCA embeddings
pca_embeddings = rescale(PCA(n_components=5).fit_transform(embeddings))

# Start UMAP from PCA embeddings
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    init=pca_embeddings,
)

# Pass the model to BERTopic:
topic_model = BERTopic(umap_model=umap_model)
```