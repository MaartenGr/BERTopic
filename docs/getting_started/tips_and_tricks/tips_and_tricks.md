# Tips & Tricks


## **Removing stop words**
At times, stop words might end up in our topic representations. This is something we typically want to avoid as they
contribute little to the interpretation of the topics. However, removing stop words as a preprocessing step is 
not advised as the transformer-based embedding models that we use need the full context in order to create 
accurate embeddings. 

Instead, we can use the `CountVectorizer` to preprocess our documents **after** having generated embeddings and clustered 
our documents. Personally, I have found almost no disadvantages to using the `CountVectorizer` to remove stopwords and 
it is something I would strongly advise to try out:

```python
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

vectorizer_model = CountVectorizer(stop_words="english")
topic_model = BERTopic(vectorizer_model=vectorizer_model)
```

## **Pre-computing embeddings**
...

## **PCA + UMAP**
...