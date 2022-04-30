One important aspect of BERTopic is dimensionality reduction of the embeddings. Typically, embeddings are at least 384 in length and
many clustering algorithms have difficulty clustering in such a high dimensional space. A solution is to reduce the dimensionality 
of the embeddings to a workable dimensional space (e.g., 5) for clustering algorithms to work with. 

In BERTopic, we typically use UMAP as it is able to capture both the local and global high-dimensional space in lower dimensions. 
However, there are other solutions out there, such as PCA that users might be interested in trying out. 

We have seen that developments in the artificial intelligence fields are quite fast and that whatever mights be state-of-the-art now, 
could be different a year or even months later. Therefore, BERTopic allows you to use any dimensionality reduction algorithm that 
you would like to be using. 

As a result, the `umap_model` parameter in BERTopic now allows for a variety of dimensionality reduction models. To do so, the class should have 
the following attributes:
* `.fit(X)` 
    * A function that can be used to fit the model
* `.transform(X)` 
    * A transform function that transforms the input to a lower dimensional size

In other words, it should have the following structure:

```python
class DimensionalityReduction:
    def fit(self, X):
        return self
    
    def transform(self, X):
        return X
```

In this tutorial, I will show you how to use several dimensionality reduction algorithms in BERTopic. 


## **UMAP**
As a default, BERTopic uses UMAP to perform its dimensionality reduction. To use a UMAP model with custom parameters, 
we simply define it and pass it to BERTopic:

```python
from bertopic import BERTopic
from umap import UMAP

umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
topic_model = BERTopic(umap_model=umap_model)
```

Here, we can define any parameters in UMAP to optimize for the best performance based on whatever validation metrics that you are using. 

## **PCA**
Although UMAP works quite well in BERTopic and is typically advised, you might want to be using PCA instead. It can be faster to train and to perform
inference with. To use PCA, we can simply import it from `sklearn` and pass it to the `umap_model` parameter:


```python
from bertopic import BERTopic
from sklearn.decomposition import PCA

dim_model = PCA(n_components=5)
topic_model = BERTopic(umap_model=dim_model)
```

As a small note, PCA and k-Means have worked quite well in my experiments and might be interesting to use instead of PCA and HDBSCAN. 


!!! note
    As you might have noticed, the `dim_model` is passed to `umap_model` which might be a bit confusing considering 
    you are not passing a UMAP model. For now, the name of the parameter is kept the same to adhere to the current 
    state of the API. Changing the name could lead to deprecation issues, which I want to prevent as much as possible. 

## **Truncated SVD**
Like PCA, there are a bunch more dimensionality reduction techniques in `sklearn` that you can be using. Here, we will demonstrate Truncated SVD 
but any model can be used as long as it has both a `.fit()` and `.transform()` method:


```python
from bertopic import BERTopic
from sklearn.decomposition import TruncatedSVD

dim_model = TruncatedSVD(n_components=5)
topic_model = BERTopic(umap_model=dim_model)
```