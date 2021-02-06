## **Custom Models**
There are three models underpinning BERTopic that are most important in creating the topics, 
namely UMAP, HDBSCAN, and CountVectorizer. The parameters of these models have been carefully 
selected to give the best results. However, there is no one-size-fits-all solution using these 
default parameters.

Therefore, BERTopic allows you to pass in any custom UMAP, HDBSCAN, and/or CountVectorizer 
with the parameters that best suit your use-case. For example, you might want to change the 
minimum document frequency in CountVectorizer or use a different distance metric in HDBSCAN or UMAP. 

To do this, simply create the instances of these models and initialize BERTopic with them:

```python
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# Prepare custom models
hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
umap_model = UMAP(n_neighbors=15, n_components=10, min_dist=0.0, metric='cosine')
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")

# Pass the custom models to BERTopic
topic_model = BERTopic(umap_model=umap_model, 
                       hdbscan_model=hdbscan_model, 
                       vectorizer_model=vectorizer_model)
```
