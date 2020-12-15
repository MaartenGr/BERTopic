In `BERTopic`, there are several arguments that might be helpful if you tend to end up with too many or too few 
topics. 

## **Topic Parameters**
The arguments discussed here all relate to the cluster step of BERTopic. 

#### **Minimum topic size**
The `min_topic_size` parameter is actually used in `HDBSCAN`. It tells HDBSCAN what the minimum size of a cluster 
should be before it is accepted as a cluster. When you set this parameter very high, you will get very little clusters 
as they all need to be high. In constrast, if you set this too low you might end with too many extremely specific 
clusters. 

```python
from bertopic import BERTopic
model = BERTopic(min_topic_size=10)
```

You can increase this value if you have more data available or if you expect clusters to be quite large. 

#### **Local Neighborhood**
The `n_neighbors` parameter is used in `UMAP` when reducing the dimensionality. It is the size of the local 
neighborhood used for manifold approximation. If we set this relatively high, we get a more global view of 
the data which might reduce the number of clusters. Smaller values result in more local data being preseverd which 
could result in more clusters.

```python
from bertopic import BERTopic
model = BERTopic(n_neighbors=15)
```

If you have more data, you can increase the value of this parameter as you are more likely to have more neighbors. 
 
#### **Dimensionality**
The `n_components` refers to the dimension size we reduce the document embeddings to. This is necessary for HDBSCAN 
to properly find clusters. A higher value will preserve more local structure but makes clustering more complicated 
for HDBSCAN which can result in fewer clusters if set to high. A small value will preserve less of the local structure 
but makes clustering easier for HDBSCAN. Similarly, this can result in fewer clusters if set to low.

```python
from bertopic import BERTopic
model = BERTopic(n_components=5)
```
 
I would recommend a value between 3 and 10 dimensions.  

## **Hierarchical Topic Reduction**
It is not possible for HDBSCAN to specify the number of clusters you would want. To a certain extent, 
this is actually an advantage, as we can trust HDBSCAN to be better in finding the number of clusters than we ware.

Instead, we can try to reduce the number of topics after they have been created. Each resulting topic has its own 
feature vector constructed from c-TF-IDF. Using those feature vectors, we can find the most similar 
topics and merge them. If we do this iteratively, starting from the least frequent topic, we can reduce the number 
of topics quite easily. We do this until we reach the value of `nr_topics`:  

```python
from bertopic import BERTopic
model = BERTopic(nr_topics=20)
```

One issue with this approach is that it will merge topics regardless of whether they are actually very similar. They 
are simply the most similar out of all options. This can be resolved by reducing the number of topics automatically. 
It will reduce the number of topics, starting from the least frequent topic, as long as it exceeds a minimum 
similarity of 0.9. To use this option, we simply set `nr_topics` to `"auto"`:

```python
from bertopic import BERTopic
model = BERTopic(nr_topics="auto")
```




